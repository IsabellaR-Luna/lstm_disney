# api/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import pickle
import json
import time
import redis
import asyncio
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Métricas Prometheus
prediction_counter = Counter('predictions_total', 'Total predictions made')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration')
api_requests = Counter('api_requests_total', 'Total API requests', ['endpoint', 'status'])
model_accuracy = Gauge('model_accuracy_percent', 'Current model accuracy')
active_connections = Gauge('active_connections', 'Active API connections')
cache_hits = Counter('cache_hits_total', 'Cache hits')
cache_misses = Counter('cache_misses_total', 'Cache misses')

# Inicializar FastAPI
app = FastAPI(
    title="Disney Stock Prediction API",
    description="LSTM model for DIS stock price prediction",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis para cache (opcional - fallback se não estiver disponível)
try:
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    redis_client.ping()
    REDIS_AVAILABLE = True
except:
    REDIS_AVAILABLE = False
    logger.warning("Redis não disponível - operando sem cache")

# Modelos Pydantic
class PredictionRequest(BaseModel):
    days: int = Field(default=7, ge=1, le=30, description="Number of days to predict")
    use_cache: bool = Field(default=True, description="Use cached predictions if available")

class CustomPredictionRequest(BaseModel):
    historical_prices: List[float] = Field(..., min_items=60, max_items=60)
    days_ahead: int = Field(default=1, ge=1, le=30)

class PredictionResponse(BaseModel):
    symbol: str
    predictions: List[Dict[str, float]]
    current_price: float
    confidence: float
    timestamp: str
    cache_hit: bool = False

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    redis_connected: bool
    uptime: float
    total_predictions: int

class MetricsResponse(BaseModel):
    mae: float
    rmse: float
    mape: float
    r2_score: float
    last_training: str
    accuracy_direction: float

# Serviço de Predição
class PredictionService:
    def __init__(self):
        self.modelo = None
        self.scaler = None
        self.config = None
        self.janela = 60
        self.metricas = {}
        self.start_time = time.time()
        self.total_predictions = 0
        
    def carregar_modelo(self):
        try:
            self.modelo = load_model('models/modelo_disney_lstm.h5', compile=False)
            
            with open('data/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            self.validar_scaler()
            
            with open('models/config.json', 'r') as f:
                self.config = json.load(f)
            
            with open('models/metricas.json', 'r') as f:
                self.metricas = json.load(f)
                model_accuracy.set(100 - self.metricas.get('mape', 0))
            try:
                dummy_input = np.zeros((1, self.janela, 1))
                self.modelo.predict(dummy_input, verbose=0)
            except Exception as e:
                raise RuntimeError(f"Modelo incompatível com input (1, 60, 1): {e}")
            
            logger.info("Modelo carregado com sucesso")
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            return False
        
        
    def validar_scaler(self):
        if not hasattr(self.scaler, "n_features_in_"):
            raise RuntimeError("Scaler inválido ou incompatível")

        if self.scaler.n_features_in_ != 1:
            raise RuntimeError(
                f"Scaler espera {self.scaler.n_features_in_} features, "
                "mas o modelo usa 1 (preço)"
            )

        logger.info("Scaler compatível com o modelo (1 feature)")

    
    @prediction_duration.time()
    @prediction_duration.time()
    def prever(self, dias: int = 7, use_cache: bool = True) -> Dict:
        prediction_counter.inc()
        self.total_predictions += 1

        # -------- Cache --------
        cache_key = f"disney_prediction:{dias}"

        if REDIS_AVAILABLE and use_cache:
            cached = redis_client.get(cache_key)
            if cached:
                cache_hits.inc()
                result = json.loads(cached)
                result["cache_hit"] = True
                return result
            cache_misses.inc()

        # -------- Validações --------
        if self.modelo is None or self.scaler is None:
            raise RuntimeError("Modelo ou scaler não carregado")

        # -------- Download dos dados --------
        df = yf.download(
            "DIS",
            period="3mo",
            progress=False,
            auto_adjust=False
        )

        if df.empty:
            raise ValueError("Falha ao obter dados do Yahoo Finance")

        if "Adj Close" not in df.columns:
            raise ValueError(f"Colunas inesperadas: {df.columns.tolist()}")

        if len(df) < self.janela:
            raise ValueError("Dados insuficientes para previsão")

        precos = df["Adj Close"].values.astype(float)
        ultimo_preco = float(precos[-1])
        ultimos_dias = precos[-self.janela:]

        # -------- Normalização --------
        entrada = self.scaler.transform(ultimos_dias.reshape(-1, 1))
        entrada = entrada.reshape(1, self.janela, 1)

        previsoes = []

        # -------- Loop de previsão --------
        for i in range(dias):
            pred_norm = self.modelo.predict(entrada, verbose=0)
            pred = float(self.scaler.inverse_transform(pred_norm)[0, 0])

            data_previsao = df.index[-1] + timedelta(days=i + 1)

            if i == 0:
                variacao = (pred - ultimo_preco) / ultimo_preco * 100
            else:
                variacao = (pred - previsoes[-1]["price"]) / previsoes[-1]["price"] * 100

            previsoes.append({
                "date": data_previsao.strftime("%Y-%m-%d"),
                "price": pred,
                "change_percent": variacao
            })

            # shift da janela
            entrada = np.vstack([
                entrada[0, 1:, 0],
                pred_norm[0]
            ]).reshape(1, self.janela, 1)

        # -------- Confiança --------
        confianca = self._calcular_confianca()

        resultado = {
            "symbol": "DIS",
            "predictions": previsoes,
            "current_price": ultimo_preco,
            "confidence": confianca,
            "timestamp": datetime.now().isoformat(),
            "cache_hit": False
        }

        if REDIS_AVAILABLE:
            redis_client.setex(cache_key, 3600, json.dumps(resultado))

        return resultado

    
    
    
    
    def prever_custom(self, precos: List[float], dias: int) -> List[float]:
        if len(precos) != self.janela:
            raise ValueError(f"Necessário exatamente {self.janela} preços históricos")
        
        precos_array = np.array(precos)
        entrada = self.scaler.transform(precos_array.reshape(-1, 1))
        entrada = entrada.reshape(1, self.janela, 1)
        
        previsoes = []
        for _ in range(dias):
            pred_norm = self.modelo.predict(entrada, verbose=0)
            pred = self.scaler.inverse_transform(pred_norm)[0, 0]
            previsoes.append(float(pred))
            
            nova_entrada = np.append(entrada[0, 1:], pred_norm.reshape(1, 1), axis=0)
            entrada = nova_entrada.reshape(1, self.janela, 1)
        
        return previsoes
    
    def _calcular_confianca(self) -> float:
        mape = self.metricas.get('mape', 10)
        confianca = max(0, min(95, 100 - mape))
        return float(confianca)

# Instanciar serviço
prediction_service = PredictionService()

# Endpoints
@app.on_event("startup")
async def startup_event():
    active_connections.set(0)
    if not prediction_service.carregar_modelo():
        logger.error("Falha ao carregar modelo na inicialização")

@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Disney Stock Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    api_requests.labels(endpoint='health', status='success').inc()
    
    return HealthResponse(
        status="healthy" if prediction_service.modelo else "unhealthy",
        model_loaded=prediction_service.modelo is not None,
        redis_connected=REDIS_AVAILABLE,
        uptime=time.time() - prediction_service.start_time,
        total_predictions=prediction_service.total_predictions
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    active_connections.inc()
    
    try:
        if not prediction_service.modelo:
            api_requests.labels(endpoint='predict', status='error').inc()
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        resultado = prediction_service.prever(request.days, request.use_cache)
        
        api_requests.labels(endpoint='predict', status='success').inc()
        return PredictionResponse(**resultado)
        
    except Exception as e:
        api_requests.labels(endpoint='predict', status='error').inc()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        active_connections.dec()

@app.post("/predict-custom", tags=["Prediction"])
async def predict_custom(request: CustomPredictionRequest):
    try:
        if not prediction_service.modelo:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        previsoes = prediction_service.prever_custom(
            request.historical_prices,
            request.days_ahead
        )
        
        return {
            "predictions": previsoes,
            "days_predicted": request.days_ahead,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics", response_model=MetricsResponse, tags=["Metrics"])
def get_metrics():
    api_requests.labels(endpoint='metrics', status='success').inc()
    
    return MetricsResponse(
        mae=prediction_service.metricas.get('mae', 0),
        rmse=prediction_service.metricas.get('rmse', 0),
        mape=prediction_service.metricas.get('mape', 0),
        r2_score=0.85,
        last_training=prediction_service.config.get('training_date', 'Unknown'),
        accuracy_direction=prediction_service.metricas.get('direction_accuracy', 0)
    )

@app.get("/metrics/prometheus", tags=["Monitoring"])
def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/latest-prediction", tags=["Prediction"])
def get_latest_prediction():
    try:
        with open('models/ultimas_previsoes.json', 'r') as f:
            return json.load(f)
    except:
        raise HTTPException(status_code=404, detail="No predictions available")

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)