from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List, Optional
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import pickle
import logging
from keras.models import load_model
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Disney Stock Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # URL do frontend React
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

MODEL_PATH = '/app/app/models/modelo_disney_lstm.h5'
DATA_PATH = '/app/app/data/dados_lstm.pkl'
METRICS_PATH = '/app/app/models/metricas.json'
HISTORICAL_DATA_PATH = '/app/app/data/dados_processados.csv'


class OHLCVData(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @validator('date')
    def validate_date(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except:
            raise ValueError('Date must be YYYY-MM-DD format')
    
    @validator('high')
    def validate_high_low(cls, v, values):
        if 'low' in values and v < values['low']:
            raise ValueError('High must be >= Low')
        return v

class NextDayRequest(BaseModel):
    historical_data: Optional[List[OHLCVData]] = None

class MultiDayRequest(BaseModel):
    days: int
    historical_data: Optional[List[OHLCVData]] = None
    
    @validator('days')
    def validate_days(cls, v):
        if v < 1 or v > 30:
            raise ValueError('Days must be between 1 and 30')
        return v

class InvestmentRequest(BaseModel):
    capital: Optional[float] = None
    horizon: str = "medium"
    risk_profile: str = "moderate"
    
    @validator('horizon')
    def validate_horizon(cls, v):
        if v not in ['short', 'medium', 'long']:
            raise ValueError('Horizon must be short, medium, or long')
        return v
    
    @validator('risk_profile')
    def validate_risk(cls, v):
        if v not in ['conservative', 'moderate', 'aggressive']:
            raise ValueError('Risk must be conservative, moderate, or aggressive')
        return v

class ModelLoader:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = load_model(MODEL_PATH, compile=False)
            logger.info(f'Model Loaded from {MODEL_PATH}')

            with open(DATA_PATH, 'rb') as f:
                cls._instance.data = pickle.load(f)
            logger.info(f'Data Loaded from {DATA_PATH}')

            cls._instance.scaler = cls._instance.data['simples']['scaler']
            logger.info(f"Model loaded - scaler has {cls._instance.scaler.n_features_in_} feature(s)")
        return cls._instance

def validate_ohlcv_data(data: List[OHLCVData]):
    if len(data) < 60:
        raise HTTPException(400, f"Need at least 60 days of data, got {len(data)}")
    
    dates = [datetime.strptime(d.date, '%Y-%m-%d') for d in data]
    if dates != sorted(dates):
        raise HTTPException(400, "Dates must be in chronological order")
    
    logger.info(f"Validated {len(data)} days of OHLCV data")
    return data

def prepare_data_for_prediction(data: List[OHLCVData], scaler):
    """CORRIGIDO: Usar apenas Close"""
    df = pd.DataFrame([d.dict() for d in data])
    df['date'] = pd.to_datetime(df['date'])
    
    # Usar apenas Close (1 feature)
    close_data = df['close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(close_data)
    
    # Pegar últimos 60 dias e reshape para (1, 60, 1)
    X = scaled_data[-60:].reshape(1, 60, 1)
    return X

def inverse_transform_price(pred_price, scaler):
    """Inverter normalização do preço previsto"""
    return float(scaler.inverse_transform(pred_price.reshape(-1, 1))[0, 0])

def calculate_trend(current_price, predicted_price):
    """NOVO: Calcular tendência baseado na diferença de preço"""
    diff_percent = ((predicted_price - current_price) / current_price) * 100
    
    if diff_percent > 1.0:
        return "Alta", min(abs(diff_percent) * 20, 100)
    elif diff_percent < -1.0:
        return "Baixa", min(abs(diff_percent) * 20, 100)
    else:
        return "Neutro", 100 - abs(diff_percent) * 10

@app.get("/health")
def health_check():
    try:
        loader = ModelLoader()
        model_loaded = loader.model is not None
        data_loaded = loader.data is not None
        
        return {
            "status": "healthy" if model_loaded and data_loaded else "unhealthy",
            "model_loaded": model_loaded,
            "data_loaded": data_loaded,
            "scaler_features": loader.scaler.n_features_in_,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "data_loaded": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/predict/next-day")
def predict_next_day(request: NextDayRequest):
    try:
        loader = ModelLoader()
        
        if request.historical_data:
            validate_ohlcv_data(request.historical_data)
            X = prepare_data_for_prediction(request.historical_data, loader.scaler)
            current_price = request.historical_data[-1].close
        else:
            X = loader.data['simples']['X_test'][-1:, :, :]
            # Pegar último preço real do teste
            current_price = float(loader.scaler.inverse_transform(
                loader.data['simples']['y_test'][-1].reshape(-1, 1)
            )[0, 0])
        
        # Modelo retorna apenas preço
        pred_price = loader.model.predict(X, verbose=0)
        price = inverse_transform_price(pred_price, loader.scaler)
        
        # Calcular tendência manualmente
        trend, confidence = calculate_trend(current_price, price)
        
        recommendation = "MANTER"
        if trend == "Alta" and confidence > 65:
            recommendation = "COMPRAR"
        elif trend == "Baixa" and confidence > 65:
            recommendation = "VENDER"
        
        logger.info(f"Next day prediction: ${price:.2f}, {trend}, {recommendation}")
        
        return {
            "success": True,
            "data": {
                "predicted_price": round(price, 2),
                "current_price": round(current_price, 2),
                "trend": trend,
                "confidence": round(confidence, 1),
                "recommendation": recommendation
            },
            "metadata": {"timestamp": datetime.now().isoformat(), "model_version": "1.0"}
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(500, str(e))

@app.post("/api/predict/multi-day")
def predict_multi_day(request: MultiDayRequest):
    try:
        loader = ModelLoader()
        
        if request.historical_data:
            validate_ohlcv_data(request.historical_data)
            X = prepare_data_for_prediction(request.historical_data, loader.scaler)
            last_prices = [d.close for d in request.historical_data[-60:]]
        else:
            X = loader.data['simples']['X_test'][-1:, :, :]
            # Pegar últimos 60 preços reais
            last_60 = loader.data['simples']['X_test'][-1, :, 0]
            last_prices = loader.scaler.inverse_transform(last_60.reshape(-1, 1)).flatten().tolist()
        
        predictions = []
        current_sequence = X.copy()
        
        for i in range(request.days):
            # Prever próximo preço
            pred_price = loader.model.predict(current_sequence, verbose=0)
            price = inverse_transform_price(pred_price, loader.scaler)
            
            # Calcular tendência
            current_price = last_prices[-1]
            trend, _ = calculate_trend(current_price, price)
            
            predictions.append({
                "day": i + 1,
                "date": (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d'),
                "predicted_price": round(price, 2),
                "trend": trend
            })
            
            # Atualizar sequência para próxima previsão
            pred_scaled = loader.scaler.transform([[price]])[0, 0]
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = pred_scaled
            last_prices.append(price)
        
        logger.info(f"Multi-day prediction for {request.days} days completed")
        
        return {
            "success": True,
            "data": {"predictions": predictions},
            "metadata": {"timestamp": datetime.now().isoformat(), "model_version": "1.0"}
        }
    except Exception as e:
        logger.error(f"Multi-day prediction error: {str(e)}")
        raise HTTPException(500, str(e))

@app.get("/api/model/metrics")
def get_model_metrics():
    try:
        with open(METRICS_PATH, 'r') as f:  # Nome correto
            
            metrics = json.load(f)
        
        logger.info("Model metrics retrieved")
        
        return {
            "success": True,
            "data": {
                "mae": round(metrics['mae'], 2),
                "rmse": round(metrics['rmse'], 2),
                "mape": round(metrics['mape'], 2),
                "direction_accuracy": round(metrics['direction_accuracy'], 1)
            },
            "metadata": {"timestamp": datetime.now().isoformat(), "model_version": "1.0"}
        }
    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        raise HTTPException(500, str(e))

@app.get("/api/data/historical")
def get_historical_data(start_date: str, end_date: str):
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        if start >= end:
            raise HTTPException(400, "Start date must be before end date")
        
        df = pd.read_csv(HISTORICAL_DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        
        mask = (df['Date'] >= start) & (df['Date'] <= end)
        filtered = df[mask][['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].to_dict('records')
        
        logger.info(f"Historical data from {start_date} to {end_date}: {len(filtered)} records")
        
        return {
            "success": True,
            "data": {"historical": filtered},
            "metadata": {"timestamp": datetime.now().isoformat(), "records": len(filtered)}
        }
    except ValueError:
        raise HTTPException(400, "Invalid date format, use YYYY-MM-DD")
    except Exception as e:
        logger.error(f"Historical data error: {str(e)}")
        raise HTTPException(500, str(e))

@app.post("/api/analyze/investment")
def analyze_investment(request: InvestmentRequest):
    try:
        loader = ModelLoader()
        X = loader.data['simples']['X_test'][-1:, :, :]
        
        # Pegar preço atual
        current_price = float(loader.scaler.inverse_transform(
            loader.data['simples']['y_test'][-1].reshape(-1, 1)
        )[0, 0])
        
        # Prever preço
        pred_price = loader.model.predict(X, verbose=0)
        price = inverse_transform_price(pred_price, loader.scaler)
        
        # Calcular tendência
        trend, confidence = calculate_trend(current_price, price)
        
        risk_multiplier = {'conservative': 0.7, 'moderate': 1.0, 'aggressive': 1.3}[request.risk_profile]
        adjusted_confidence = confidence * risk_multiplier
        
        if adjusted_confidence > 65 and trend == 'Alta':
            recommendation = "COMPRAR"
        elif adjusted_confidence > 65 and trend == 'Baixa':
            recommendation = "VENDER"
        else:
            recommendation = "MANTER"
        
        score = min(100, adjusted_confidence)
        
        logger.info(f"Investment analysis: {recommendation}, score: {score:.1f}")
        
        return {
            "success": True,
            "data": {
                "recommendation": recommendation,
                "confidence_score": round(score, 1),
                "predicted_price": round(price, 2),
                "current_price": round(current_price, 2),
                "trend": trend,
                "risk_profile": request.risk_profile,
                "horizon": request.horizon
            },
            "metadata": {"timestamp": datetime.now().isoformat(), "model_version": "1.0"}
        }
    except Exception as e:
        logger.error(f"Investment analysis error: {str(e)}")
        raise HTTPException(500, str(e))