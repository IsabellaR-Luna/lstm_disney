# 🎯 Disney Stock Predictor

Sistema completo de predição de preços de ações da Disney (DIS) utilizando Deep Learning (LSTM) com FastAPI backend e React frontend.

Link da apresentação do Projeto : 

## 🚀 Sobre o Projeto

O **Disney Stock Predictor** é uma aplicação de inteligência artificial que utiliza redes neurais LSTM (Long Short-Term Memory) para prever movimentos futuros dos preços das ações da Disney. O sistema oferece:

- 📈 Previsão do preço para o próximo dia útil
- 📅 Previsões para múltiplos dias (1-30 dias)
- 📊 Consulta de dados históricos reais
- 🎯 Recomendações de trading (COMPRAR/VENDER/MANTER)
- 📉 Análise de tendências e métricas de performance

### 🎓 Objetivo Educacional

Este projeto foi desenvolvido para fins educacionais e de pesquisa, demonstrando a aplicação de Deep Learning em análise de séries temporais financeiras.

---

## 🛠 Tecnologias

### Backend
- **Python 3.8+**
- **FastAPI** - Framework web assíncrono
- **TensorFlow/Keras** - Rede neural LSTM
- **Scikit-learn** - Pré-processamento e métricas
- **Pandas & NumPy** - Manipulação de dados
- **yfinance** - Coleta de dados financeiros

### Frontend
- **React 18** - Interface do usuário
- **Recharts** - Visualização de gráficos
- **Lucide React** - Ícones modernos
- **CSS3** - Estilização responsiva

---

## 📁 Estrutura do Repositório
```
disney-stock-predictor/
│
├── backend/
│   ├── app/
│   │   ├── data/                      # Dados processados e cache
│   │   │   ├── dados_lstm.pkl         # Dados de treino/validação/teste
│   │   │   └── dados_processados.csv  # Histórico formatado
│   │   │
│   │   ├── models/                    # Modelos treinados
│   │   │   ├── modelo_disney_lstm.h5  # Modelo LSTM salvo
│   │   │   ├── metricas.json          # Métricas de performance
│   │   │   ├── config.json            # Configurações do modelo
│   │   │   ├── historico_treino.csv   # Histórico de treinamento
│   │   │   └── resultados_modelo.png  # Gráficos de avaliação
│   │   │
│   │   ├── app.py                     # API FastAPI
│   │   └── model.py                   # Script de treinamento
│   │
│   └── requirements.txt               # Dependências Python
│
├── frontend/
│   ├── node_modules/                  # Dependências Node
│   ├── public/                        # Arquivos públicos
│   ├── src/
│   │   ├── App.js                     # Componente principal React
│   │   ├── App.css                    # Estilos da aplicação
│   │   ├── index.js                   # Entry point
│   │   └── index.css                  # Estilos globais
│   │
│   ├── package.json                   # Dependências e scripts
│   └── package-lock.json
│
└── README.md                          # Este arquivo
```

---

## ✅ Pré-requisitos

### Backend
- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Frontend
- Node.js 14 ou superior
- npm (gerenciador de pacotes Node)

---

## 📦 Instalação

### 1️⃣ Backend (API)
```bash
# Navegar para a pasta do backend
cd backend/app

# Criar ambiente virtual (recomendado)
python -m venv venv

# Ativar ambiente virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar dependências
pip install fastapi uvicorn tensorflow scikit-learn pandas numpy yfinance matplotlib pydantic

# OU usar requirements.txt
pip install -r requirements.txt
```

### 2️⃣ Treinar o Modelo (primeira vez)
```bash
# Ainda na pasta backend/app
python model.py
```

Este comando irá:
- ✅ Baixar dados históricos da Disney (2018-2024)
- ✅ Treinar o modelo LSTM
- ✅ Salvar modelo e métricas nas pastas `data/` e `models/`
- ✅ Gerar gráficos de avaliação

**Tempo estimado:** 5-15 minutos dependendo do hardware

### 3️⃣ Frontend (Interface)
```bash
# Navegar para a pasta do frontend
cd frontend

# Instalar dependências
npm install

# Instalar bibliotecas adicionais
npm install lucide-react recharts
```

---

## 🚀 Como Usar

### Iniciar o Backend (API)
```bash
# Terminal 1 - Na pasta backend/app
uvicorn app:app --reload
```

✅ API disponível em: **http://localhost:8000**  
✅ Documentação Swagger: **http://localhost:8000/docs**

### Iniciar o Frontend
```bash
# Terminal 2 - Na pasta frontend
npm start
```

✅ Interface disponível em: **http://localhost:3000**

### Fluxo de Uso

1. **Acesse** http://localhost:3000
2. **Leia** a aba "Início" para entender o sistema
3. **Clique** em "Próximo Dia" para gerar uma previsão
4. **Explore** as outras abas:
   - 📅 **Múltiplos Dias** - Previsões de 1 a 30 dias
   - 📊 **Histórico** - Consulte dados reais da Disney
   - 📈 **Métricas** - Avalie a performance do modelo

---

## 🔌 Endpoints da API

### Health Check
```http
GET /health
```
Verifica o status da API e do modelo

### Próximo Dia
```http
POST /api/predict/next-day
Body: {} ou { "historical_data": [...] }
```
Prevê o preço para o próximo dia útil

### Múltiplos Dias
```http
POST /api/predict/multi-day
Body: { "days": 7 }
```
Prevê preços para 1-30 dias futuros

### Métricas do Modelo
```http
GET /api/model/metrics
```
Retorna MAE, RMSE, MAPE e acurácia

### Dados Históricos
```http
GET /api/data/historical?start_date=2024-01-01&end_date=2024-07-20
```
Retorna preços históricos reais

### Análise de Investimento
```http
POST /api/analyze/investment
Body: { "risk_profile": "moderate", "horizon": "medium" }
```
Análise personalizada com recomendações

---

## 📊 Métricas do Modelo

O modelo LSTM foi treinado com dados de **2018 a 2024** e apresenta as seguintes métricas:

| Métrica | Descrição | Valor Esperado |
|---------|-----------|----------------|
| **MAE** | Erro Absoluto Médio | ~$2-5 |
| **RMSE** | Raiz do Erro Quadrático | ~$3-6 |
| **MAPE** | Erro Percentual Médio | 3-8% |
| **Acurácia Direcional** | Taxa de acerto da direção | 60-75% |

### Arquitetura do Modelo
```
Input: 60 dias de preços de fechamento
    ↓
LSTM Layer (50 units) + Dropout (0.2)
    ↓
LSTM Layer (50 units) + Dropout (0.2)
    ↓
Dense Layer (25 units)
    ↓
Output: Preço previsto
```

**Total de parâmetros:** ~15.000

---

## ⚠️ Aviso Legal

**IMPORTANTE:** Este sistema é destinado **exclusivamente para fins educacionais e de pesquisa**.
---


## 🐛 Problemas Conhecidos


### Dependências
Se houver erros de importação, reinstale:
```bash
pip install --upgrade tensorflow keras scikit-learn
```

---

