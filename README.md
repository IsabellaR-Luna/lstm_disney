# 🚀 Sistema de Previsão de Ações com LSTM Híbrido

Sistema avançado de Machine Learning para previsão de preços de ações e classificação de tendências, utilizando LSTM com features sazonais e eventos históricos.

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Arquitetura do Modelo](#arquitetura-do-modelo)
- [Features Implementadas](#features-implementadas)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Estrutura dos Arquivos](#estrutura-dos-arquivos)
- [Métricas de Avaliação](#métricas-de-avaliação)
- [Exemplos](#exemplos)

---

## 🎯 Visão Geral

Este sistema implementa um modelo híbrido LSTM que realiza:

1. **Regressão**: Prevê o preço de fechamento do próximo dia
2. **Classificação**: Prevê a tendência (Baixa/Neutro/Alta) baseado em mudanças percentuais

### Principais Diferenciais

✅ **Features Sazonais**: Captura padrões temporais (dia da semana, mês, trimestre)
✅ **Indicadores Técnicos**: 20+ indicadores (RSI, MACD, Bollinger Bands, etc.)
✅ **Eventos Históricos**: Considera impacto de COVID-19, crises financeiras
✅ **Modelo Híbrido**: Duas saídas independentes para regressão e classificação
✅ **Scaler Consistente**: Todas features normalizadas juntas, evitando incompatibilidades

---

## 🏗️ Arquitetura do Modelo

```
Input (60 timesteps, N features)
          |
    LSTM (128 units)
          |
    Dropout (0.3)
          |
    LSTM (64 units)
          |
    Dropout (0.3)
          |
    LSTM (32 units)
          |
    Dropout (0.2)
          |
    +-----------------+
    |                 |
Dense (50)      Dense (50)
    |                 |
Dropout         Dropout
    |                 |
Dense (1)       Dense (3)
    |                 |
LINEAR        SOFTMAX
    |                 |
  PREÇO        TENDÊNCIA
```

### Características Técnicas

- **Lookback**: 60 dias de histórico
- **Otimizador**: Adam (lr=0.001)
- **Loss Regressão**: MSE (Mean Squared Error)
- **Loss Classificação**: Categorical Crossentropy
- **Early Stopping**: Paciência de 20 épocas
- **Learning Rate Reduction**: Factor 0.5 após 7 épocas sem melhoria

---

## 🎨 Features Implementadas

### 1. Features Básicas (OHLCV)
- Open, High, Low, Close, Volume

### 2. Features Sazonais (12 features)
- `DayOfWeek`, `Month`, `Quarter`
- `Month_sin/cos`: Ciclicidade mensal
- `DayOfWeek_sin/cos`: Ciclicidade semanal
- `IsStartOfMonth`, `IsEndOfMonth`: Padrões de início/fim de mês

### 3. Indicadores Técnicos (20+ features)
- **Médias Móveis**: SMA_5, SMA_20, SMA_50, EMA_12, EMA_26
- **MACD**: MACD, MACD_signal
- **RSI**: Relative Strength Index (14 períodos)
- **Bollinger Bands**: BB_upper, BB_middle, BB_lower, BB_width
- **Volatilidade**: Rolling std de retornos (20 dias)
- **Volume**: Volume_MA_20, Volume_Ratio
- **Retornos**: Returns, Returns_5d, Returns_20d
- **Range**: Price_Range, Price_Range_Pct

### 4. Eventos Históricos (7 features)
- `COVID_Period`: Período da pandemia (mar/2020 - jun/2021)
- `COVID_Intensity`: Intensidade do impacto (1.0 → 0.3)
- `Financial_Crisis_2008`: Crise financeira de 2008
- `Recession_2015`: Recessão de 2015-2016
- `Crisis_2022`: Crise de 2022 (inflação/guerra)
- `Is_Crisis_Period`: Flag de período de crise
- `Days_Since_Last_Crisis`: Dias desde último evento (normalizado)

---

## 📦 Instalação

### Requisitos

```bash
python >= 3.8
tensorflow >= 2.10
numpy
pandas
scikit-learn
matplotlib
```

### Instalação

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

---

## 🚀 Como Usar

### Opção 1: Pipeline Completo (Recomendado)

```bash
python main_pipeline.py --modo completo
```

Isso executa:
1. Preparação de dados com features avançadas
2. Treinamento do modelo híbrido
3. Avaliação e visualização
4. Exemplos de previsão

### Opção 2: Passo a Passo

#### 1. Preparar Dados

```python
from data_preparation_enhanced import executar_preparacao

dados_lstm = executar_preparacao(
    caminho_dados='data/disney_stock_data.csv',
    lookback=60
)
```

#### 2. Treinar Modelo

```python
from model_trainer_enhanced import executar_treinamento_hybrid

modelo, metricas = executar_treinamento_hybrid(tipo_dados='simples')
```

#### 3. Usar Modelo Treinado

```python
from tensorflow.keras.models import load_model
import pickle
import numpy as np

# Carregar modelo
modelo = load_model('models/modelo_disney_hybrid_lstm.h5')

# Carregar dados
with open('data/dados_lstm.pkl', 'rb') as f:
    dados = pickle.load(f)

scaler = dados['simples']['scaler']
X_test = dados['simples']['X_test']

# Fazer previsão
ultimo_batch = X_test[-1:, :, :]  # Últimos 60 dias
pred_price, pred_trend = modelo.predict(ultimo_batch)

# Interpretar resultados
pred_full = np.zeros((1, scaler.n_features_in_))
pred_full[:, 0] = pred_price.flatten()
preco_previsto = scaler.inverse_transform(pred_full)[:, 0][0]

trend_class = np.argmax(pred_trend[0])
tendencia = ['Baixa', 'Neutro', 'Alta'][trend_class]
confianca = pred_trend[0][trend_class] * 100

print(f"Preço: ${preco_previsto:.2f}")
print(f"Tendência: {tendencia} ({confianca:.1f}%)")
```

---

## 📁 Estrutura dos Arquivos

```
.
├── data/
│   ├── disney_stock_data.csv          # Dados históricos (input)
│   ├── dados_lstm.pkl                 # Dados preparados (gerado)
│   └── dados_processados.csv          # Dados com features (gerado)
│
├── models/
│   ├── modelo_disney_hybrid_lstm.h5   # Modelo treinado
│   ├── best_hybrid_model.h5           # Melhor checkpoint
│   ├── metricas_hybrid.json           # Métricas de avaliação
│   ├── config_hybrid.json             # Configuração do modelo
│   ├── historico_treino_hybrid.csv    # Histórico de treino
│   └── resultados_modelo_hybrid.png   # Visualizações
│
├── data_preparation_enhanced.py       # Preparação de dados
├── model_trainer_enhanced.py          # Treinamento do modelo
├── main_pipeline.py                   # Pipeline completo
└── README.md                          # Este arquivo
```

---

## 📊 Métricas de Avaliação

### Métricas de Regressão

- **MAE** (Mean Absolute Error): Erro médio em dólares
- **RMSE** (Root Mean Squared Error): Raiz do erro quadrático médio
- **MAPE** (Mean Absolute Percentage Error): Erro percentual médio

### Métricas de Classificação

- **Acurácia Total**: % de acerto em todas as classes
- **Acurácia Direcional**: % de acerto em Alta/Baixa (ignora Neutro)
- **Precision, Recall, F1-Score**: Por classe (Baixa/Neutro/Alta)

### Visualizações Geradas

1. **Previsão vs Real**: Comparação temporal
2. **Histórico de Loss**: Evolução do treinamento (regressão)
3. **Scatter Plot**: Correlação real vs previsto
4. **Distribuição de Erros**: Histograma dos erros
5. **Acurácia de Classificação**: Evolução da acurácia
6. **Matriz de Confusão**: Desempenho por classe

---

## 💡 Exemplos

### Exemplo 1: Previsão Simples

```python
from model_trainer_enhanced import DisneyHybridLSTMModel
import pickle

# Carregar dados
with open('data/dados_lstm.pkl', 'rb') as f:
    dados = pickle.load(f)

modelo = DisneyHybridLSTMModel(input_shape=(60, N_FEATURES))
modelo.model = load_model('models/modelo_disney_hybrid_lstm.h5')

# Prever próximo dia
resultado = modelo.prever_proximo_dia(
    ultimos_dados=X_test[-1:],
    scaler=dados['simples']['scaler']
)

print(resultado)
# Output:
# {
#     'preco_previsto': 105.23,
#     'tendencia': 'Alta',
#     'confianca_baixa': 15.2,
#     'confianca_neutro': 20.8,
#     'confianca_alta': 64.0
# }
```

### Exemplo 2: Recomendação de Investimento

```python
def recomendar_investimento(resultado):
    tendencia = resultado['tendencia']
    confianca_max = max(
        resultado['confianca_baixa'],
        resultado['confianca_neutro'],
        resultado['confianca_alta']
    )
    
    if tendencia == 'Alta' and resultado['confianca_alta'] > 65:
        return "📈 COMPRAR - Alta confiança de valorização"
    elif tendencia == 'Baixa' and resultado['confianca_baixa'] > 65:
        return "📉 VENDER - Alta confiança de desvalorização"
    elif confianca_max < 50:
        return "⚠️ CAUTELA - Baixa confiança nas previsões"
    else:
        return "⚖️ MANTER - Tendência neutra ou incerta"

# Usar
recomendacao = recomendar_investimento(resultado)
print(recomendacao)
```

### Exemplo 3: Análise de Múltiplos Dias

```python
# Prever próximos 5 dias
previsoes = []

for i in range(5):
    resultado = modelo.prever_proximo_dia(ultimos_dados, scaler)
    previsoes.append(resultado)
    
    # Atualizar ultimos_dados com previsão (para próxima iteração)
    # Nota: Isso é uma simplificação, na prática você precisaria
    # reconstruir todas as features

for i, prev in enumerate(previsoes):
    print(f"Dia +{i+1}: ${prev['preco_previsto']:.2f} - {prev['tendencia']}")
```

---

## 🔧 Personalização

### Ajustar Limites de Classificação

No arquivo `model_trainer_enhanced.py`, função `criar_labels_tendencia`:

```python
# Padrão: -0.5% a +0.5% = Neutro
labels[mudancas_pct < -0.5] = 0  # Baixa
labels[(mudancas_pct >= -0.5) & (mudancas_pct <= 0.5)] = 1  # Neutro
labels[mudancas_pct > 0.5] = 2  # Alta

# Ajustar para ser mais/menos sensível:
labels[mudancas_pct < -1.0] = 0  # Mais conservador
labels[mudancas_pct > 1.0] = 2
```

### Adicionar Novas Features

No arquivo `data_preparation_enhanced.py`:

```python
def adicionar_feature_customizada(self, df):
    # Exemplo: Momentum de 14 dias
    df['Momentum_14'] = df['Close'] - df['Close'].shift(14)
    
    # Exemplo: Volume médio móvel
    df['Volume_SMA_50'] = df['Volume'].rolling(50).mean()
    
    return df
```

---

## 📈 Resultados Esperados

Com dados de qualidade e treinamento adequado:

- **MAE**: ~$2-5 (para ações de $100)
- **MAPE**: ~2-5%
- **Acurácia de Tendência**: 55-70%
- **Acurácia Direcional**: 60-75%

---

## ⚠️ Avisos Importantes

1. **Não é Conselho Financeiro**: Este modelo é para fins educacionais
2. **Dados Históricos**: Performance passada não garante resultados futuros
3. **Validação**: Sempre valide previsões com análise fundamentalista
4. **Risco**: Investimentos em ações envolvem risco de perda de capital

---

## 🤝 Contribuindo

Sugestões de melhorias:

1. Adicionar mais eventos históricos específicos
2. Implementar ensemble com múltiplos modelos
3. Adicionar análise de sentimento de notícias
4. Incorporar dados macroeconômicos
5. Implementar attention mechanism

---

## 📝 Licença

Este projeto é fornecido "como está", sem garantias.

---

## 📧 Contato

Para dúvidas ou sugestões sobre o modelo, consulte a documentação inline no código.

---

**Desenvolvido com ❤️ usando TensorFlow e Python**