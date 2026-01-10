# train_model.py
import yfinance as yf
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURAÇÕES ====================
SYMBOL = 'DIS'
START_DATE = '2018-01-01'
END_DATE = '2024-07-20'
WINDOW_SIZE = 60
EPOCHS = 100
BATCH_SIZE = 32

# ==================== COLETA DE DADOS ====================
print("📊 Baixando dados da Disney...")
df = yf.download(SYMBOL, start=START_DATE, end=END_DATE, progress=False)
print(f"✅ {len(df)} dias coletados")

# ==================== PREPARAÇÃO DOS DADOS ====================
print("\n🔧 Preparando dados...")

# Usar apenas Close
close_prices = df['Close'].values.reshape(-1, 1)

# Normalizar
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Criar sequências
X, y = [], []
for i in range(WINDOW_SIZE, len(scaled_data)):
    X.append(scaled_data[i-WINDOW_SIZE:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Dividir dados: 70% treino, 15% validação, 15% teste
n_train = int(len(X) * 0.7)
n_val = int(len(X) * 0.85)

X_train, y_train = X[:n_train], y[:n_train]
X_val, y_val = X[n_train:n_val], y[n_train:n_val]
X_test, y_test = X[n_val:], y[n_val:]

datas = df.index[WINDOW_SIZE:]
datas_train = datas[:n_train]
datas_val = datas[n_train:n_val]
datas_test = datas[n_val:]

print(f"Treino: {len(X_train)} | Validação: {len(X_val)} | Teste: {len(X_test)}")

# ==================== CRIAR MODELO ====================
print("\n🤖 Criando modelo LSTM...")

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
print(f"Parâmetros: {model.count_params()}")

# ==================== TREINAR MODELO ====================
print("\n🚀 Treinando modelo...")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ==================== AVALIAR MODELO ====================
print("\n📊 Avaliando modelo...")

y_pred = model.predict(X_test, verbose=0)

# Inverter normalização
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = scaler.inverse_transform(y_pred)

# Calcular métricas
mae = mean_absolute_error(y_test_inv, y_pred_inv)
mse = mean_squared_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100

direction_true = np.diff(y_test_inv.flatten()) > 0
direction_pred = np.diff(y_pred_inv.flatten()) > 0
direction_accuracy = np.mean(direction_true == direction_pred) * 100

metrics = {
    'mae': float(mae),
    'rmse': float(rmse),
    'mape': float(mape),
    'direction_accuracy': float(direction_accuracy)
}

print(f"\nMAE: ${mae:.2f}")
print(f"RMSE: ${rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"Acurácia Direção: {direction_accuracy:.1f}%")

# ==================== VISUALIZAÇÕES ====================
print("\n📈 Criando visualizações...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Previsão vs Real
ax = axes[0, 0]
ax.plot(datas_test, y_test_inv, label='Real', alpha=0.7)
ax.plot(datas_test, y_pred_inv, label='Previsão', alpha=0.7)
ax.set_title('Previsão vs Valores Reais')
ax.set_xlabel('Data')
ax.set_ylabel('Preço ($)')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Histórico de Loss
ax = axes[0, 1]
ax.plot(history.history['loss'], label='Treino')
ax.plot(history.history['val_loss'], label='Validação')
ax.set_title('Histórico de Perda')
ax.set_xlabel('Época')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Scatter Plot
ax = axes[1, 0]
ax.scatter(y_test_inv, y_pred_inv, alpha=0.5)
ax.plot([y_test_inv.min(), y_test_inv.max()], 
        [y_test_inv.min(), y_test_inv.max()], 
        'r--', lw=2)
ax.set_title('Correlação: Real vs Previsão')
ax.set_xlabel('Valor Real ($)')
ax.set_ylabel('Valor Previsto ($)')
ax.grid(True, alpha=0.3)

# 4. Distribuição do Erro
ax = axes[1, 1]
erros = y_test_inv.flatten() - y_pred_inv.flatten()
ax.hist(erros, bins=30, edgecolor='black', alpha=0.7)
ax.axvline(x=0, color='r', linestyle='--')
ax.set_title('Distribuição dos Erros')
ax.set_xlabel('Erro ($)')
ax.set_ylabel('Frequência')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/resultados_modelo.png', dpi=100)
print("✅ Gráfico salvo: models/resultados_modelo.png")

# ==================== SALVAR MODELO E DADOS ====================
print("\n💾 Salvando modelo e dados...")

import os
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Salvar modelo
model.save('models/modelo_disney_lstm.h5')

# Salvar métricas
with open('models/metricas.json', 'w') as f:
    json.dump(metrics, f, indent=4)

# Salvar histórico
pd.DataFrame(history.history).to_csv('models/historico_treino.csv', index=False)

# Salvar configuração
config = {
    'symbol': SYMBOL,
    'start_date': START_DATE,
    'end_date': END_DATE,
    'window_size': WINDOW_SIZE,
    'features': ['close'],
    'n_features': 1,
    'total_params': int(model.count_params()),
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'epochs_trained': len(history.history['loss']),
    'best_val_loss': float(min(history.history['val_loss']))
}

with open('models/config.json', 'w') as f:
    json.dump(config, f, indent=4)

# Salvar dados preparados para API
dados_lstm = {
    'simples': {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'datas_train': datas_train,
        'datas_val': datas_val,
        'datas_test': datas_test,
        'scaler': scaler
    }
}

with open('data/dados_lstm.pkl', 'wb') as f:
    pickle.dump(dados_lstm, f)

# Salvar dados processados
df_processed = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
df_processed.reset_index(inplace=True)
df_processed.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
df_processed.to_csv('data/dados_processados.csv', index=False)

print("\n✅ TREINAMENTO CONCLUÍDO!")
print(f"📁 Arquivos salvos:")
print(f"   • models/modelo_disney_lstm.h5")
print(f"   • models/metricas.json")
print(f"   • models/config.json")
print(f"   • models/historico_treino.csv")
print(f"   • models/resultados_modelo.png")
print(f"   • data/dados_lstm.pkl")
print(f"   • data/dados_processados.csv")
print(f"\n🎯 Scaler: {scaler.n_features_in_} feature(s)")
print(f"🎯 Modelo: input_shape = ({WINDOW_SIZE}, 1)")