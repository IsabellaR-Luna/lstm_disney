import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataPreparation:
    """
    Preparação de dados com features avançadas:
    - Features sazonais (dia da semana, mês, trimestre)
    - Indicadores técnicos (médias móveis, RSI, volatilidade)
    - Features de eventos históricos (COVID, recessões)
    """
    
    def __init__(self, lookback=60):
        self.lookback = lookback
        self.scaler = None
        
    def carregar_dados(self, caminho):
        """Carrega dados históricos"""
        try:
            df = pd.read_csv(caminho)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            print(f"Dados carregados: {len(df)} registros")
            print(f"Período: {df['Date'].min()} até {df['Date'].max()}")
            return df
        except FileNotFoundError:
            
            return " Arquivo não encontrado. Gerando dados sintéticos..."
    
    
    
    def adicionar_features_sazonais(self, df):
        """Adiciona features de sazonalidade"""
        df = df.copy()
        
        # Features temporais
        df['DayOfWeek'] = df['Date'].dt.dayofweek  # 0=Segunda, 4=Sexta
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['DayOfMonth'] = df['Date'].dt.day
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        
        # Features cíclicas (sin/cos para capturar ciclicidade)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 5)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 5)
        
        # Início/fim de mês (dias 1-5 e 25-31)
        df['IsStartOfMonth'] = (df['DayOfMonth'] <= 5).astype(int)
        df['IsEndOfMonth'] = (df['DayOfMonth'] >= 25).astype(int)
        
        print(f"✅ Features sazonais adicionadas")
        return df
    
    def adicionar_indicadores_tecnicos(self, df):
        """Adiciona indicadores técnicos"""
        df = df.copy()
        
        # Médias Móveis
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        
        # Volatilidade
        df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
        
        # Volume médio
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA_20'] + 1e-10)
        
        # Retornos
        df['Returns'] = df['Close'].pct_change()
        df['Returns_5d'] = df['Close'].pct_change(periods=5)
        df['Returns_20d'] = df['Close'].pct_change(periods=20)
        
        # Range (High-Low)
        df['Price_Range'] = df['High'] - df['Low']
        df['Price_Range_Pct'] = df['Price_Range'] / df['Close']
        
        print(f"✅ Indicadores técnicos adicionados")
        return df
    
    def adicionar_eventos_historicos(self, df):
        """Adiciona features de eventos históricos importantes"""
        df = df.copy()
        
        # COVID-19 (mar 2020 - jun 2021 como período principal)
        df['COVID_Period'] = (
            (df['Date'] >= '2020-03-01') & (df['Date'] <= '2021-06-30')
        ).astype(int)
        
        # Intensidade do COVID (mais forte no início)
        df['COVID_Intensity'] = 0.0
        mask_high = (df['Date'] >= '2020-03-01') & (df['Date'] <= '2020-06-30')
        mask_medium = (df['Date'] > '2020-06-30') & (df['Date'] <= '2020-12-31')
        mask_low = (df['Date'] > '2020-12-31') & (df['Date'] <= '2021-06-30')
        
        df.loc[mask_high, 'COVID_Intensity'] = 1.0
        df.loc[mask_medium, 'COVID_Intensity'] = 0.6
        df.loc[mask_low, 'COVID_Intensity'] = 0.3
        
        # Crise Financeira 2008
        df['Financial_Crisis_2008'] = (
            (df['Date'] >= '2008-09-01') & (df['Date'] <= '2009-06-30')
        ).astype(int)
        
        # Recessão 2015-2016
        df['Recession_2015'] = (
            (df['Date'] >= '2015-08-01') & (df['Date'] <= '2016-02-28')
        ).astype(int)
        
        # Crise 2022 (inflação, guerra)
        df['Crisis_2022'] = (
            (df['Date'] >= '2022-02-01') & (df['Date'] <= '2022-10-31')
        ).astype(int)
        
        # Flag geral de período de crise
        df['Is_Crisis_Period'] = (
            df['COVID_Period'] | 
            df['Financial_Crisis_2008'] | 
            df['Recession_2015'] |
            df['Crisis_2022']
        ).astype(int)
        
        # Dias desde último evento importante
        eventos = df[df['Is_Crisis_Period'] == 1]['Date']
        if len(eventos) > 0:
            df['Days_Since_Last_Crisis'] = df['Date'].apply(
                lambda x: (x - eventos[eventos <= x].max()).days 
                if len(eventos[eventos <= x]) > 0 else 9999
            )
        else:
            df['Days_Since_Last_Crisis'] = 9999
        
        # Normalizar dias desde crise (0-1)
        df['Days_Since_Last_Crisis'] = np.clip(
            df['Days_Since_Last_Crisis'] / 365, 0, 5
        )
        
        print(f"✅ Features de eventos históricos adicionadas")
        return df
    
    def preparar_dados_completos(self, df):
        """Pipeline completo de preparação"""
        print("\n🔧 Preparando dados com features avançadas...")
        
        # Adicionar todas as features
        df = self.adicionar_features_sazonais(df)
        df = self.adicionar_indicadores_tecnicos(df)
        df = self.adicionar_eventos_historicos(df)
        
        # Remover NaNs (gerados pelos indicadores técnicos)
        df = df.dropna().reset_index(drop=True)
        
        print(f"✅ Dataset final: {len(df)} registros com {len(df.columns)} features")
        
        return df
    
    def criar_sequencias(self, df):
        """Cria sequências para LSTM"""
        
        # Features para o modelo (tudo exceto Date e Close original)
        feature_cols = [col for col in df.columns if col not in ['Date', 'Close']]
        
        # Target: Close price
        target_col = 'Close'
        
        # Separar features e target
        X_data = df[feature_cols].values
        y_data = df[target_col].values
        datas = df['Date'].values
        
        # Normalizar TODAS as features juntas
        self.scaler = MinMaxScaler()
        
        # Concatenar X e y para normalizar juntos
        all_data = np.column_stack([y_data.reshape(-1, 1), X_data])
        all_data_scaled = self.scaler.fit_transform(all_data)
        
        # Separar novamente
        y_scaled = all_data_scaled[:, 0]
        X_scaled = all_data_scaled[:, 1:]
        
        print(f"\n📊 Scaler configurado com {self.scaler.n_features_in_} features")
        print(f"   Features: Close (target) + {len(feature_cols)} features adicionais")
        
        # Criar sequências
        X, y, dates_seq = [], [], []
        
        for i in range(self.lookback, len(X_scaled)):
            X.append(X_scaled[i-self.lookback:i])
            y.append(y_scaled[i])  # Predizer próximo Close
            dates_seq.append(datas[i])
        
        X = np.array(X)
        y = np.array(y)
        dates_seq = np.array(dates_seq)
        
        print(f"✅ Sequências criadas: {X.shape}")
        print(f"   Shape de cada sequência: ({self.lookback} timesteps, {X.shape[2]} features)")
        
        return X, y, dates_seq
    
    def dividir_dados(self, X, y, dates, train_size=0.7, val_size=0.15):
        """Divide dados em treino, validação e teste"""
        
        n = len(X)
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        dates_train = dates[:train_end]
        
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        dates_val = dates[train_end:val_end]
        
        X_test = X[val_end:]
        y_test = y[val_end:]
        dates_test = dates[val_end:]
        
        print(f"\n📊 Divisão dos dados:")
        print(f"   Treino:    {len(X_train)} ({len(X_train)/n*100:.1f}%)")
        print(f"   Validação: {len(X_val)} ({len(X_val)/n*100:.1f}%)")
        print(f"   Teste:     {len(X_test)} ({len(X_test)/n*100:.1f}%)")
        
        return {
            'X_train': X_train, 'y_train': y_train, 'datas_train': dates_train,
            'X_val': X_val, 'y_val': y_val, 'datas_val': dates_val,
            'X_test': X_test, 'y_test': y_test, 'datas_test': dates_test,
            'scaler': self.scaler,
            'feature_names': [col for col in None]  # Será preenchido depois
        }


def executar_preparacao(caminho_dados='data/disney_processed.csv', lookback=60):
    """
    Executa preparação completa dos dados
    """
    print("="*70)
    print("🚀 PREPARAÇÃO DE DADOS AVANÇADA")
    print("="*70)
    
    # Inicializar
    prep = DataPreparation(lookback=lookback)
    
    # Carregar dados
    df = prep.carregar_dados(caminho_dados)
    
    # Preparar features completas
    df_completo = prep.preparar_dados_completos(df)
    
    print(f"\n📋 Features disponíveis:")
    print(f"   Total: {len(df_completo.columns)}")
    print(f"   Colunas: {', '.join(df_completo.columns[:10])}...")
    
    # Criar sequências
    X, y, dates = prep.criar_sequencias(df_completo)
    
    # Dividir dados
    dados = prep.dividir_dados(X, y, dates)
    
    # Salvar
    import os
    os.makedirs('data', exist_ok=True)
    
    # Salvar em formato compatível com código original
    dados_lstm = {
        'simples': dados,  # Manter compatibilidade
        'avancado': dados   # Mesmo dataset
    }
    
    with open('data/dados_lstm.pkl', 'wb') as f:
        pickle.dump(dados_lstm, f)
    
    # Salvar DataFrame completo para referência
    df_completo.to_csv('data/dados_processados.csv', index=False)
    
    print(f"\n✅ Dados salvos em:")
    print(f"   - data/dados_lstm.pkl")
    print(f"   - data/dados_processados.csv")
    
    # Estatísticas finais
    print(f"\n📊 Estatísticas Finais:")
    print(f"   Lookback: {lookback} dias")
    print(f"   Total de sequências: {len(X)}")
    print(f"   Features por timestep: {X.shape[2]}")
    print(f"   Período dos dados: {dates[0]} até {dates[-1]}")
    
    return dados_lstm


if __name__ == "__main__":
    # Executar preparação
    dados_lstm = executar_preparacao(lookback=60)
    
    print("\n" + "="*70)
    print("✅ PREPARAÇÃO CONCLUÍDA!")
    print("="*70)
    print("\n📝 Próximo passo: Execute 'model_trainer_enhanced.py' para treinar o modelo")