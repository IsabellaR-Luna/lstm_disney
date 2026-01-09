import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class DisneyHybridLSTMModel:
    """
    Modelo híbrido LSTM para:
    1. Regressão: Prever preço de fechamento do próximo dia
    2. Classificação: Prever tendência (0=Baixa, 1=Neutro, 2=Alta)
    """
    
    def __init__(self, input_shape, model_type='hybrid'):
        self.input_shape = input_shape
        self.model_type = model_type
        self.model = None
        self.history = None
        self.metrics = {}
        
    def criar_modelo(self):
        """Cria modelo com duas saídas: regressão e classificação"""
        
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # Camadas LSTM compartilhadas
        x = LSTM(128, return_sequences=True)(inputs)
        x = Dropout(0.3)(x)
        x = LSTM(64, return_sequences=True)(x)
        x = Dropout(0.3)(x)
        x = LSTM(32, return_sequences=False)(x)
        x = Dropout(0.2)(x)
        
        # Branch para Regressão (preço)
        regression_branch = Dense(50, activation='relu', name='reg_dense1')(x)
        regression_branch = Dropout(0.2)(regression_branch)
        regression_output = Dense(1, activation='linear', name='price_output')(regression_branch)
        
        # Branch para Classificação (tendência)
        classification_branch = Dense(50, activation='relu', name='class_dense1')(x)
        classification_branch = Dropout(0.2)(classification_branch)
        classification_output = Dense(3, activation='softmax', name='trend_output')(classification_branch)
        
        # Criar modelo com múltiplas saídas
        self.model = Model(
            inputs=inputs,
            outputs=[regression_output, classification_output]
        )
        
        # Compilar com perdas diferentes para cada saída
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'price_output': 'mse',
                'trend_output': 'categorical_crossentropy'
            },
            loss_weights={
                'price_output': 1.0,
                'trend_output': 0.5
            },
            metrics={
                'price_output': ['mae'],
                'trend_output': ['accuracy']
            }
        )
        
        return self.model
    
    def treinar(self, X_train, y_train_price, y_train_trend, 
                X_val, y_val_price, y_val_trend, 
                epochs=100, batch_size=32):
        """
        Treina o modelo híbrido
        
        Args:
            X_train: Features de treinamento
            y_train_price: Target de preço (regressão)
            y_train_trend: Target de tendência (classificação one-hot)
            X_val: Features de validação
            y_val_price: Target de preço validação
            y_val_trend: Target de tendência validação
        """
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_hybrid_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        print(f"\n🚀 Treinando modelo híbrido...")
        print(f"   - Regressão: Previsão de preço")
        print(f"   - Classificação: Tendência (Baixa/Neutro/Alta)")
        
        self.history = self.model.fit(
            X_train,
            {
                'price_output': y_train_price,
                'trend_output': y_train_trend
            },
            validation_data=(
                X_val,
                {
                    'price_output': y_val_price,
                    'trend_output': y_val_trend
                }
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def avaliar(self, X_test, y_test_price, y_test_trend, scaler):
        """
        Avalia modelo em ambas as tarefas
        """
        # Fazer predições
        pred_price, pred_trend = self.model.predict(X_test, verbose=0)
        
        # === AVALIAÇÃO DE REGRESSÃO ===
        # Criar arrays com todas as features para inverse_transform
        y_test_full = np.zeros((len(y_test_price), scaler.n_features_in_))
        y_test_full[:, 0] = y_test_price.flatten()
        
        y_pred_full = np.zeros((len(pred_price), scaler.n_features_in_))
        y_pred_full[:, 0] = pred_price.flatten()
        
        # Inverse transform apenas da coluna de preço
        y_test_inv = scaler.inverse_transform(y_test_full)[:, 0]
        y_pred_inv = scaler.inverse_transform(y_pred_full)[:, 0]
        
        # Métricas de regressão
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
        
        # === AVALIAÇÃO DE CLASSIFICAÇÃO ===
        # Converter one-hot para classes
        y_test_trend_classes = np.argmax(y_test_trend, axis=1)
        y_pred_trend_classes = np.argmax(pred_trend, axis=1)
        
        # Métricas de classificação
        trend_accuracy = accuracy_score(y_test_trend_classes, y_pred_trend_classes) * 100
        
        # Acurácia de direção (apenas Alta vs Baixa, ignorando Neutro)
        direction_mask = y_test_trend_classes != 1  # Não é neutro
        if np.sum(direction_mask) > 0:
            direction_accuracy = accuracy_score(
                y_test_trend_classes[direction_mask],
                y_pred_trend_classes[direction_mask]
            ) * 100
        else:
            direction_accuracy = 0
        
        self.metrics = {
            # Métricas de Regressão
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            
            # Métricas de Classificação
            'trend_accuracy': float(trend_accuracy),
            'direction_accuracy': float(direction_accuracy),
            
            # Relatório detalhado
            'classification_report': classification_report(
                y_test_trend_classes, 
                y_pred_trend_classes,
                target_names=['Baixa', 'Neutro', 'Alta'],
                output_dict=True
            )
        }
        
        return self.metrics, y_pred_inv, y_pred_trend_classes, y_test_trend_classes
    
    def plot_resultados(self, y_test_price, y_pred_price, 
                       y_test_trend, y_pred_trend,
                       datas_test, scaler):
        """
        Plota resultados de regressão e classificação
        """
        # Inverse transform dos preços
        y_test_full = np.zeros((len(y_test_price), scaler.n_features_in_))
        y_test_full[:, 0] = y_test_price.flatten()
        y_test_inv = scaler.inverse_transform(y_test_full)[:, 0]
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        
        # Plot 1: Previsão de Preço vs Real
        ax = axes[0, 0]
        ax.plot(datas_test, y_test_inv, label='Real', linewidth=2, alpha=0.8)
        ax.plot(datas_test, y_pred_price, label='Previsão', linewidth=2, alpha=0.8)
        ax.set_title('Previsão de Preço vs Valores Reais', fontsize=12, fontweight='bold')
        ax.set_xlabel('Data')
        ax.set_ylabel('Preço ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Histórico de Perda (Regressão)
        ax = axes[0, 1]
        ax.plot(self.history.history['price_output_loss'], label='Treino (Preço)', linewidth=2)
        ax.plot(self.history.history['val_price_output_loss'], label='Validação (Preço)', linewidth=2)
        ax.set_title('Histórico de Perda - Regressão', fontsize=12, fontweight='bold')
        ax.set_xlabel('Época')
        ax.set_ylabel('Loss (MSE)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Scatter Plot Preço
        ax = axes[1, 0]
        ax.scatter(y_test_inv, y_pred_price, alpha=0.6, s=30)
        ax.plot([y_test_inv.min(), y_test_inv.max()], 
                [y_test_inv.min(), y_test_inv.max()], 
                'r--', lw=2, label='Linha Ideal')
        ax.set_title('Correlação: Real vs Previsão', fontsize=12, fontweight='bold')
        ax.set_xlabel('Valor Real ($)')
        ax.set_ylabel('Valor Previsto ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Distribuição do Erro de Preço
        ax = axes[1, 1]
        erros = y_test_inv - y_pred_price
        ax.hist(erros, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero')
        ax.set_title('Distribuição dos Erros de Preço', fontsize=12, fontweight='bold')
        ax.set_xlabel('Erro ($)')
        ax.set_ylabel('Frequência')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Acurácia de Classificação ao longo do tempo
        ax = axes[2, 0]
        ax.plot(self.history.history['trend_output_accuracy'], 
                label='Treino (Tendência)', linewidth=2)
        ax.plot(self.history.history['val_trend_output_accuracy'], 
                label='Validação (Tendência)', linewidth=2)
        ax.set_title('Acurácia da Classificação de Tendência', fontsize=12, fontweight='bold')
        ax.set_xlabel('Época')
        ax.set_ylabel('Acurácia')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Matriz de Confusão
        ax = axes[2, 1]
        cm = confusion_matrix(y_test_trend, y_pred_trend)
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title('Matriz de Confusão - Tendência', fontsize=12, fontweight='bold')
        
        # Adicionar valores na matriz
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
        
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(['Baixa', 'Neutro', 'Alta'])
        ax.set_yticklabels(['Baixa', 'Neutro', 'Alta'])
        ax.set_xlabel('Previsto')
        ax.set_ylabel('Real')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig('models/resultados_modelo_hybrid.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 Gráfico salvo em 'models/resultados_modelo_hybrid.png'")
    
    def salvar_modelo(self, caminho='models/'):
        """Salva modelo, histórico e métricas"""
        import os
        os.makedirs(caminho, exist_ok=True)
        
        # Salvar modelo completo
        self.model.save(f'{caminho}modelo_disney_hybrid_lstm.h5')
        
        # Salvar histórico
        pd.DataFrame(self.history.history).to_csv(f'{caminho}historico_treino_hybrid.csv')
        
        # Salvar métricas
        with open(f'{caminho}metricas_hybrid.json', 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        # Salvar configuração
        config = {
            'model_type': 'hybrid_regression_classification',
            'input_shape': self.input_shape,
            'total_params': int(self.model.count_params()),
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'epochs_trained': len(self.history.history['loss']),
            'tasks': ['price_regression', 'trend_classification']
        }
        
        with open(f'{caminho}config_hybrid.json', 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"\n✅ Modelo híbrido salvo em '{caminho}'")
    
    def prever_proximo_dia(self, ultimos_dados, scaler):
        """
        Faz previsão para o próximo dia
        
        Args:
            ultimos_dados: Últimas N observações (shape: [1, timesteps, features])
            scaler: Scaler usado no treinamento
            
        Returns:
            dict com preço previsto e tendência
        """
        pred_price, pred_trend = self.model.predict(ultimos_dados, verbose=0)
        
        # Inverse transform do preço
        pred_full = np.zeros((1, scaler.n_features_in_))
        pred_full[:, 0] = pred_price.flatten()
        pred_price_inv = scaler.inverse_transform(pred_full)[:, 0][0]
        
        # Classe de tendência
        trend_class = np.argmax(pred_trend[0])
        trend_prob = pred_trend[0]
        trend_names = ['Baixa', 'Neutro', 'Alta']
        
        resultado = {
            'preco_previsto': float(pred_price_inv),
            'tendencia': trend_names[trend_class],
            'confianca_baixa': float(trend_prob[0] * 100),
            'confianca_neutro': float(trend_prob[1] * 100),
            'confianca_alta': float(trend_prob[2] * 100)
        }
        
        return resultado


def executar_treinamento_hybrid(tipo_dados='simples'):
    """
    Executa treinamento do modelo híbrido
    """
    # Carregar dados preparados
    with open('data/dados_lstm.pkl', 'rb') as f:
        dados_preparados = pickle.load(f)
    
    dados = dados_preparados[tipo_dados]
    
    X_train = dados['X_train']
    y_train = dados['y_train']
    X_val = dados['X_val']
    y_val = dados['y_val']
    X_test = dados['X_test']
    y_test = dados['y_test']
    scaler = dados['scaler']
    
    # Criar labels de classificação (tendência)
    # Baseado na mudança percentual: < -1% = Baixa, -1% a 1% = Neutro, > 1% = Alta
    def criar_labels_tendencia(y_prices, scaler):
        # Inverse transform para calcular mudanças reais
        y_full = np.zeros((len(y_prices), scaler.n_features_in_))
        y_full[:, 0] = y_prices.flatten()
        y_inv = scaler.inverse_transform(y_full)[:, 0]
        
        # Calcular mudança percentual (aproximada, usando o valor como referência)
        # Para melhor precisão, precisaríamos do valor anterior real
        mudancas = np.diff(y_inv, prepend=y_inv[0])
        mudancas_pct = (mudancas / (y_inv + 1e-8)) * 100
        
        # Classificar
        labels = np.zeros(len(mudancas_pct), dtype=int)
        labels[mudancas_pct < -0.5] = 0  # Baixa
        labels[(mudancas_pct >= -0.5) & (mudancas_pct <= 0.5)] = 1  # Neutro
        labels[mudancas_pct > 0.5] = 2  # Alta
        
        # Converter para one-hot
        from tensorflow.keras.utils import to_categorical
        return to_categorical(labels, num_classes=3)
    
    y_train_trend = criar_labels_tendencia(y_train, scaler)
    y_val_trend = criar_labels_tendencia(y_val, scaler)
    y_test_trend = criar_labels_tendencia(y_test, scaler)
    
    print(f"\n📊 Distribuição de classes de tendência (treino):")
    classes_treino = np.argmax(y_train_trend, axis=1)
    print(f"   Baixa: {np.sum(classes_treino == 0)} ({np.sum(classes_treino == 0)/len(classes_treino)*100:.1f}%)")
    print(f"   Neutro: {np.sum(classes_treino == 1)} ({np.sum(classes_treino == 1)/len(classes_treino)*100:.1f}%)")
    print(f"   Alta: {np.sum(classes_treino == 2)} ({np.sum(classes_treino == 2)/len(classes_treino)*100:.1f}%)")
    
    # Criar e treinar modelo
    input_shape = (X_train.shape[1], X_train.shape[2])
    modelo = DisneyHybridLSTMModel(input_shape, model_type='hybrid')
    
    modelo.criar_modelo()
    print(f"\n🏗️  Modelo híbrido criado: {modelo.model.count_params():,} parâmetros")
    print(modelo.model.summary())
    
    # Treinar
    modelo.treinar(
        X_train, y_train, y_train_trend,
        X_val, y_val, y_val_trend,
        epochs=100, 
        batch_size=32
    )
    
    # Avaliar
    metricas, y_pred_price, y_pred_trend, y_test_trend_classes = modelo.avaliar(
        X_test, y_test, y_test_trend, scaler
    )
    
    print("\n" + "="*70)
    print("📊 MÉTRICAS DE AVALIAÇÃO - MODELO HÍBRIDO")
    print("="*70)
    print("\n🎯 REGRESSÃO (Previsão de Preço):")
    print(f"   MAE:  ${metricas['mae']:.2f}")
    print(f"   RMSE: ${metricas['rmse']:.2f}")
    print(f"   MAPE: {metricas['mape']:.2f}%")
    
    print("\n📈 CLASSIFICAÇÃO (Tendência):")
    print(f"   Acurácia Total: {metricas['trend_accuracy']:.1f}%")
    print(f"   Acurácia Direcional (Alta/Baixa): {metricas['direction_accuracy']:.1f}%")
    
    print("\n📋 Relatório Detalhado por Classe:")
    report = metricas['classification_report']
    for classe in ['Baixa', 'Neutro', 'Alta']:
        if classe in report:
            print(f"   {classe}:")
            print(f"      Precision: {report[classe]['precision']*100:.1f}%")
            print(f"      Recall:    {report[classe]['recall']*100:.1f}%")
            print(f"      F1-Score:  {report[classe]['f1-score']*100:.1f}%")
    
    # Plotar resultados
    modelo.plot_resultados(
        y_test, y_pred_price,
        y_test_trend_classes, y_pred_trend,
        dados['datas_test'], scaler
    )
    
    # Salvar modelo
    modelo.salvar_modelo()
    
    # Exemplo de previsão para o próximo dia
    print("\n" + "="*70)
    print("🔮 EXEMPLO: Previsão para o próximo dia")
    print("="*70)
    ultimo_batch = X_test[-1:, :, :]  # Última sequência do teste
    resultado = modelo.prever_proximo_dia(ultimo_batch, scaler)
    
    print(f"\n💰 Preço Previsto: ${resultado['preco_previsto']:.2f}")
    print(f"📊 Tendência: {resultado['tendencia']}")
    print(f"\n   Probabilidades:")
    print(f"      Baixa:  {resultado['confianca_baixa']:.1f}%")
    print(f"      Neutro: {resultado['confianca_neutro']:.1f}%")
    print(f"      Alta:   {resultado['confianca_alta']:.1f}%")
    
    return modelo, metricas


if __name__ == "__main__":
    print("="*70)
    print("🚀 TREINAMENTO DE MODELO HÍBRIDO LSTM")
    print("   Regressão: Previsão de Preço + Classificação: Tendência")
    print("="*70)
    
    modelo, metricas = executar_treinamento_hybrid(tipo_dados='simples')
    
    print("\n" + "="*70)
    print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("="*70)
    print(f"\n📁 Arquivos salvos em 'models/':")
    print("   - modelo_disney_hybrid_lstm.h5")
    print("   - metricas_hybrid.json")
    print("   - config_hybrid.json")
    print("   - historico_treino_hybrid.csv")
    print("   - resultados_modelo_hybrid.png")