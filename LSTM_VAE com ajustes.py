    import os
    import sys

    # Configure matplotlib backend based on environment
    if 'ipykernel' in sys.modules:
        # Running in Jupyter/IPython
        try:
            import IPython
            ipython = IPython.get_ipython()
            ipython.run_line_magic('matplotlib', 'inline')
        except Exception:
            os.environ['MPLBACKEND'] = 'TkAgg'
    else:
        # Running as script
        os.environ['MPLBACKEND'] = 'TkAgg'

    import numpy as np
    np.random.seed(0)
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    tf.random.set_seed(0)
    import glob
    from keras_tuner import RandomSearch
    import sys
    import keras_tuner as kt
    import pickle
    import datetime

    # Configuração otimizada para GPU NVIDIA/CUDA
    def setup_gpu():
        print("\nVerificando configuração GPU:")
        print("TensorFlow versão:", tf.__version__)
        print("CUDA disponível:", tf.test.is_built_with_cuda())
        print("GPU disponível para TensorFlow:", tf.test.is_gpu_available())
        
        try:
            # Listar GPUs disponíveis
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print("\nGPUs disponíveis:", len(gpus))
                for gpu in gpus:
                    print(" -", gpu.name)
                
                # Permitir crescimento de memória dinâmico
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Configurar para formato de dados mixed precision
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                
                print("\nGPU configurada com sucesso!")
                print("Usando mixed precision:", policy.name)
                return True
            else:
                print("\nNenhuma GPU encontrada. Usando CPU.")
                return False
                
        except Exception as e:
            print("\nErro ao configurar GPU:", str(e))
            print("Usando CPU como fallback.")
            return False

    # Configurar GPU no início do script
    using_gpu = setup_gpu()

    from tensorflow import keras, data
    import tensorflow_probability as tfp
    from tensorflow.keras import layers, regularizers, activations, optimizers
    from tensorflow.keras import backend as K
    import seaborn as sns
    import matplotlib.pyplot as plt

    dataset_name = "bearing_dataset"  # Apenas para referência
    #train_ratio = 0.75
    row_mark = 740
    batch_size = 128
    time_step = 1
    x_dim = 4
    lstm_h_dim = 8
    z_dim = 4
    epoch_num = 100
    threshold = None

    mode = 'train'
    model_dir = "./lstm_vae_model/"
    image_dir = "./lstm_vae_images/"
    results_dir = 'C:/Users/Augusto-PC/Documents/GitHub/LSTM/Resumo resultados/'

    # Criar diretórios se não existirem
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Ler o diretório de dados do arquivo data_path.txt
    try:
        with open('data_path.txt', 'r') as f:
            data_dir = f.read().strip()
        print(f"Usando diretório de dados: {data_dir}")
    except Exception as e:
        print(f"Erro ao ler data_path.txt: {str(e)}")
        sys.exit(1)

    # Criar diretórios necessários
    os.makedirs(data_dir, exist_ok=True)

    # Parâmetros de ativação
    lstm_activation = 'softplus'  # Pode mudar para 'tanh', 'relu', etc
    sigma_activation = 'tanh'     # Ativação para sigma_x

    def split_normalize_data(all_df):
        #row_mark = int(all_df.shape[0] * train_ratio)
        train_df = all_df[:row_mark]
        test_df = all_df[row_mark:]

        scaler = MinMaxScaler()
        scaler.fit(np.array(all_df)[:, 1:])
        train_scaled = scaler.transform(np.array(train_df)[:, 1:])
        test_scaled = scaler.transform(np.array(test_df)[:, 1:])
        return train_scaled, test_scaled

    def reshape(da):
        return da.reshape(da.shape[0], time_step, da.shape[1]).astype("float32")

    class Sampling(layers.Layer):
        """Camada de amostragem usando o truque de reparametrização"""
        
        def call(self, inputs):
            mu, logvar = inputs
            batch = tf.shape(mu)[0]
            dim = tf.shape(mu)[1]
            
            # Garantir que todos os tensores estejam no mesmo dtype
            dtype = mu.dtype
            epsilon = tf.random.normal(shape=(batch, dim), dtype=dtype)
            
            # Calcular usando o mesmo dtype
            return mu + tf.cast(tf.exp(0.5 * logvar), dtype) * epsilon

    class Encoder(layers.Layer):
        def __init__(self, time_step, x_dim, lstm_h_dim, z_dim, name='encoder', activation=lstm_activation, **kwargs):
            super(Encoder, self).__init__(name=name, **kwargs)
            self.time_step = time_step
            self.x_dim = x_dim
            self.lstm_h_dim = lstm_h_dim
            self.z_dim = z_dim
            self.activation = activation
            
            # Camadas do encoder
            self.lstm = layers.LSTM(
                lstm_h_dim,
                activation=activation,
                return_sequences=True,
                name='encoder_lstm'
            )
            
            self.flatten = layers.Flatten(name='encoder_flatten')
            self.dense = layers.Dense(lstm_h_dim, activation=activation, name='encoder_dense')
            
            # Camadas para média e log variância
            self.z_mean = layers.Dense(z_dim, name='z_mean')
            self.z_log_var = layers.Dense(z_dim, name='z_log_var')
            
            # Camada de amostragem
            self.sampling = Sampling()
        
        def call(self, inputs):
            # Passar pela LSTM
            x = self.lstm(inputs)
            
            # Flatten e dense
            x = self.flatten(x)
            x = self.dense(x)
            
            # Calcular média e log variância
            z_mean = self.z_mean(x)
            z_log_var = self.z_log_var(x)
            
            return z_mean, z_log_var
        
        def sampling(self, inputs):
            """Amostra do espaço latente"""
            z_mean, z_log_var = inputs
            return self.sampling([z_mean, z_log_var])

    class Decoder(layers.Layer):
        def __init__(self, time_step, x_dim, lstm_h_dim, z_dim, name='decoder', activation=lstm_activation, sigma_activation=sigma_activation, **kwargs):
            super(Decoder, self).__init__(name=name, **kwargs)

            self.z_inputs = layers.RepeatVector(time_step, name='repeat_vector')
            self.decoder_lstm_hidden = layers.LSTM(
                lstm_h_dim, 
                activation=activation,
                return_sequences=True, 
                name='decoder_lstm'
            )
            self.x_mean = layers.Dense(x_dim, name='x_mean')
            self.x_sigma = layers.Dense(
                x_dim, 
                name='x_sigma', 
                activation=sigma_activation  # Usar parâmetro
            )
        
        def call(self, inputs):
            z = self.z_inputs(inputs)
            hidden = self.decoder_lstm_hidden(z)
            mu_x = self.x_mean(hidden)
            sigma_x = self.x_sigma(hidden)
            return mu_x, sigma_x
        
        def get_config(self):
            config = super(Decoder, self).get_config()
            config.update({
                'name': self.name
            })
            return config

    class LSTMVAE(keras.Model):
        def __init__(self, time_step, x_dim, lstm_h_dim, z_dim, name='lstm_vae', **kwargs):
            super(LSTMVAE, self).__init__(name=name, **kwargs)
            self.encoder = Encoder(time_step, x_dim, lstm_h_dim, z_dim)
            self.decoder = Decoder(time_step, x_dim, lstm_h_dim, z_dim)
            self.time_step = time_step
            self.x_dim = x_dim
            self.z_dim = z_dim
            self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
            self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
            self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
            self.sampling = Sampling()

        @property
        def metrics(self):
            return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
            ]

        def call(self, inputs):
            z_mean, z_log_var = self.encoder(inputs)
            z = self.sampling([z_mean, z_log_var])
            reconstruction = self.decoder(z)
            return reconstruction

        def predict(self, inputs, batch_size=None, verbose=0, steps=None):
            """Retorna z_mean, z_log_var e log_px para os inputs"""
            # Processar os inputs diretamente
            z_mean, z_log_var = self.encoder(inputs)
            z = self.sampling([z_mean, z_log_var])
            reconstruction = self.decoder(z)
            
            # Calcular log p(x|z)
            log_px = -tf.reduce_sum(
                keras.losses.mse(inputs, reconstruction),
                axis=[1, 2]
            )
            
            return z_mean, z_log_var, log_px

        def train_step(self, data):
            with tf.GradientTape() as tape:
                z_mean, z_log_var = self.encoder(data)
                z = self.sampling([z_mean, z_log_var])
                reconstruction = self.decoder(z)
                
                # Calcular reconstruction loss
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        keras.losses.mse(data, reconstruction),
                        axis=[1, 2]
                    )
                )
                
                # Calcular KL loss
                kl_loss = -0.5 * tf.reduce_mean(
                    tf.reduce_sum(
                        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                        axis=1
                    )
                )
                
                # Loss total
                total_loss = reconstruction_loss + kl_loss

            # Calcular gradientes e atualizar pesos
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            
            # Atualizar métricas
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            
            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }

        def test_step(self, data):
            z_mean, z_log_var = self.encoder(data)
            z = self.sampling([z_mean, z_log_var])
            reconstruction = self.decoder(z)
            
            # Calcular reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mse(data, reconstruction),
                    axis=[1, 2]
                )
            )
            
            # Calcular KL loss
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1
                )
            )
            
            # Loss total
            total_loss = reconstruction_loss + kl_loss
            
            # Atualizar métricas
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            
            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }

    def prepare_training_data(all_df, batch_size=128):
        print("Pre-processing data...")
        
        # Normalizar os dados
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(all_df)
        
        # Reshape para formato LSTM (amostras, time_steps, features)
        n_samples = scaled_data.shape[0] - time_step
        X = np.zeros((n_samples, time_step, scaled_data.shape[1]))
        
        for i in range(n_samples):
            X[i] = scaled_data[i:i + time_step]
        
        # Dividir em treino e teste
        train_size = int(0.8 * n_samples)
        X_train = X[:train_size]
        X_test = X[train_size:]
        
        # Converter para float16
        X_train = tf.cast(X_train, tf.float16)
        X_test = tf.cast(X_test, tf.float16)
        
        # Criar datasets
        train_dataset = tf.data.Dataset.from_tensor_slices(X_train)
        train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        test_dataset = tf.data.Dataset.from_tensor_slices(X_test)
        test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
        test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
        
        print(f"Shape dos dados de treino: {X_train.shape}")
        print(f"Shape dos dados de teste: {X_test.shape}")
        
        return train_dataset, test_dataset, scaler

    def hyperparameter_tuning(train_dataset):
        """Otimização de hiperparâmetros usando keras-tuner"""
        
        class VAEHyperModel(kt.HyperModel):
            def __init__(self, time_step, input_dim):
                super().__init__()
                self.time_step = time_step
                self.input_dim = input_dim
            
            def build(self, hp):
                # Hiperparâmetros
                hp_lstm_dim = hp.Int('lstm_dim', min_value=8, max_value=32, step=4)
                hp_z_dim = hp.Int('z_dim', min_value=2, max_value=16, step=2)
                hp_learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
                
                # Configurar política de mixed precision
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                
                # Criar modelo
                model = LSTMVAE(
                    time_step=self.time_step,
                    x_dim=self.input_dim,
                    lstm_h_dim=hp_lstm_dim,
                    z_dim=hp_z_dim
                )
                
                # Compilar modelo com loss scaling para mixed precision
                optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)
                optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
                model.compile(optimizer=optimizer)
                
                # Construir modelo explicitamente
                dummy_data = tf.zeros((1, self.time_step, self.input_dim), dtype=tf.float16)
                _ = model(dummy_data)
                
                return model
        
        # Criar diretório para o tuner se não existir
        tuner_dir = os.path.join(os.path.dirname(model_dir), 'keras_tuner')
        os.makedirs(tuner_dir, exist_ok=True)
        
        # Obter as dimensões do dataset
        for batch in train_dataset.take(1):
            input_shape = batch.shape
            input_dim = input_shape[-1]
            break
        
        # Calcular tamanho do dataset
        dataset_size = sum(1 for _ in train_dataset)
        val_size = int(0.2 * dataset_size)
        train_size = dataset_size - val_size
        
        # Criar datasets de treino e validação
        train_data = train_dataset.take(train_size)
        val_data = train_dataset.skip(train_size)
        
        print(f"Tamanho do dataset de treino: {train_size}")
        print(f"Tamanho do dataset de validação: {val_size}")
        
        # Configurar tuner
        tuner = kt.RandomSearch(
            hypermodel=VAEHyperModel(
                time_step=time_step,
                input_dim=input_dim
            ),
            objective='val_loss',
            max_trials=5,
            executions_per_trial=1,
            directory=tuner_dir,
            project_name='lstm_vae'
        )
        
        # Configurar callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
        
        # Buscar melhores hiperparâmetros
        tuner.search(
            train_data,
            validation_data=val_data,
            epochs=50,
            callbacks=callbacks
        )
        
        # Obter e mostrar melhores hiperparâmetros
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("\nMelhores hiperparâmetros encontrados:")
        print(f"LSTM dim: {best_hps.get('lstm_dim')}")
        print(f"Z dim: {best_hps.get('z_dim')}")
        print(f"Learning rate: {best_hps.get('learning_rate')}")
        
        # Retornar melhor modelo
        best_model = tuner.get_best_models(num_models=1)[0]
        return best_model

    def generate_sample_data():
        """Generate sample data if no Excel files are found."""
        print("\nGenerating sample data for testing...")
        
        # Create sample timestamps
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='H')
        n_samples = len(dates)
        
        # Generate synthetic data
        np.random.seed(42)
        data = {
            'timestamp': dates,
            'nivel_agua': np.random.normal(100, 10, n_samples),  # Normal distribution
            'vazao': np.abs(np.random.normal(50, 5, n_samples)),  # Positive values
            'temperatura': np.random.normal(25, 3, n_samples),  # Normal distribution
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add some anomalies
        anomaly_idx = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        df.loc[anomaly_idx, 'nivel_agua'] *= 1.5
        df.loc[anomaly_idx, 'vazao'] *= 2
        df.loc[anomaly_idx, 'temperatura'] += 10
        
        # Save to Excel file
        sample_file = os.path.join(data_dir, 'sample_data.xlsx')
        print(f"Saving sample data to {sample_file}")
        df.to_excel(sample_file, index=False)
        return df

    def load_and_prepare_data():
        print("\nCarregando e preparando dados...")
        print(f"Diretório de dados: {data_dir}")
        
        dfs = []
        
        # Get list of Excel files (ignorando arquivos temporários que começam com ~$)
        excel_files = [f for f in glob.glob(os.path.join(data_dir, "*.xlsx")) if not os.path.basename(f).startswith('~$')]
        
        if not excel_files:
            print(f"\nERRO: Nenhum arquivo Excel encontrado em {data_dir}")
            print("\nPor favor, verifique:")
            print("1. Se o diretório está correto")
            print("2. Se os arquivos Excel (.xlsx) foram carregados")
            print("3. Se os arquivos não estão em uma subpasta")
            sys.exit(1)
        
        print(f"\nEncontrados {len(excel_files)} arquivos Excel:")
        for f in excel_files[:5]:
            print(f"  - {os.path.basename(f)}")
        if len(excel_files) > 5:
            print(f"  ... e mais {len(excel_files)-5} arquivos")
        
        # Define required columns
        required_cols = {
            'Data': ['Data', 'DATA', 'data'],
            'temperatura': ['TEMPERATURA MAXIMA, DIARIA(Â°C)', 'TEMPERATURA MAXIMA, DIARIA(°C)', 'TEMPERATURA MAXIMA DIARIA']
        }
        
        # Load and combine data from all files
        for file in excel_files:
            print(f"\nCarregando {os.path.basename(file)}...")
            try:
                # Ler o Excel com parse_dates e dayfirst=True
                df = pd.read_excel(
                    file,
                    parse_dates=['Data'],
                    date_parser=lambda x: pd.to_datetime(x, dayfirst=True)
                )
                print(f"Colunas encontradas: {list(df.columns)}")
                
                # Map actual columns to required columns
                column_mapping_found = {}
                for required_col, possible_names in required_cols.items():
                    found_col = next((col for col in df.columns if col in possible_names), None)
                    if found_col:
                        column_mapping_found[required_col] = found_col
                
                # Check if we found all required columns
                missing = [col for col in required_cols.keys() if col not in column_mapping_found]
                
                if missing:
                    print(f"Aviso: {os.path.basename(file)} está faltando colunas {missing}")
                    print("Colunas esperadas podem ter os seguintes nomes:")
                    for col, alternatives in required_cols.items():
                        if col in missing:
                            print(f"  - {col}: {alternatives}")
                    continue
                
                # Rename columns to standard names
                df = df.rename(columns=dict((v, k) for k, v in column_mapping_found.items()))
                
                # Adicionar identificador da estação
                station_id = os.path.splitext(os.path.basename(file))[0]
                df['station_id'] = station_id
                
                # Remover linhas com valores nulos na temperatura
                df = df.dropna(subset=['temperatura'])
                
                # Verificar se as datas foram convertidas corretamente
                if not pd.api.types.is_datetime64_any_dtype(df['Data']):
                    print(f"Aviso: Falha ao converter datas em {os.path.basename(file)}")
                    continue
                    
                dfs.append(df)
                print(f"Processado com sucesso: {len(df)} registros")
                print(f"Intervalo de datas: {df['Data'].min()} até {df['Data'].max()}")
                
            except Exception as e:
                print(f"Erro ao ler {file}: {str(e)}")
                print("Verifique se o arquivo está corrompido ou se tem o formato esperado")
                continue
        
        if not dfs:
            print("\nErro: Nenhum arquivo válido para processar")
            print("Verifique se os arquivos Excel têm as colunas necessárias nos formatos aceitos:")
            for col, alternatives in required_cols.items():
                print(f"  - {col}: {alternatives}")
            raise ValueError("Nenhum arquivo válido para processar")
        
        # Combine all dataframes
        print("\nCombinando dados de todas as estações...")
        all_df = pd.concat(dfs, ignore_index=True)
        
        # Extrair características temporais da coluna Data
        all_df['year'] = all_df['Data'].dt.year
        all_df['month'] = all_df['Data'].dt.month
        all_df['day'] = all_df['Data'].dt.day
        
        # Selecionar apenas as colunas relevantes para o modelo
        feature_cols = ['temperatura', 'year', 'month', 'day']
        all_df = all_df[feature_cols]
        
        # Update x_dim based on actual number of features
        global x_dim
        x_dim = len(feature_cols)
        print(f"Número de características (x_dim): {x_dim}")
        print(f"Número total de registros: {len(all_df)}")
        
        return all_df

    def plot_loss_moment(history):
        _, ax = plt.subplots(figsize=(14, 6), dpi=80)
        ax.plot(history['loss'], 'blue', label='Loss', linewidth=1)
        ax.plot(history['reconstruction_loss'], 'red', label='Reconstruction loss', linewidth=1)
        ax.plot(history['kl_loss'], 'green', label='KL loss', linewidth=1)
        ax.set_title('Loss and reconstruction loss over epochs')
        ax.set_ylabel('Loss and reconstruction loss')
        ax.set_xlabel('Epoch')
        ax.legend(loc='upper right')
        plt.savefig(image_dir + 'loss_lstm_vae_' + mode + '.png')

    def plot_log_likelihood(df_log_px):
        plt.figure(figsize=(14, 6), dpi=80)
        plt.title("Log likelihood")
        sns.set_color_codes()
        sns.distplot(df_log_px, bins=40, kde=True, rug=True, color='blue')
        plt.savefig(image_dir + 'log_likelihood_' + mode + '.png')

    def save_processing_summary(predictions, originals, log_px, threshold, model):
        """Salva um resumo do processamento em Excel"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = os.path.join(results_dir, f'resumo_processamento_{timestamp}.xlsx')
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Aba 1: Dados de Reconstrução
            df_reconstruction = pd.DataFrame({
                'Original': originals.flatten(),
                'Reconstruído': predictions.flatten(),
                'Diferença': originals.flatten() - predictions.flatten()
            })
            df_reconstruction.to_excel(writer, sheet_name='Reconstrução', index=True)
            
            # Aba 2: Métricas de Erro
            errors = -np.mean(log_px, axis=0)
            if np.isscalar(errors):
                errors = np.array([errors])
                
            df_errors = pd.DataFrame({
                'Log-likelihood': errors,
                'Threshold': threshold,
                'É Anomalia': errors > threshold
            }, index=range(len(errors)))
            df_errors.to_excel(writer, sheet_name='Métricas', index=True)
            
            # Aba 3: Configurações do Modelo
            df_config = pd.DataFrame({
                'Parâmetro': [
                    'Total Parâmetros',
                    'Parâmetros Treináveis',
                    'Batch Size',
                    'Threshold',
                    'Data Processamento',
                    'Diretório Modelo',
                    'Diretório Imagens',
                    'Diretório Resultados'
                ],
                'Valor': [
                    model.count_params(),
                    len(model.trainable_variables),
                    batch_size,
                    threshold,
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    os.path.abspath(model_dir),
                    os.path.abspath(image_dir),
                    os.path.abspath(results_dir)
                ]
            })
            df_config.to_excel(writer, sheet_name='Configurações', index=False)
            
            # Aba 4: Estatísticas
            stats_values = [
                np.mean(originals),
                np.mean(predictions),
                np.std(originals),
                np.std(predictions),
                np.mean(np.abs(originals - predictions)),
                np.mean((originals - predictions)**2),
                np.sum(errors > threshold),
                (np.sum(errors > threshold) / len(errors)) * 100
            ]
            
            df_stats = pd.DataFrame({
                'Métrica': [
                    'Média Original',
                    'Média Reconstruída',
                    'Desvio Padrão Original',
                    'Desvio Padrão Reconstruído',
                    'Erro Médio Absoluto',
                    'Erro Quadrático Médio',
                    'Total Anomalias',
                    '% Anomalias'
                ],
                'Valor': stats_values
            })
            df_stats.to_excel(writer, sheet_name='Estatísticas', index=False)
        
        print(f"\nResumo do processamento salvo em: {excel_path}")
        return excel_path

    def plot_training_results(model, train_dataset, threshold):
        """Plota os resultados do treinamento e salva resumo em Excel"""
        # Coletar predições
        all_predictions = []
        all_originals = []
        all_log_px = []
        
        for batch in train_dataset:
            # O modelo retorna (z_mean, reconstructed)
            z_mean, reconstructed = model(batch)
            all_predictions.append(reconstructed.numpy())
            all_originals.append(batch.numpy())
            _, _, log_px = model.predict(batch)
            all_log_px.append(log_px.numpy())
        
        predictions = np.concatenate(all_predictions)
        originals = np.concatenate(all_originals)
        log_px = np.concatenate(all_log_px)
        
        # Salvar resumo em Excel
        excel_path = save_processing_summary(predictions, originals, log_px, threshold, model)
        
        # Criar figura com subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Reconstrução vs Original
        ax1.plot(predictions[:100, 0, 0], label='Reconstruído', alpha=0.7)
        ax1.plot(originals[:100, 0, 0], label='Original', alpha=0.7)
        ax1.set_title('Comparação entre Dados Originais e Reconstruídos')
        ax1.set_xlabel('Tempo')
        ax1.set_ylabel('Valor')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Log-likelihood e Threshold
        errors = -np.mean(log_px, axis=0)
        ax2.plot(errors, label='Log-likelihood')
        ax2.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        ax2.set_title('Log-likelihood e Threshold')
        ax2.set_xlabel('Amostra')
        ax2.set_ylabel('Log-likelihood')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, 'training_results.png'))
        plt.close()
        
        return excel_path

    def save_model(model):
        """Salva o modelo e seus pesos"""
        # Criar diretório se não existir
        os.makedirs(model_dir, exist_ok=True)
        
        # Salvar pesos do modelo
        weights_path = os.path.join(model_dir, 'lstm_vae.weights.h5')
        model.save_weights(weights_path)
        print(f"Modelo salvo em: {weights_path}")

    def load_model():
        """Carrega o modelo salvo"""
        # Criar modelo com arquitetura padrão
        model = LSTMVAE(
            time_step=time_step,
            x_dim=3,  # número de features
            lstm_h_dim=16,  # dimensão padrão do LSTM
            z_dim=8  # dimensão padrão do espaço latente
        )
        
        # Carregar pesos
        weights_path = os.path.join(model_dir, 'lstm_vae.weights.h5')
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            print(f"Pesos carregados de: {weights_path}")
        else:
            raise FileNotFoundError(f"Arquivo de pesos não encontrado em: {weights_path}")
        
        return model

    def save_model_and_scaler(model, scaler):
        """Salva o modelo, scaler e threshold"""
        # Salvar modelo
        save_model(model)
        
        # Salvar scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler salvo em: {scaler_path}")
        
        # Salvar threshold
        threshold_path = os.path.join(model_dir, 'threshold.npy')
        np.save(threshold_path, threshold)
        print(f"Threshold salvo em: {threshold_path}")

    def main():
        print("\nIniciando processamento...")
        if using_gpu:
            setup_gpu()
        
        try:
            print("\nCarregando dados...")
            all_df = load_and_prepare_data()
            
            print("\nPreparando dados de treino...")
            train_dataset, test_dataset, scaler = prepare_training_data(all_df, batch_size=batch_size)
            
            if mode == "train":
                print("\nIniciando otimização de hiperparâmetros...")
                model = hyperparameter_tuning(train_dataset)
                model.summary()
                
                # Calcular threshold dinâmico (95º percentil dos erros de treino)
                all_train_log_px = []
                for batch in train_dataset:
                    _, _, train_log_px = model.predict(batch)
                    all_train_log_px.append(train_log_px.numpy())
                
                train_errors = -np.mean(np.concatenate(all_train_log_px), axis=0)
                global threshold
                threshold = np.percentile(train_errors, 95)
                
                print(f"Threshold calculado: {threshold}")
                
                # Salvar modelo e scaler
                print("Salvando modelo e scaler...")
                save_model_and_scaler(model, scaler)
                
                # Plotar resultados do treinamento
                print("Gerando visualizações...")
                plot_training_results(model, train_dataset, threshold)
                
                print("\nTreinamento concluído com sucesso!")
                print(f"Modelo salvo em: {model_dir}")
                print(f"Visualizações salvas em: {image_dir}")
                
            elif mode == "infer":
                model = load_model()
                model.compile(optimizer=optimizers.Adam(learning_rate=0.001, epsilon=1e-6, amsgrad=True))
                
                # Carregar threshold
                threshold = np.load(os.path.join(model_dir, 'threshold.npy'))
                print(f"Threshold carregado: {threshold:.4f}")
                
                # Fazer predições
                all_log_px = []
                for batch in train_dataset:
                    _, _, log_px = model.predict(batch)
                    all_log_px.append(log_px.numpy())
                
                train_log_px = np.concatenate(all_log_px)
                train_errors = -np.mean(train_log_px, axis=0)
                
                # Criar DataFrame com resultados
                df_train_log_px = pd.DataFrame()
                df_train_log_px['log_likelihood'] = train_errors
                df_train_log_px['anomaly'] = df_train_log_px['log_likelihood'].apply(
                    lambda x: 1 if x > threshold else 0
                )
                
                # Plotar resultados
                plot_log_likelihood(df_train_log_px, mode='train')
                
                # Mostrar estatísticas das anomalias
                n_anomalies = df_train_log_px['anomaly'].sum()
                print(f"\nEstatísticas das Anomalias:")
                print(f"Total de anomalias detectadas: {n_anomalies}")
                print(f"Porcentagem de anomalias: {(n_anomalies/len(df_train_log_px))*100:.2f}%")
                
                print("\nProcesso de inferência concluído!")
                
            else:
                print(f"\nModo {mode} não reconhecido!")
                exit(1)
            
        except Exception as e:
            print(f"\nErro durante a execução: {str(e)}")
            raise e

    if __name__ == "__main__":
        main()