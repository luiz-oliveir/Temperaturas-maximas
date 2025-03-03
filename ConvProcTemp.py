import pandas as pd
import os
import logging
from tqdm import tqdm
import traceback
from dateutil.relativedelta import relativedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('csv_conversion.log'),
        logging.StreamHandler()
    ]
)

def create_summary_report(output_dir, processing_stats):
    """Create summary reports of the processing results."""
    # Texto summary
    summary_path = os.path.join(output_dir, 'processing_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("Processing Summary Report\n")
        f.write("=======================\n\n")
        
        for stats in processing_stats:
            f.write(f"File: {stats['arquivo']}\n")
            f.write(f"Status: {stats['status']}\n")
            f.write(f"Rows removed: {stats['rows_removed']}\n")
            if stats['erro']:
                f.write(f"Error: {stats['erro']}\n")
            f.write("-----------------------\n")
    
    # Excel summary
    excel_summary_path = os.path.join(output_dir, 'processing_summary.xlsx')
    summary_data = []
    
    for stats in processing_stats:
        row = {
            'Arquivo': stats['arquivo'],
            'Status': stats['status'],
            'Data Inicial': stats.get('data_inicial', None),
            'Data Final': stats.get('data_final', None),
            'Período (anos)': stats.get('periodo_anos', None),
            'Erro': stats['erro']
        }
        summary_data.append(row)
    
    df_summary = pd.DataFrame(summary_data)
    
    # Formatando as datas no DataFrame
    for col in ['Data Inicial', 'Data Final']:
        if df_summary[col].notna().any():
            df_summary[col] = pd.to_datetime(df_summary[col]).dt.strftime('%d/%m/%Y')
    
    # Formatando o período com uma casa decimal
    if df_summary['Período (anos)'].notna().any():
        df_summary['Período (anos)'] = df_summary['Período (anos)'].apply(lambda x: f'{x:.1f}' if pd.notna(x) else x)
    
    df_summary.to_excel(excel_summary_path, index=False, engine='openpyxl')
    
    logging.info(f"Summary reports created at:\n- {summary_path}\n- {excel_summary_path}")

def process_excel_files(input_dir, output_dir):
    """
    Process Excel files in the input directory and save filtered data to the output directory.
    """
    if not os.path.exists(input_dir):
        logging.error(f"Input directory does not exist: {input_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    excel_files = [f for f in os.listdir(input_dir) if f.endswith('.xlsx')]
    
    if not excel_files:
        logging.error(f"No Excel files found in input directory: {input_dir}")
        return
    
    logging.info(f"Found {len(excel_files)} Excel files to process")
    processing_stats = []
    
    for file in tqdm(excel_files, desc="Processing files", unit="file"):
        file_stats = {'arquivo': file, 'status': 'failed', 'rows_removed': 0, 'erro': ''}
        
        try:
            input_path = os.path.join(input_dir, file)
            
            # Extrair o código da estação do nome do arquivo
            # Tenta primeiro o formato completo, senão usa o nome do arquivo sem extensão
            try:
                codigo_estacao = file.split('_')[1]
            except IndexError:
                codigo_estacao = os.path.splitext(file)[0]  # Remove a extensão .xlsx
                
            output_path = os.path.join(output_dir, f"{codigo_estacao}.xlsx")
            
            # Ler o arquivo Excel
            df = pd.read_excel(input_path, engine='openpyxl')
            initial_rows = len(df)
            
            if df.empty:
                file_stats['erro'] = 'Arquivo vazio'
                processing_stats.append(file_stats)
                continue
            
            logging.info(f"\nProcessando arquivo: {file}")
            logging.info(f"Colunas originais: {list(df.columns)}")
            
            # Passo 1: Remover colunas desnecessárias
            colunas_para_remover = ['D', 'F', 'G', 'H']  # Colunas a serem removidas
            indices_para_remover = [ord(col) - ord('A') for col in colunas_para_remover]
            
            if len(df.columns) > max(indices_para_remover):
                colunas_removidas = [df.columns[i] for i in indices_para_remover if i < len(df.columns)]
                df = df.drop(columns=colunas_removidas, axis=1)
                logging.info(f"Colunas removidas: {colunas_removidas}")
            
            if 'Unnamed: 6' in df.columns:
                df = df.drop('Unnamed: 6', axis=1)
            
            logging.info(f"Colunas após remoção: {list(df.columns)}")
            
            # Passo 2: Criar coluna de data única
            try:
                # Verificar se as colunas de data existem
                if not all(col in df.columns for col in ['Dia', 'Mes', 'Ano']):
                    raise ValueError(f"Colunas de data não encontradas. Colunas atuais: {list(df.columns)}")
                
                # Converter para string e formatar
                df['Dia'] = df['Dia'].astype(str).str.zfill(2)
                df['Mes'] = df['Mes'].astype(str).str.zfill(2)
                df['Ano'] = df['Ano'].astype(str)
                
                # Criar coluna Data
                df['Data'] = pd.to_datetime(df['Ano'] + '-' + df['Mes'] + '-' + df['Dia'])
                
                # Formatar a data como dd/mm/aaaa
                df['Data'] = df['Data'].dt.strftime('%d/%m/%Y')
                
                # Remover colunas antigas de data
                df = df.drop(columns=['Dia', 'Mes', 'Ano'])
                
                # Mover coluna Data para o início
                cols = df.columns.tolist()
                cols = ['Data'] + [col for col in cols if col != 'Data']
                df = df[cols]
                
                # Remover linhas com células vazias, em branco ou com valor zero
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                # Remove linhas onde todas as colunas numéricas são nulas ou zero
                df = df.dropna(subset=numeric_cols, how='all')  # Remove linhas com todas as colunas numéricas nulas
                df = df[~(df[numeric_cols] == 0).all(axis=1)]  # Remove linhas onde todas as colunas numéricas são zero
                
                # Renomear a coluna B para "Temperatura Maxima"
                colunas = df.columns.tolist()
                if len(colunas) > 1:  # Verifica se existe a coluna B
                    df = df.rename(columns={colunas[1]: 'Temperatura Maxima'})
                
                # Reconverter para datetime para cálculos
                data_inicio = pd.to_datetime(df['Data'].min(), format='%d/%m/%Y')
                data_fim = pd.to_datetime(df['Data'].max(), format='%d/%m/%Y')
                diff = relativedelta(data_fim, data_inicio)
                periodo_anos = diff.years + (diff.months / 12.0) + (diff.days / 365.25)
                ano_final = data_fim.year
                
                logging.info(f"Data inicial: {data_inicio}")
                logging.info(f"Data final: {data_fim}")
                logging.info(f"Período em anos: {periodo_anos:.2f}")
                logging.info(f"Ano final: {ano_final}")
                
                # Verificar critérios e salvar
                if periodo_anos >= 5 and ano_final >= 2000:
                    df.to_excel(output_path, index=False, engine='openpyxl')
                    file_stats['status'] = 'success'
                    file_stats['data_inicial'] = data_inicio
                    file_stats['data_final'] = data_fim
                    file_stats['periodo_anos'] = periodo_anos
                    logging.info("Arquivo salvo com sucesso")
                else:
                    if periodo_anos < 5:
                        file_stats['erro'] = f'Período insuficiente: {periodo_anos:.1f} anos'
                    else:
                        file_stats['erro'] = f'Ano final ({ano_final}) anterior a 2000'
                    file_stats['status'] = 'skipped'
                    logging.info(f"Arquivo ignorado: {file_stats['erro']}")
            
            except Exception as e:
                file_stats['erro'] = f'Erro ao processar datas: {str(e)}'
                logging.error(traceback.format_exc())
            
        except Exception as e:
            file_stats['erro'] = str(e)
            logging.error(f"Erro processando {file}: {str(e)}")
            logging.error(traceback.format_exc())
            df = pd.DataFrame()  # Criar DataFrame vazio em caso de erro
        
        # Calcular linhas removidas apenas se o DataFrame foi criado com sucesso
        if 'df' in locals() and not df.empty:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            df = df.dropna(subset=numeric_cols, how='all')
            rows_removed = initial_rows - len(df)
            file_stats['rows_removed'] = rows_removed
        else:
            file_stats['rows_removed'] = 0
        
        processing_stats.append(file_stats)
    
    create_summary_report(output_dir, processing_stats)
    
    logging.info("\nProcessing Summary:")
    logging.info(f"Total files processed: {len(excel_files)}")
    logging.info(f"Successfully processed: {sum(1 for s in processing_stats if s['status'] == 'success')}")
    logging.info(f"Skipped (period < 5 years): {sum(1 for s in processing_stats if s['status'] == 'skipped' and 'insuficiente' in s['erro'])}")
    logging.info(f"Skipped (year < 2000): {sum(1 for s in processing_stats if s['status'] == 'skipped' and 'anterior a 2000' in s['erro'])}")
    logging.info(f"Failed processing: {sum(1 for s in processing_stats if s['status'] == 'failed')}")

def main():
    input_dir = "/content/LSTM/Convencionais processadas temperaturas"
    output_dir = "/content/LSTM/Convencionais processadas temperaturas"
    process_excel_files(input_dir, output_dir)

if __name__ == "__main__":
    main()