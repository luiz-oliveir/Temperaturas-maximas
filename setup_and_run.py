import sys
import os
import subprocess
import shutil
from pathlib import Path
import time
from IPython.display import display, HTML

# Check if running in Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def run_command(cmd, check=True, shell=False):
    """Helper function to run commands and handle errors"""
    try:
        print(f"Executando comando: {' '.join(str(x) for x in cmd)}")
        result = subprocess.run(cmd, check=check, shell=shell, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE,
                            universal_newlines=True)
        print(f"Output: {result.stdout}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(str(x) for x in cmd)}")
        print(f"Error output: {e.stderr}")
        return None

def clone_repository():
    """Clone the GitHub repository if in Colab"""
    if IN_COLAB:
        # Change to /content directory
        os.chdir('/content')
        
        # Remove existing directory if it exists
        if os.path.exists('/content/LSTM'):
            print("Removing existing repository...")
            shutil.rmtree('/content/LSTM')
        
        print("Cloning repository...")
        # Clone the repository
        cmd = ["git", "clone", "https://github.com/luiz-oliveir/LSTM.git"]
        result = run_command(cmd)
        
        if result is None or result.returncode != 0:
            print("Failed to clone repository")
            return False
            
        print("Repository cloned successfully")
        return True
    return True

def get_project_dir():
    """Get the project directory in Google Colab or local environment"""
    if IN_COLAB:
        project_dir = Path('/content/LSTM')
    else:
        # For local environment, use the current script's directory
        project_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    print(f"Environment: {'Google Colab' if IN_COLAB else 'Local Windows'}")
    print(f"Project directory: {project_dir}")
    print(f"Path exists: {project_dir.exists()}")
    
    if not project_dir.exists():
        print("Project directory not found")
        return None
        
    print(f"Directory contents: {[f.name for f in project_dir.iterdir()]}")
    
    return project_dir

def setup_environment():
    """Setup the Python environment with required packages"""
    if not clone_repository():
        return False
        
    project_dir = get_project_dir()
    if not project_dir:
        print("Failed to locate project directory")
        return False
    
    # Set data directory based on environment
    if IN_COLAB:
        # Change to project directory to ensure we can find requirements.txt
        os.chdir(project_dir)
        
        data_dir = os.path.join(project_dir, 'Convencionais processadas temperaturas')
        os.makedirs(data_dir, exist_ok=True)
        
        print("\nPreparing Colab environment...")
        
        # Install required packages for Colab
        print("\nInstalando dependências...")
        requirements_path = os.path.join(project_dir, 'requirements.txt')
        if not os.path.exists(requirements_path):
            print(f"Creating requirements.txt at {requirements_path}")
            requirements = """tensorflow>=2.8.0
numpy>=1.19.2
pandas>=1.3.0
scikit-learn>=0.24.2
keras-tuner>=1.1.0
matplotlib>=3.4.3"""
            with open(requirements_path, 'w') as f:
                f.write(requirements)
        
        cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        result = run_command(cmd)
        if not result or result.returncode != 0:
            print("Failed to install requirements")
            return False
        
        # Display upload instructions with HTML formatting
        upload_instructions = """
        <div style='background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin: 10px 0;'>
            <h3 style='color: #2c5282;'>📤 Upload dos Arquivos Excel</h3>
            <p><b>Siga os passos abaixo:</b></p>
            <ol>
                <li>Clique no ícone de pasta 📁 no menu lateral esquerdo do Colab</li>
                <li>Navegue até a pasta: <code>content/LSTM/Convencionais processadas temperaturas</code></li>
                <li>Faça upload dos seus arquivos Excel</li>
            </ol>
            <p style='color: #718096;'><i>Aguardando upload dos arquivos...</i></p>
        </div>
        """
        display(HTML(upload_instructions))
        
        # Wait for file upload confirmation
        while True:
            excel_files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
            if excel_files:
                print(f"\n✅ Encontrados {len(excel_files)} arquivos Excel:")
                for f in excel_files[:5]:
                    print(f"  - {f}")
                if len(excel_files) > 5:
                    print(f"  ... e mais {len(excel_files)-5} arquivos")
                break
            time.sleep(2)  # Check every 2 seconds
            
    else:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Convencionais processadas temperaturas')
        os.makedirs(data_dir, exist_ok=True)
        
        print(f"\nChecking data directory: {data_dir}")
        excel_files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
        
        if not excel_files:
            print("\nWARNING: No Excel files found in the data directory")
            print(f"Please ensure your Excel files are in: {data_dir}")
            return False
            
        print(f"\nFound {len(excel_files)} Excel files")
    
    # Store the data directory path
    with open(os.path.join(project_dir, 'data_path.txt'), 'w') as f:
        f.write(str(data_dir))  # Convert Path to string
    print(f"\nStored data path in data_path.txt")
    
    return True

def run_notebook():
    """Run the main notebook"""
    try:
        project_dir = get_project_dir()
        if not project_dir:
            return False
            
        # Try both filename variants
        notebook_variants = [
            'LSTM_VAE_com_ajustes.ipynb',
            'LSTM_VAE com ajustes.ipynb'
        ]
        
        notebook_path = None
        for variant in notebook_variants:
            temp_path = os.path.join(project_dir, variant)
            if os.path.exists(temp_path):
                notebook_path = temp_path
                break
        
        if not notebook_path:
            print(f"Error: Notebook not found. Tried:")
            for variant in notebook_variants:
                print(f"  - {os.path.join(project_dir, variant)}")
            return False
            
        if IN_COLAB:
            print("\nExecutando o notebook no Colab...")
            from google.colab import files
            import nbformat
            from IPython.display import display
            from IPython import get_ipython
            
            # Ensure we're in the project directory
            os.chdir(project_dir)
            
            # Carregar o notebook
            with open(notebook_path, encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            # Executar cada célula do notebook
            for cell in nb.cells:
                if cell.cell_type == 'code':
                    print("\nExecutando célula:", cell.source[:50] + "..." if len(cell.source) > 50 else cell.source)
                    try:
                        get_ipython().run_cell(cell.source)
                    except SystemExit:
                        # Ignore SystemExit exceptions
                        pass
                    except Exception as e:
                        print(f"Erro ao executar célula: {str(e)}")
                        raise
            
            print("\nNotebook executado com sucesso!")
        else:
            print(f"\nNotebook path: {notebook_path}")
            print("Please open and run the notebook in Jupyter.")
        
        return True
        
    except Exception as e:
        print(f"Error running notebook: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("Starting setup in", "Google Colab" if IN_COLAB else "Local Windows environment")
    if setup_environment():
        run_notebook()
    else:
        print("\nSetup failed. Please check the error messages above.")