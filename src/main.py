import argparse
import os
import subprocess
import sys
from dotenv import load_dotenv

def run_etl_script(script_name: str):
    """
    Executa um script Python focado em ETL (que termina por conta própria),
    capturando e exibindo sua saída.

    Args:
        script_name (str): O nome do arquivo .py a ser executado.
    Returns:
        bool: True se o script foi executado com sucesso, False caso contrário.
    """
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
    if not os.path.exists(script_path):
        print(f"Erro: O script {script_path} não foi encontrado.")
        return False

    try:
        print(f"Executando {script_name}...")
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            bufsize=1, # Line-buffered
            universal_newlines=True
        )

        # Imprime stdout em tempo real
        if process.stdout:
            for line in process.stdout:
                print(f"[{script_name}] {line}", end='', flush=True)
        
        # Aguarda o término e captura o restante do stderr
        stdout, stderr_output = process.communicate() # Captura qualquer saída restante

        if process.returncode != 0:
            print(f"Erro ao executar {script_name}. Código de retorno: {process.returncode}")
            if stderr_output:
                print(f"Saída de erro de {script_name}:\n{stderr_output}", flush=True)
            return False
        
        print(f"{script_name} concluído com sucesso.")
        return True

    except FileNotFoundError:
        print(f"Erro: O script {script_name} não foi encontrado no caminho especificado: {script_path}.")
        return False
    except Exception as e:
        print(f"Ocorreu uma exceção não esperada ao tentar executar {script_name}: {e}")
        return False

def run_app_interactively(script_name: str):
    """
    Executa um script de aplicativo interativo (como Gradio) permitindo que ele
    use o console diretamente.

    Args:
        script_name (str): O nome do arquivo .py a ser executado.
    Returns:
        bool: True se o script iniciou e foi encerrado (normalmente por CTRL+C),
              False se houve erro na inicialização.
    """
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
    if not os.path.exists(script_path):
        print(f"Erro: O script {script_path} não foi encontrado.")
        return False
    try:
        print(f"\nTentando iniciar {script_name} interativamente...")
        print(f"Se o aplicativo iniciar corretamente, você deverá ver a URL de acesso (ex: http://127.0.0.1:7860).")
        print(f"Pressione CTRL+C no terminal para parar o aplicativo.")
        
        # Permite que o script do app use o stdout/stderr do processo pai (main.py)
        process = subprocess.run([sys.executable, script_path], check=False)
        
        if process.returncode != 0 and process.returncode != -9: # -9 pode ser SIGKILL (CTRL+C as vezes)
             # Em alguns sistemas, CTRL+C pode resultar em códigos como -SIGINT (e.g., -2 no Linux) ou 130
            if process.returncode not in [1, 2, 130, -2]: # Códigos comuns para interrupção por CTRL+C
                 print(f"\n{script_name} terminou com código de erro: {process.returncode}.")
                 return False
        
        print(f"\n{script_name} encerrado.")
        return True
    except KeyboardInterrupt:
        print(f"\n{script_name} interrompido pelo usuário (CTRL+C).")
        return True
    except Exception as e:
        print(f"Ocorreu uma exceção ao tentar executar {script_name}: {e}")
        return False

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Script de orquestração para o Chatbot Jurídico MPES.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # ... (seus argumentos continuam os mesmos)
    parser.add_argument(
        "--collect-data",
        action="store_true",
        help="Executa a coleta de dados (backend/data_collection.py)."
    )
    parser.add_argument(
        "--process-text",
        action="store_true",
        help="Executa o processamento de texto (backend/text_processing.py).\nEste script divide os textos coletados em blocos menores."
    )
    parser.add_argument(
        "--generate-embeddings",
        action="store_true",
        help="Executa a geração de embeddings (backend/embedding_generation.py).\nEste script gera vetores semânticos para os blocos de texto."
    )
    parser.add_argument(
        "--run-app",
        action="store_true",
        help="Inicia a aplicação Gradio do chatbot (frontend/app_gradio.py)."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Executa todas as etapas em sequência: \n1. Coleta de dados\n2. Processamento de texto\n3. Geração de embeddings\n4. Inicia a aplicação Gradio."
    )
    parser.add_argument(
        "--etl-only",
        action="store_true",
        help="Executa apenas as etapas de ETL (Extração, Transformação, Carga):\n1. Coleta de dados\n2. Processamento de texto\n3. Geração de embeddings.\nNão inicia a aplicação Gradio."
    )
    args = parser.parse_args()

    if args.all or args.etl_only:
        print("Iniciando pipeline de ETL...")
        if run_etl_script("backend/data_collection.py"):
            if run_etl_script("backend/text_processing.py"):
                if not run_etl_script("backend/embedding_generation.py"):
                    print("Pipeline ETL interrompido devido a erro na geração de embeddings.")
            else:
                print("Pipeline ETL interrompido devido a erro no processamento de texto.")
        else:
            print("Pipeline ETL interrompido devido a erro na coleta de dados.")
        
        if args.all and not args.etl_only:
            print("\nETL concluído.")
            run_app_interactively("frontend/app_gradio.py")
        elif args.etl_only:
            print("\nPipeline ETL concluído.")

    else:
        if args.collect_data:
            run_etl_script("backend/data_collection.py")
        if args.process_text:
            run_etl_script("backend/text_processing.py")
        if args.generate_embeddings:
            run_etl_script("backend/embedding_generation.py")
        
        # Deve ser 'elif' para evitar que --run-app seja acionado se outro argumento individual foi usado
        # No entanto, a lógica original permitia múltiplas flags. Se isso for intencional, mantenha 'if'.
        # Para clareza, se apenas --run-app é esperado isoladamente ou após --all:
        if args.run_app and not (args.collect_data or args.process_text or args.generate_embeddings or args.all or args.etl_only):
            run_app_interactively("frontend/app_gradio.py")
        elif args.run_app and (args.all or args.etl_only): # Se --run-app foi com --all (já tratado) ou com --etl-only (não deveria rodar app)
            pass # Já tratado ou não aplicável
            

    if not any(vars(args).values()):
        parser.print_help()
        print("\nNenhuma ação especificada. Use uma das opções acima ou --help para mais informações.")