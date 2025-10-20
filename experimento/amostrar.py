import os
import random
from pathlib import Path
import re # <<<< MODIFICADO: Importado para usar expressões regulares na busca

# --- CONFIGURAÇÕES ---

# 1. Pasta que contém todos os seus documentos .txt
PASTA_DOCUMENTOS = "experimento/documentos_txt" # <<<< VERIFIQUE SE O NOME DA SUA PASTA ESTÁ CORRETO

# 2. Quantos documentos você quer selecionar aleatoriamente para a amostra
TAMANHO_AMOSTRA = 100

# 3. Nome do arquivo de saída que conterá o prompt final pronto para uso
ARQUIVO_SAIDA_PROMPT = "experimento/prompt_para_llm_final.txt"

# --- FIM DAS CONFIGURAÇÕES ---


# <<<< MODIFICADO: Prompt aprimorado para refletir que a fonte é um link.
PROMPT_TEMPLATE = """
# CONTEXTO E PERSONA
Você é um analista de dados e especialista em legislação, encarregado de criar um rigoroso conjunto de avaliação para um sistema de chatbot de Inteligência Artificial (RAG).
Sua tarefa é analisar os documentos legais do Ministério Público do Espírito Santo (MP-ES) fornecidos abaixo e gerar um conjunto diversificado e desafiador de
 pares de pergunta e resposta.

# SIMULAÇÃO DE COMPORTAMENTO DO USUÁRIO (MUITO IMPORTANTE)
Formule as perguntas do ponto de vista de um usuário real que **NÃO SABE** em qual documento ou link a resposta se encontra.
As perguntas devem ser naturais e gerais. **NUNCA inclua o link da fonte ou qualquer nome de arquivo no texto da pergunta.**
A fonte (o link) deve ser usada **APENAS** para preencher o campo `"source"` no JSON de saída.

# TAREFA PRINCIPAL
Com base **EXCLUSIVAMENTE** no conteúdo dos documentos fornecidos em `[DOCUMENTOS FORNECIDOS]`, gere um total de **50 perguntas**, divididas igualmente entre as 5 categorias a seguir (exatamente 10 perguntas por categoria).

# CATEGORIAS E INSTRUÇÕES DETALHADAS
(Instruções das categorias 1 a 5 permanecem as mesmas)

1.  **FATUAL (10 perguntas):** "Qual o prazo para a interposição de recursos?".
2.  **SÍNTESE (10 perguntas):** "Quais são as diferenças entre os procedimentos de férias e licença-prêmio, considerando as normas apresentadas?".
3.  **PROCEDIMENTAL (10 perguntas):** "Descreva o procedimento para solicitar o teletrabalho.".
4.  **AGULHA NO PALHEIRO (10 perguntas):** "Qual o valor exato do auxílio-alimentação citado?".
5.  **DEPENDENTE DE HISTÓRICO (10 pares):** "Q1: Qual norma regulamenta o teletrabalho? Q2: E quais são as metas de produtividade mencionadas nela?".

# FORMATO DA SAÍDA
Sua resposta final deve ser **APENAS um array JSON válido**. Não inclua nenhuma introdução. Para cada item, use o **link** fornecido na `Fonte` do documento correspondente para preencher o campo `"source"`.

{{
  "category": "NOME_DA_CATEGORIA",
  "question": "TEXTO_DA_PERGUNTA (GERAL E SEM FONTES)",
  "ground_truth_answer": "A_RESPOSTA_COMPLETA_E_IDEAL_BASEADA_NO_TEXTO",
  "evidence_quote": "A_CITAÇÃO_EXATA_DO_DOCUMENTO_QUE_COMPROVA_A_RESPOSTA",
  "source": "http://link.da.fonte.gov.br/documento"
}}

[DOCUMENTOS FORNECIDOS]
{documentos_formatados}
"""

# <<<< MODIFICADO: Nova função para extrair a fonte de dentro do arquivo.
def extrair_fonte_do_conteudo(texto_documento: str) -> str | None:
    """
    Procura por uma linha no formato 'FONTE: [link]' no texto e extrai o link.
    A busca é case-insensitive e ignora espaços em branco ao redor.
    """
    # Procura por uma linha que começa com "FONTE:", ignorando case e espaços,
    # e captura o que vem depois.
    match = re.search(r"^\s*FONTE\s*:\s*(.*)", texto_documento, re.IGNORECASE | re.MULTILINE)
    if match:
        # Retorna o link, removendo espaços extras
        return match.group(1).strip()
    return None

def criar_prompt_com_amostra():
    """
    Função principal que amostra documentos, combina o conteúdo e gera o prompt final.
    """
    caminho_origem = Path(PASTA_DOCUMENTOS)

    if not caminho_origem.is_dir():
        print(f"ERRO: A pasta de origem '{caminho_origem}' não foi encontrada.")
        return

    todos_os_documentos = list(caminho_origem.glob("*.txt"))
    if not todos_os_documentos:
        print(f"ERRO: Nenhum arquivo .txt encontrado em '{caminho_origem}'.")
        return

    total_docs = len(todos_os_documentos)
    print(f"Encontrados {total_docs} documentos .txt.")

    if total_docs < TAMANHO_AMOSTRA:
        print(f"ERRO: Você quer uma amostra de {TAMANHO_AMOSTRA} documentos, mas só existem {total_docs}.")
        return

    print(f"Selecionando uma amostra aleatória de {TAMANHO_AMOSTRA} documentos...")
    documentos_amostrados = random.sample(todos_os_documentos, TAMANHO_AMOSTRA)

    print("Formatando, extraindo fontes e combinando o conteúdo dos documentos...")
    conteudo_estruturado = []
    tamanho_total_bytes = 0
    for doc_path in documentos_amostrados:
        try:
            texto_documento = doc_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            print(f"AVISO: Não foi possível ler '{doc_path.name}' com UTF-8. Tentando com 'latin-1'.")
            try:
                texto_documento = doc_path.read_text(encoding='latin-1')
            except Exception as e:
                print(f"ERRO: Falha ao ler o arquivo '{doc_path.name}'. Pulando. Erro: {e}")
                continue

        tamanho_total_bytes += len(texto_documento.encode('utf-8'))

        # <<<< MODIFICADO: Usa a nova função para extrair a fonte e tem um fallback.
        fonte_final = extrair_fonte_do_conteudo(texto_documento)
        if not fonte_final:
            print(f"AVISO: Não foi encontrada uma linha 'FONTE: link' no arquivo '{doc_path.name}'. Usando o nome do arquivo como fonte substituta.")
            fonte_final = doc_path.name # Fallback para o nome do arquivo

        # Cria um bloco de texto estruturado para cada documento
        bloco_documento = f"""
-------------------------------------------
### INÍCIO DO DOCUMENTO ###
Fonte: {fonte_final}
Conteúdo:
{texto_documento}
### FIM DO DOCUMENTO ###
-------------------------------------------
"""
        conteudo_estruturado.append(bloco_documento)

    documentos_formatados_final = "".join(conteudo_estruturado)
    prompt_final = PROMPT_TEMPLATE.format(documentos_formatados=documentos_formatados_final)

    caminho_saida = Path(ARQUIVO_SAIDA_PROMPT)
    caminho_saida.write_text(prompt_final, encoding='utf-8')

    tamanho_kb = tamanho_total_bytes / 1024

    print("\n--- SUCESSO! ---")
    print(f"O prompt final foi gerado e salvo em: '{caminho_saida.resolve()}'")
    print(f"O tamanho do conteúdo dos documentos combinados é de aproximadamente {tamanho_kb:.2f} KB.")
    print("Agora, basta abrir este arquivo, copiar todo o conteúdo e colar no NotebookLLM.")


if __name__ == "__main__":
    criar_prompt_com_amostra()