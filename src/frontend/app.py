# ==================================================================================================
# frontend.py (v4.2 - Lógica de Duas Etapas para Responsividade Garantida)
#
# DESCRIÇÃO:
# - Implementa uma lógica de duas funções para garantir que a caixa de texto seja
#   limpa imediatamente, resolvendo problemas de responsividade do Gradio.
# ==================================================================================================

import gradio as gr
import requests
import logging

# --- 1. Configuração ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
BACKEND_URL = "http://127.0.0.1:8000/responder"

# --- 2. Funções de Lógica ---

def etapa_1_preparar_ui_e_limpar_caixa(pergunta, historico_chat):
    """ETAPA 1: Executada imediatamente. Atualiza a UI e limpa a caixa de texto."""
    if not pergunta or not pergunta.strip():
        # Se a pergunta for vazia, não faz nada e retorna os valores originais.
        return historico_chat, "", ""

    # Adiciona a pergunta do usuário ao chat com um placeholder de "pensando...".
    novo_historico = historico_chat + [[pergunta, "..."]]
    
    # Retorna:
    # 1. O novo histórico para o chatbot.
    # 2. Uma string vazia para limpar a caixa de texto.
    # 3. A pergunta original para o componente invisível, que vai disparar a etapa 2.
    return novo_historico, "", pergunta

def etapa_2_chamar_backend_e_atualizar_resposta(pergunta, historico_chat):
    """ETAPA 2: Executada em seguida. Chama a API e atualiza a resposta final."""
    if not pergunta or not pergunta.strip():
        # Não faz nada se não houver pergunta para processar.
        return historico_chat

    # Prepara o payload para a API.
    payload = {
        "pergunta": pergunta,
        "session_id": "default-session",
        "embedding_model": "text-embedding-3-small",
        "search_type": "vetorial",
        "top_k": 5,
        "chunking_strategy": "recursive_1000_200",
        "model": "gpt-4o-mini",
        "temperature": 0.1
    }

    try:
        logging.info(f"Enviando payload: {payload}")
        response = requests.post(BACKEND_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        resposta_ia = data.get('resposta', "Erro: campo 'resposta' não encontrado.")
    except Exception as e:
        logging.error(f"Erro na API: {e}")
        resposta_ia = "Desculpe, ocorreu um erro ao tentar conectar ao chatbot."

    # Substitui o placeholder "..." pela resposta real da IA.
    historico_chat[-1][1] = resposta_ia
    
    return historico_chat

# --- 3. Construção da Interface Gráfica ---

with gr.Blocks() as demo:
    gr.Markdown("<h1>Chatbot de Legislação do MPES</h1>")
    
    chatbot = gr.Chatbot(label="Conversa", height=500)
    
    # Componente invisível para passar a pergunta da etapa 1 para a etapa 2.
    pergunta_para_backend_state = gr.Textbox(visible=False)
    
    with gr.Row():
        caixa_pergunta = gr.Textbox(
            label="Sua Pergunta",
            placeholder="Digite sua pergunta aqui...",
            scale=4
        )
        botao_enviar = gr.Button("Enviar", variant="primary", scale=1)

    botao_limpar = gr.ClearButton([caixa_pergunta, chatbot], value="🗑️ Limpar Conversa")

    # --- 4. Conexão dos Eventos em Cadeia ---
    
    # Quando o usuário envia a pergunta (com Enter ou botão)...
    # ...a Etapa 1 é executada.
    botao_enviar.click(
        fn=etapa_1_preparar_ui_e_limpar_caixa,
        inputs=[caixa_pergunta, chatbot],
        outputs=[chatbot, caixa_pergunta, pergunta_para_backend_state]
    ).then(
        # O .then() garante que a Etapa 2 só comece DEPOIS da Etapa 1.
        # Ela é disparada pela mudança no componente invisível 'pergunta_para_backend_state'.
        fn=etapa_2_chamar_backend_e_atualizar_resposta,
        inputs=[pergunta_para_backend_state, chatbot],
        outputs=[chatbot]
    )

    caixa_pergunta.submit(
        fn=etapa_1_preparar_ui_e_limpar_caixa,
        inputs=[caixa_pergunta, chatbot],
        outputs=[chatbot, caixa_pergunta, pergunta_para_backend_state]
    ).then(
        fn=etapa_2_chamar_backend_e_atualizar_resposta,
        inputs=[pergunta_para_backend_state, chatbot],
        outputs=[chatbot]
    )


# --- 5. Execução da Aplicação ---

if __name__ == "__main__":
    print("=" * 60)
    print("== INICIANDO INTERFACE GRÁFICA (v4.2 - Lógica de 2 Etapas) ==")
    print("== Certifique-se de que o backend (chatbot.py) está rodando. ==")
    print("=" * 60)
    demo.queue()
    demo.launch()