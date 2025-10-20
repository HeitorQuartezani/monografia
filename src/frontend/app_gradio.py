# app.py
import gradio as gr
from chatbot import ChatbotMPES
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

try:
    chatbot_instance = ChatbotMPES()
    print("ChatbotMPES instanciado com sucesso.")
except Exception as e:
    print(f"Erro ao instanciar ChatbotMPES: {e}")
    chatbot_instance = None

# --- Funções do chatbot (chat_interface_streaming, give_feedback) permanecem as mesmas ---
def chat_interface_streaming(user_input, history):
    if chatbot_instance is None:
        history.append((user_input, "Desculpe, o chatbot não está disponível no momento devido a um erro de inicialização."))
        yield history, None, user_input
        return
    if not user_input.strip():
        yield history, None, ""
        return
    history.append((user_input, "..."))
    yield history, None, ""
    try:
        bot_response, interaction_id = chatbot_instance.responder(user_input)
        history[-1] = (user_input, bot_response)
        yield history, interaction_id, ""
    except Exception as e:
        print(f"Erro durante a chamada ao chatbot.responder: {e}")
        error_message = f"Ocorreu um erro ao processar sua pergunta: {str(e)[:100]}..."
        history[-1] = (user_input, error_message)
        yield history, None, ""

def give_feedback(feedback_value, interaction_id_for_feedback):
    if chatbot_instance is None:
        return "Feedback não pode ser registrado: chatbot não inicializado.", interaction_id_for_feedback
    if not interaction_id_for_feedback:
        return "Nenhuma resposta anterior para fornecer feedback (ID não encontrado).", interaction_id_for_feedback
    try:
        chatbot_instance.update_feedback(str(interaction_id_for_feedback), feedback_value)
        feedback_message = f"Feedback ({'positivo' if feedback_value == 1 else 'negativo'}) registrado para a interação ID: {interaction_id_for_feedback}."
        return feedback_message, interaction_id_for_feedback
    except Exception as e:
        print(f"Erro ao registrar feedback: {e}")
        return f"Erro ao registrar feedback: {e}", interaction_id_for_feedback
# --- Fim das funções do chatbot ---

# CSS Customizado focado em fontes apropriadas para ambiente corporativo público
custom_css = """
/* Importa Open Sans e Roboto do Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

/* Aplica a fonte a todo o corpo da aplicação Gradio e elementos principais */
body, .gradio-container, button, .gr-button, input, .gr-textbox, textarea, .gr-markdown, * {
    font-family: 'Open Sans', 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif !important;
}

/* Ajustes opcionais para consistência e legibilidade */
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    font-family: 'Open Sans', 'Roboto', sans-serif !important; /* Mantém consistência nos cabeçalhos */
    font-weight: 600; /* Um peso um pouco mais forte para cabeçalhos */
}

.gr-button {
    font-weight: 600 !important; /* Botões com texto ligeiramente mais destacado */
}

/* Aumentar um pouco o tamanho base da fonte para melhor legibilidade */
/* body, .gradio-container {
    font-size: 15px !important;
} */

/* Garante que o placeholder também use a fonte definida */
::placeholder { /* Chrome, Firefox, Opera, Safari 10.1+ */
  font-family: 'Open Sans', 'Roboto', sans-serif !important;
  opacity: 0.7 !important; /* Adjust placeholder opacity if needed */
}
:-ms-input-placeholder { /* Internet Explorer 10-11 */
  font-family: 'Open Sans', 'Roboto', sans-serif !important;
  opacity: 0.7 !important;
}
::-ms-input-placeholder { /* Microsoft Edge */
  font-family: 'Open Sans', 'Roboto', sans-serif !important;
  opacity: 0.7 !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# Chatbot Jurídico MPES")
    gr.Markdown("Faça sua pergunta sobre a legislação do Ministério Público do Estado do Espírito Santo.")

    chatbot_ui = gr.Chatbot(label="Conversa", height=500, bubble_full_width=False)
    msg_input = gr.Textbox(
        label="Sua Pergunta:",
        placeholder="Digite sua pergunta aqui...",
    )

    interaction_id_hidden_state = gr.State(None)
    submit_button = gr.Button("Enviar Pergunta")

    with gr.Row():
        gr.Markdown("Avalie a última resposta:")
    with gr.Row():
        like_button = gr.Button("👍 Gostei")
        dislike_button = gr.Button("👎 Não Gostei")

    feedback_status_display = gr.Markdown()

    event_args = {
        "fn": chat_interface_streaming,
        "inputs": [msg_input, chatbot_ui],
        "outputs": [chatbot_ui, interaction_id_hidden_state, msg_input]
    }
    msg_input.submit(**event_args)
    submit_button.click(**event_args)

    like_button.click(
        fn=lambda current_id: give_feedback(1, current_id),
        inputs=[interaction_id_hidden_state],
        outputs=[feedback_status_display, interaction_id_hidden_state]
    )
    dislike_button.click(
        fn=lambda current_id: give_feedback(-1, current_id),
        inputs=[interaction_id_hidden_state],
        outputs=[feedback_status_display, interaction_id_hidden_state]
    )

    clear_button = gr.ClearButton(
        components=[msg_input, chatbot_ui, feedback_status_display, interaction_id_hidden_state],
        value="Limpar Conversa e Feedback"
    )
    gr.Markdown("---")
    gr.Markdown("Desenvolvido para auxiliar na consulta de portarias do MPES.")

if __name__ == "__main__":
    if chatbot_instance is None:
        print("Não foi possível iniciar a interface Gradio porque o ChatbotMPES falhou ao inicializar.")
        with gr.Blocks(css=custom_css) as error_demo:
            gr.Markdown("# Erro de Inicialização do Chatbot")
            gr.Markdown("O Chatbot Jurídico MPES não pôde ser iniciado. Verifique os logs do console para mais detalhes...")
        error_demo.launch()
    else:
        print("Iniciando a interface Gradio...")
        demo.launch()