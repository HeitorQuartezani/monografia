# ==============================================================================
# ETAPA 1: LEITURA E LIMPEZA (CORRIGIDO PARA TODAS AS MÉTRICAS)
# ==============================================================================

# 1. Carregar bibliotecas
if(!require(pacman)) install.packages("pacman")
pacman::p_load(dplyr, here, readr)

# 2. Ler o arquivo CSV
caminho_arquivo <- here::here("analise/dados.csv")
dados_brutos <- read_csv(caminho_arquivo)

# 3. Definir nomes das colunas (Verifique se bate com seu CSV!)
# Fatores Experimentais (Variáveis Independentes)
fatores <- c("chunking_strategy", "search_type", "model", "top_k")

# Métricas de Avaliação (Variáveis Dependentes)
# Importante: Estou assumindo que no seu CSV elas têm o sufixo "_gpt.4o"
# Se não tiverem, apague o sufixo aqui na lista.
metricas <- c(
  "faithfulness_gpt.4o", 
  "context_recall_gpt.4o", 
  "context_precision_gpt.4o", 
  "answer_relevancy_gpt.4o", 
  "answer_correctness_gpt.4o"
)

# 4. Seleção e Conversão de Tipos
dados_limpos <- dados_brutos %>%
  # Seleciona TODAS as colunas relevantes (Fatores + 5 Métricas)
  select(all_of(c(fatores, metricas))) %>%
  
  # Remove linhas que tenham NA em QUALQUER uma das colunas selecionadas
  na.omit() %>%
  
  # Converte as variáveis independentes para Fator (essencial para ANOVA)
  mutate(
    chunking_strategy = as.factor(chunking_strategy),
    search_type = as.factor(search_type),
    model = as.factor(model),
    top_k = as.factor(top_k) # Numérico vira Categórico aqui
  )

# --- VERIFICAÇÃO ---
print("--- Estrutura dos Dados Limpos ---")
str(dados_limpos)

print(paste("Total de observações válidas:", nrow(dados_limpos)))

# Verifica se sobrou alguma métrica zerada ou estranha
summary(dados_limpos[metricas])