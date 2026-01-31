# ==============================================================================
# SCRIPT UNIFICADO: ANÁLISE ART (ANOVA + POST-HOC COMPLETO)
# ==============================================================================

# 1. Carregar bibliotecas
if(!require(ARTool)) install.packages("ARTool")
if(!require(emmeans)) install.packages("emmeans")
if(!require(dplyr)) install.packages("dplyr")

library(ARTool)
library(emmeans)
library(dplyr)

# 2. Definição das métricas a serem analisadas
metricas_alvo <- c("faithfulness_gpt.4o", 
                   "context_recall_gpt.4o", 
                   "context_precision_gpt.4o", 
                   "answer_relevancy_gpt.4o", 
                   "answer_correctness_gpt.4o")

# 3. Função Mestra de Análise
rodar_analise_completa <- function(dados, nome_metrica) {
  
  cat(paste0("\n", paste(rep("=", 60), collapse = ""), "\n"))
  cat(paste(">>> ANALISANDO MÉTRICA:", nome_metrica, "<<<\n"))
  cat(paste0(paste(rep("=", 60), collapse = ""), "\n"))
  
  # --- Preparação dos Dados ---
  # Seleciona colunas e remove NAs
  df_model <- dados[, c(nome_metrica, "chunking_strategy", "search_type", "model", "top_k")]
  df_model <- na.omit(df_model)
  
  # Garante que são fatores
  df_model$chunking_strategy <- as.factor(df_model$chunking_strategy)
  df_model$search_type       <- as.factor(df_model$search_type)
  df_model$model             <- as.factor(df_model$model)
  # Força Top-K como fator ordenado (opcional, mas bom para gráficos) ou fator nominal
  df_model$top_k             <- factor(df_model$top_k, levels = c("5", "10", "15", "20"))
  
  # Renomeia Y dinamicamente
  colnames(df_model)[1] <- "Y"
  
  # --- 1. Ajuste do Modelo ART ---
  # Inclui todas as interações possíveis
  cat("Ajustando modelo ART (pode demorar alguns segundos)...\n")
  m_art <- art(Y ~ chunking_strategy * search_type * model * top_k, data = df_model)
  
  # --- 2. Tabela ANOVA ---
  cat("\n--- Tabela de Significância (ART ANOVA) ---\n")
  print(anova(m_art))
  
  # --- 3. Post-hoc (Contrastes) ---
  # Usamos adjust="bonferroni" para ser conservador e robusto
  
  # A) Chunking Strategy
  cat("\n--- Contrastes: Chunking Strategy ---\n")
  try({
    print(art.con(m_art, "chunking_strategy", adjust = "bonferroni") %>% summary())
  }, silent = TRUE)
  
  # B) Search Type
  cat("\n--- Contrastes: Search Type ---\n")
  try({
    print(art.con(m_art, "search_type", adjust = "bonferroni") %>% summary())
  }, silent = TRUE)
  
  # C) Top-K (Foco: Comparar K=5, 10, 15 contra K=20)
  cat("\n--- Contrastes: Top-K ---\n")
  try({
    print(art.con(m_art, "top_k", adjust = "bonferroni") %>% summary())
  }, silent = TRUE)
  
  # D) Modelo (GPT-4o vs Mini)
  cat("\n--- Contrastes: Modelo LLM ---\n")
  try({
    print(art.con(m_art, "model", adjust = "bonferroni") %>% summary())
  }, silent = TRUE)
  
  cat("\nDone.\n")
}

# 4. Execução do Loop
# Verifica se dados_limpos existe
if(exists("dados_limpos")) {
  for (metrica in metricas_alvo) {
    rodar_analise_completa(dados_limpos, metrica)
  }
} else {
  stop("Erro: O objeto 'dados_limpos' não foi encontrado no ambiente.")
}

library(ggplot2)
library(dplyr)

# 1. Preparar dados resumidos para o gráfico
df_interaction <- dados_limpos %>%
  group_by(chunking_strategy, search_type) %>%
  summarise(
    mean_score = mean(answer_correctness_gpt.4o, na.rm = TRUE),
    se = sd(answer_correctness_gpt.4o, na.rm = TRUE) / sqrt(n())
  ) %>%
  ungroup()

# 2. Plotar Interação (Chunking x Busca)
p_interacao <- ggplot(df_interaction, aes(x = chunking_strategy, y = mean_score, 
                                          color = search_type, group = search_type)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = mean_score - se, ymax = mean_score + se), width = 0.2) +
  labs(title = "Interação: Estratégia de Chunking x Tipo de Busca",
       y = "Média de Answer Correctness",
       x = "Estratégia de Chunking",
       color = "Tipo de Busca") +
  theme_minimal(base_size = 14) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Salvar
ggsave("interacao_chunk_busca.pdf", plot = p_interacao, width = 10, height = 6)
print(p_interacao)