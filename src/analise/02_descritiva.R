# ==============================================================================
# SCRIPT: DISTRIBUIÇÃO DAS MÉTRICAS (EXPORTAÇÃO VETORIAL PDF)
# ==============================================================================
library(ggplot2)
library(dplyr)
library(tidyr)
library(knitr)

# 1. Preparação dos Dados (Formato Longo para o ggplot)
# Certifique-se de que 'dados_limpos' já está carregado no seu ambiente
dados_long <- dados_limpos %>%
  select(faithfulness_gpt.4o, context_recall_gpt.4o, 
         context_precision_gpt.4o, answer_relevancy_gpt.4o, 
         answer_correctness_gpt.4o) %>%
  pivot_longer(cols = everything(), names_to = "Metrica", values_to = "Score") %>%
  mutate(Metrica = dplyr::recode(Metrica,
                                 "faithfulness_gpt.4o" = "Faithfulness",
                                 "context_recall_gpt.4o" = "Context Recall",
                                 "context_precision_gpt.4o" = "Context Precision",
                                 "answer_relevancy_gpt.4o" = "Answer Relevancy",
                                 "answer_correctness_gpt.4o" = "Answer Correctness"))

# 2. Criação do Gráfico (Histogramas)
p <- ggplot(dados_long, aes(x = Score)) +
  # Binwidth ajustado para mostrar bem a granularidade (0.05 = 20 barras)
  geom_histogram(binwidth = 0.05, fill = "#4E79A7", color = "white", alpha = 0.9) +
  
  # Facet wrap para separar os 5 gráficos
  facet_wrap(~Metrica, scales = "free_y", ncol = 3) +
  
  # Estética limpa para publicação acadêmica
  labs(x = "Score da Métrica [0-1]", 
       y = "Frequência (Contagem de Observações)") +
  theme_minimal(base_size = 14) + # Fonte maior para ler bem no PDF
  theme(
    strip.text = element_text(face = "bold", size = 12),
    axis.title = element_text(face = "bold"),
    panel.grid.minor = element_blank() # Remove grades muito finas
  )

# 3. Exportação para PDF (VETORIAL)
# width = largura em polegadas (10 é bom para largura da página A4)
# height = altura (6 deixa proporcional)
ggsave("distribuicao_metricas.pdf", plot = p, device = "pdf", width = 10, height = 6)

# Mensagem de confirmação
print("Gráfico salvo como 'distribuicao_metricas.pdf' com sucesso!")
print(p) # Exibe no plot viewer também


# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================
# Ordem lógica: Recuperação (Recall/Prec) -> Geração (Faith/Rel) -> Global (Corr)
metricas_todas <- c("context_recall_gpt.4o", "context_precision_gpt.4o", 
                    "faithfulness_gpt.4o", "answer_relevancy_gpt.4o", 
                    "answer_correctness_gpt.4o")

labels_todas <- c("Recall", "Precision", "Faithfulness", "Relevancy", "Correctness")

# ==============================================================================
# FUNÇÃO VISUALIZADORA (Média + DP)
# ==============================================================================
visualizar_completa <- function(dados, coluna_grupo, nome_grupo) {
  
  tabela <- dados %>%
    group_by(across(all_of(coluna_grupo))) %>%
    summarise(across(all_of(metricas_todas), 
                     ~ paste0(sprintf("%.2f", mean(., na.rm = TRUE)), 
                              " (", sprintf("%.2f", sd(., na.rm = TRUE)), ")"),
                     .names = "{.col}"))
  
  colnames(tabela) <- c(nome_grupo, labels_todas)
  
  print(kable(tabela, format = "pipe", align = "c", 
              caption = paste("DESEMPENHO COMPLETO: Média (Desvio) por", nome_grupo)))
  cat("\n")
}

# ==============================================================================
# EXECUÇÃO
# ==============================================================================

# 1. Chunking (Impacta tudo?)
visualizar_completa(dados_limpos, "chunking_strategy", "Estratégia Chunking")

# 2. Busca (Impacta tudo?)
visualizar_completa(dados_limpos, "search_type", "Tipo de Busca")

# 3. Top-K (Impacta tudo?)
# Ordenando fator numérico
dados_limpos$top_k_fator <- factor(dados_limpos$top_k, levels = c("5", "10", "15", "20"))
visualizar_completa(dados_limpos, "top_k_fator", "Top-K")

# 4. Modelo (O LLM consegue salvar um Recall ruim? Veremos)
visualizar_completa(dados_limpos, "model", "Modelo LLM")

# ==============================================================================
# 5. EXPORTAÇÃO PARA CSV (GABARITO PARA O LATEX)
# ==============================================================================
library(readr)

cat("\n--- GERANDO CSV UNIFICADO DAS TABELAS ---\n")

# Função auxiliar para criar dataframes limpos
gerar_resumo_df <- function(dados, col_grupo, nome_grupo) {
  
  # Agrupa e calcula média (sd)
  tabela <- dados %>%
    group_by(across(all_of(col_grupo))) %>%
    summarise(across(all_of(metricas_todas), 
                     ~ paste0(sprintf("%.2f", mean(., na.rm = TRUE)), 
                              " (", sprintf("%.2f", sd(., na.rm = TRUE)), ")"),
                     .names = "{.col}")) %>%
    ungroup()
  
  # Renomeia para padronizar
  colnames(tabela)[1] <- "Nivel"
  tabela$Fator <- nome_grupo
  
  # Reorganiza colunas
  tabela <- tabela %>% select(Fator, Nivel, everything())
  return(tabela)
}

# 1. Gerar os pedaços
df_chunk <- gerar_resumo_df(dados_limpos, "chunking_strategy", "Estratégia Chunking")
df_busca <- gerar_resumo_df(dados_limpos, "search_type", "Estratégia Busca")
df_topk  <- gerar_resumo_df(dados_limpos, "top_k", "Top-K")
df_model <- gerar_resumo_df(dados_limpos, "model", "Modelo LLM")

# 2. Juntar tudo em um tabelão
df_final <- bind_rows(df_chunk, df_busca, df_topk, df_model)

# Renomear colunas para ficar igual ao LaTeX
colnames(df_final) <- c("Fator_Experimental", "Nivel_Configuracao", 
                        "Recall_Recuperacao", "Precision_Recuperacao", 
                        "Faithfulness_Geracao", "Relevancy_Geracao", "Correctness_Geracao")

# 3. Salvar em CSV
write_csv(df_final, "tabelas_resumo_completo.csv")

cat("Arquivo 'tabelas_resumo_completo.csv' salvo com sucesso!\n")
cat("Abra este arquivo no Excel para conferir os valores do LaTeX.\n")

# --- VERIFICAÇÃO DE SANIDADE (CHECK DO MODELO) ---
cat("\n--- CHECK DE SEGURANÇA: MODELO ---\n")
print(df_model)

if(df_model[1,3] == df_model[2,3]) {
  cat("\n[OK] As métricas de Recall são IDÊNTICAS entre os modelos.")
  cat("\nIsso confirma que o experimento está correto (Recuperação desacoplada).\n")
} else {
  cat("\n[ATENÇÃO] Há diferenças no Recall entre modelos. Verifique se rodou o Hard Fix.\n")
}