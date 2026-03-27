# ==============================================================================
# SCRIPT FINAL: GERADOR DE TABELAS LATEX (TCC)
# ==============================================================================

# 1. Carregar bibliotecas
if(!require(ARTool)) install.packages("ARTool")
if(!require(emmeans)) install.packages("emmeans")
if(!require(dplyr)) install.packages("dplyr")
if(!require(tidyr)) install.packages("tidyr")
if(!require(stringr)) install.packages("stringr")

library(ARTool)
library(emmeans)
library(dplyr)
library(tidyr)
library(stringr)

# --- DEFINIÇÕES ---
metricas_alvo <- c("context_recall_gpt.4o", 
                   "context_precision_gpt.4o", 
                   "faithfulness_gpt.4o", 
                   "answer_relevancy_gpt.4o", 
                   "answer_correctness_gpt.4o")

# Nomes bonitos para as colunas no LaTeX
nomes_metricas_latex <- c("Recall", "Precision", "Faithfulness", "Relevancy", "Correctness")
names(nomes_metricas_latex) <- metricas_alvo

# Dataframes para acumular resultados brutos
db_anova <- data.frame()
db_contrastes <- data.frame()

# ==============================================================================
# 2. LOOP DE ANÁLISE (PREENCHIMENTO DOS DADOS)
# ==============================================================================

cat(">>> Iniciando processamento estatístico (Isso pode levar 1-2 minutos)...\n")

if(!exists("dados_limpos")) stop("ERRO: O objeto 'dados_limpos' não existe. Carregue seus dados primeiro!")

for (metrica in metricas_alvo) {
  cat(paste(" Processando:", metrica, "...\n"))
  
  # Preparar dados temporários
  df_temp <- dados_limpos[, c(metrica, "chunking_strategy", "search_type", "model", "top_k")]
  df_temp <- na.omit(df_temp)
  colnames(df_temp)[1] <- "Y"
  
  # Garantir fatores
  df_temp$top_k <- factor(df_temp$top_k, levels = c("5", "10", "15", "20"))
  df_temp$model <- as.factor(df_temp$model)
  df_temp$chunking_strategy <- as.factor(df_temp$chunking_strategy)
  df_temp$search_type <- as.factor(df_temp$search_type)
  
  # 1. Modelo ART
  m_art <- art(Y ~ chunking_strategy * search_type * model * top_k, data = df_temp)
  
  # 2. Extrair ANOVA (F-Values)
  a <- anova(m_art)
  a$Fator <- rownames(a)
  a$Metrica <- metrica
  db_anova <- bind_rows(db_anova, a)
  
  # 3. Extrair Contrastes (Post-hoc)
  # A) Busca
  c_search <- summary(art.con(m_art, "search_type", adjust = "bonferroni"))
  c_search$Tipo <- "Busca"
  c_search$Metrica <- metrica
  db_contrastes <- bind_rows(db_contrastes, c_search)
  
  # B) Top-K
  c_topk <- summary(art.con(m_art, "top_k", adjust = "bonferroni"))
  c_topk$Tipo <- "TopK"
  c_topk$Metrica <- metrica
  db_contrastes <- bind_rows(db_contrastes, c_topk)
  
  # C) Modelo
  c_model <- summary(art.con(m_art, "model", adjust = "bonferroni"))
  c_model$Tipo <- "Modelo"
  c_model$Metrica <- metrica
  db_contrastes <- bind_rows(db_contrastes, c_model)
}

# ==============================================================================
# 3. FUNÇÕES DE FORMATAÇÃO LATEX
# ==============================================================================

formatar_valor <- function(est, p_val, is_f_value=FALSE) {
  # Define estrelas
  sig <- ""
  if (p_val < 0.001) { sig <- "***" }
  else if (p_val < 0.01) { sig <- "**" }
  else if (p_val < 0.05) { sig <- "*" }
  else { sig <- "\\textsuperscript{ns}" }
  
  # Formata número
  val_fmt <- sprintf("%.2f", est)
  
  # Adiciona sinal de + se for contraste positivo (não aplica para F-value)
  if(!is_f_value && est > 0) { val_fmt <- paste0("+", val_fmt) }
  
  return(paste0(val_fmt, sig))
}

# Função auxiliar para pivotar e criar a linha LaTeX
gerar_linhas_tabela <- function(df_filtrado, coluna_nome_linha) {
  # Pivota para ter métricas nas colunas
  df_wide <- df_filtrado %>%
    mutate(Valor_Formatado = mapply(formatar_valor, estimate, p.value)) %>%
    select(all_of(coluna_nome_linha), Metrica, Valor_Formatado) %>%
    pivot_wider(names_from = Metrica, values_from = Valor_Formatado)
  
  # Reordena colunas conforme ordem lógica (Recall -> Precision -> Faith -> Rel -> Corr)
  cols_ordem <- metricas_alvo[metricas_alvo %in% names(df_wide)]
  df_wide <- df_wide[, c(coluna_nome_linha, cols_ordem)]
  
  # Gera string LaTeX
  linhas_latex <- apply(df_wide, 1, function(row) {
    paste(row, collapse = " & ")
  })
  
  return(paste(linhas_latex, collapse = " \\\\ \n"))
}

# ==============================================================================
# 4. GERAÇÃO DO ARQUIVO DE SAÍDA
# ==============================================================================

arquivo_saida <- "tabelas_latex_prontas.txt"
sink(arquivo_saida)

cat("%%% ARQUIVO GERADO AUTOMATICAMENTE PELO R %%%\n")
cat("%%% Copie o conteúdo abaixo dos cabeçalhos para o seu LaTeX %%%\n\n")

# --- TABELA 1: CONTRASTES DE BUSCA ---
cat("% ==================================================\n")
cat("% TABELA 1: Contrastes de Busca (Baseline: Híbrida)\n")
cat("% ==================================================\n")

# Filtra contrastes que envolvem Híbrida
df_busca <- db_contrastes %>% 
  filter(Tipo == "Busca") %>%
  filter(str_detect(contrast, "hibrida"))

# Ajuste de sinal: Se o R fez (textual - hibrida), queremos mostrar a diferença
# Mantemos o original do R, apenas garantimos que o nome da linha reflete a conta
cat(gerar_linhas_tabela(df_busca, "contrast"))
cat(" \\\\ \n\n")


# --- TABELA 2: CONTRASTES TOP-K ---
cat("% ==================================================\n")
cat("% TABELA 2: Contrastes Top-K (Baseline: K=20)\n")
cat("% ==================================================\n")

# Filtra contrastes contra K=20
df_topk <- db_contrastes %>% 
  filter(Tipo == "TopK") %>%
  filter(str_detect(contrast, "20"))

cat(gerar_linhas_tabela(df_topk, "contrast"))
cat(" \\\\ \n\n")


# --- TABELA 3: CONTRASTES MODELO ---
cat("% ==================================================\n")
cat("% TABELA 3: Contrastes Modelo (GPT-4o vs Mini)\n")
cat("% ==================================================\n")

df_mod <- db_contrastes %>% 
  filter(Tipo == "Modelo")

cat(gerar_linhas_tabela(df_mod, "contrast"))
cat(" \\\\ \n\n")


# --- TABELA 4: ANOVA F-VALUES ---
cat("% ==================================================\n")
cat("% TABELA 4: Sumário ANOVA (Valores F)\n")
cat("% ==================================================\n")

# Formata F-Values
df_anova_fmt <- db_anova %>%
  filter(Fator %in% c("chunking_strategy", "search_type", "top_k", "model")) %>%
  mutate(Valor_Formatado = mapply(formatar_valor, `F value`, `Pr(>F)`, is_f_value=TRUE)) %>%
  select(Fator, Metrica, Valor_Formatado) %>%
  pivot_wider(names_from = Metrica, values_from = Valor_Formatado)

# Reordena colunas
cols_ordem <- metricas_alvo[metricas_alvo %in% names(df_anova_fmt)]
df_anova_fmt <- df_anova_fmt[, c("Fator", cols_ordem)]

linhas_anova <- apply(df_anova_fmt, 1, function(row) {
  paste(row, collapse = " & ")
})
cat(paste(linhas_anova, collapse = " \\\\ \n"))
cat(" \\\\ \n")

sink()

cat(paste0("\nSUCESSO! Arquivo gerado: ", getwd(), "/", arquivo_saida, "\n"))
cat("Abra o arquivo .txt e copie os dados para o Overleaf/LaTeX.\n")