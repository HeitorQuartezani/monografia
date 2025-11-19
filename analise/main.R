# ==============================================================================
# SCRIPT DE ANÁLISE ESTATÍSTICA (ANOVA) PARA RESULTADOS DE RAG
#
# MELHORIA (v1.6):
# - Adiciona o pacote 'here' para gerenciar caminhos de arquivos.
# - Isso garante que o script leia o CSV e salve o arquivo de resultados
#   na pasta do projeto (ex: C:\monografia), e não na pasta
#   "Documentos" do usuário.
# ==============================================================================

# --- 1. Carregar Bibliotecas ---
library(dplyr)    # Para manipulação de dados
library(broom)    # Para limpar os resultados do modelo (tidy)
library(car)      # Para o Teste de Levene
library(ggpubr)   # Para visualização
library(here)     # <-- ADICIONADO: Para gerenciar caminhos

# --- 2. Configuração dos Nomes ---

# O 'here::here()' garante que o R procure o arquivo na pasta raiz do projeto
ARQUIVO_CSV <- here::here("analise/dados.csv")
ARQUIVO_SAIDA_ANALISE <- here::here("analise_estatistica_resultados.txt")

# Se suas colunas de score não têm sufixo (ex: _gpt-4o), deixe esta variável como ""
LLM_JUIZ_SUFFIX <- "_gpt.4o" 

# --- 3. Carregar e Limpar os Dados ---

# Limpa o arquivo de resultados anterior (usando o caminho 'here')
if (file.exists(ARQUIVO_SAIDA_ANALISE)) {
  file.remove(ARQUIVO_SAIDA_ANALISE)
  print(paste("Arquivo de resultados anterior ('", ARQUIVO_SAIDA_ANALISE, "') removido."))
}

# Carrega os dados (usando o caminho 'here')
dados <- read.csv(ARQUIVO_CSV)

# (O restante desta seção permanece o mesmo)
fatores_independentes_base <- c(
  "chunking_strategy", "search_type", "model", 
  "top_k", "temperature", "system_prompt_override"
)
metricas_dependentes_base <- c(
  "faithfulness", "answer_relevancy", "context_recall", 
  "context_precision", "answer_correctness"
)
metricas_dependentes <- paste0(metricas_dependentes_base, LLM_JUIZ_SUFFIX)

colunas_necessarias <- c(fatores_independentes_base, metricas_dependentes)
colunas_ausentes <- setdiff(colunas_necessarias, names(dados))
if (length(colunas_ausentes) > 0) {
  print(paste("AVISO: Colunas não encontradas e removidas da análise:", paste(colunas_ausentes, collapse=", ")))
  metricas_dependentes <- setdiff(metricas_dependentes, colunas_ausentes)
  fatores_independentes_base <- setdiff(fatores_independentes_base, colunas_ausentes)
}
colunas_para_analise <- c(fatores_independentes_base, metricas_dependentes)
dados_limpos <- dados %>%
  mutate(across(all_of(fatores_independentes_base), as.factor)) %>%
  select(all_of(colunas_para_analise)) %>%
  na.omit()
print(paste("Dados carregados e limpos. Total de linhas para análise:", nrow(dados_limpos)))

# --- 4. VERIFICAÇÃO DE FATORES ---
print("Verificando os níveis de cada fator após a limpeza de dados...")
fatores_para_analise <- c() 
for (fator in fatores_independentes_base) {
  if (!fator %in% names(dados_limpos)) next # Pula se o fator foi removido
  
  niveis <- length(unique(dados_limpos[[fator]]))
  print(paste("Fator:", fator, "possui", niveis, "níveis (grupos)."))
  if (niveis > 1) {
    fatores_para_analise <- c(fatores_para_analise, fator)
  } else {
    print(paste("AVISO: Fator '", fator, "' será REMOVIDO da análise (ANOVA) pois tem 1 ou 0 níveis."))
  }
}
if (length(fatores_para_analise) == 0) {
  stop("ERRO: Nenhum fator com 2+ níveis foi encontrado. Impossível rodar o ANOVA.")
}
print(paste("Fatores que SERÃO USADOS na análise:", paste(fatores_para_analise, collapse=", ")))


# --- 5. Função Genérica para Rodar Análise Completa ---

executar_analise_anova <- function(dados, metrica_nome, fatores_validos, arquivo_saida) {
  
  print(paste("--- INICIANDO ANÁLISE PARA:", metrica_nome, "---"))
  
  # Adiciona um título para esta métrica no arquivo de saída
  write(paste("\n\n==============================================================================="), file = arquivo_saida, append = TRUE)
  write(paste("=== ANÁLISE ESTATÍSTICA PARA A MÉTRICA:", metrica_nome, "==="), file = arquivo_saida, append = TRUE)
  write(paste("===============================================================================\n"), file = arquivo_saida, append = TRUE)
  
  formula_anova <- as.formula(paste(
    metrica_nome, "~", paste(fatores_validos, collapse = " + ")
  ))
  
  # 5a. Rodar o Modelo ANOVA e Salvar Resultados Limpos
  modelo <- aov(formula_anova, data = dados)
  anova_resultados <- tidy(modelo)
  
  print("Resultados do ANOVA (Sumário do Modelo):")
  print(anova_resultados)
  
  write("--- Tabela ANOVA (Sumário do Modelo) ---", file = arquivo_saida, append = TRUE)
  write(capture.output(print(anova_resultados, width = 120)), file = arquivo_saida, append = TRUE)
  
  
  # 5b. Verificar Pressupostos (Imprime apenas no console)
  residuos_modelo <- residuals(modelo)
  if (length(residuos_modelo) > 5000) {
    shapiro_teste <- shapiro.test(sample(residuos_modelo, 5000))
  } else {
    shapiro_teste <- shapiro.test(residuos_modelo)
  }
  if (shapiro_teste$p.value < 0.05) {
    print("AVISO (Normalidade): O p-valor do Shapiro-Wilk é < 0.05. Os resíduos NÃO são normais.")
  }
  
  print("Teste de Homogeneidade (Levene):")
  for (fator in fatores_validos) {
    formula_levene <- as.formula(paste(metrica_nome, "~", fator))
    print(paste("--- Resultado do Teste de Levene para o fator:", fator, "---"))
    print(leveneTest(formula_levene, data = dados))
  }
  
  
  # 5c. Teste Post-Hoc (Tukey HSD) e Salvar Resultados Limpos
  write("\n\n--- Resultados do Teste Post-Hoc (Tukey HSD) ---", file = arquivo_saida, append = TRUE)
  
  for (fator in fatores_validos) {
    print(paste("Analisando Tukey para:", fator))
    tukey_teste <- TukeyHSD(modelo, which = fator)
    tukey_resultados <- tidy(tukey_teste)
    
    write(paste("\n--- Comparação Post-Hoc (Tukey) para:", fator, "---"), file = arquivo_saida, append = TRUE)
    write(capture.output(print(tukey_resultados, width = 120, n = 50)), file = arquivo_saida, append = TRUE)
  }
  
  print(paste("--- ANÁLISE PARA:", metrica_nome, "CONCLUÍDA ---"))
}


# --- 6. Executar as Análises ---

for (metrica in metricas_dependentes) {
  executar_analise_anova(dados_limpos, metrica, fatores_para_analise, ARQUIVO_SAIDA_ANALISE)
}

print("\n\n--- ANÁLISE COMPLETA ---")
print(paste("Todos os resultados estatísticos foram salvos em:", ARQUIVO_SAIDA_ANALISE))