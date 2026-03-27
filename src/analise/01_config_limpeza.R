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


# ==============================================================================
# SCRIPT DE AUDITORIA: VERIFICAÇÃO DE CONSISTÊNCIA DE RECUPERAÇÃO
# ==============================================================================
library(dplyr)

cat(">>> INICIANDO AUDITORIA DE DADOS (CSI MODE) <<<\n")

# 1. Definir o que constitui uma "Configuração de Recuperação"
# Se essas 3 coisas são iguais, o retrieval TEM que ser igual (fisicamente).
cols_recuperacao <- c("chunking_strategy", "search_type", "top_k")

# 2. Agrupar e Calcular a Variação (Amplitude) dentro de cada grupo
auditoria <- dados_limpos %>%
  group_by(across(all_of(cols_recuperacao))) %>%
  summarise(
    # Conta quantos modelos temos nesse grupo (deve ser 2: 4o e mini)
    n_runs = n(),
    
    # Calcula a diferença entre o maior e o menor valor encontrado para a MESMA config
    diff_recall = max(context_recall_gpt.4o, na.rm = TRUE) - min(context_recall_gpt.4o, na.rm = TRUE),
    diff_precision = max(context_precision_gpt.4o, na.rm = TRUE) - min(context_precision_gpt.4o, na.rm = TRUE),
    
    # Pega os valores reais para visualizarmos se houver erro
    vals_recall = paste(unique(context_recall_gpt.4o), collapse = " vs "),
    vals_precision = paste(unique(context_precision_gpt.4o), collapse = " vs "),
    
    .groups = "drop"
  )

# 3. Filtrar inconsistências
# Consideramos inconsistência qualquer diferença maior que 0.00001 (para evitar erros de ponto flutuante)
inconsistencias <- auditoria %>%
  filter(diff_recall > 0.00001 | diff_precision > 0.00001)

# 4. Relatório Final
total_configs <- nrow(auditoria)
total_erros <- nrow(inconsistencias)

cat(paste0("\n=== RESUMO DA AUDITORIA ===\n"))
cat(paste0("Total de Configurações Únicas testadas: ", total_configs, "\n"))
cat(paste0("Configurações com Recuperação IDÊNTICA: ", total_configs - total_erros, "\n"))
cat(paste0("Configurações com DIVERGÊNCIA (Ruído):   ", total_erros, "\n"))

if(total_erros > 0) {
  cat("\n[ALERTA] Encontramos divergências! Isso confirma o ruído do 'LLM-as-a-Judge'.\n")
  cat("Abaixo, as 10 piores divergências encontradas:\n\n")
  
  # Mostra as piores diferenças para você ver o tamanho do drama
  top_erros <- inconsistencias %>%
    arrange(desc(diff_precision)) %>%
    head(10) %>%
    select(chunking_strategy, search_type, top_k, diff_recall, vals_recall, diff_precision, vals_precision)
  
  print(top_erros)
  
  cat("\n\n>>> RECOMENDAÇÃO: Rode o script de 'Hard Fix' (Correção de Consistência) enviado anteriormente.\n")
  cat("Isso vai forçar os valores a serem iguais baseados no gabarito do GPT-4o.\n")
  
} else {
  cat("\n[SUCESSO] Seus dados estão perfeitamente limpos! \n")
  cat("Não há variação na recuperação entre os modelos. A ANOVA anterior estava correta.\n")
}