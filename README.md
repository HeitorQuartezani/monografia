# Assistente Virtual Fabi: Otimização de Sistemas RAG para o MPES

Este repositório contém a implementação técnica e o arcabouço analítico do projeto de pesquisa **"Análise de Hiperparâmetros em Sistemas de Geração Aumentada por Recuperação Aplicado ao Acervo do Ministério Público do Estado do Espírito Santo"**. Concluído em fevereiro de 2026, o trabalho propõe uma metodologia fundamentada em Planejamento de Experimentos (DoE) para analisar e otimizar parâmetros críticos de um sistema RAG.

O projeto originou-se para superar a ineficácia das buscas lexicais estritas no acervo de atos normativos e portarias do MPES, democratizando o acesso a esse capital informacional através da assistente virtual Fabi.

## 🧠 Arquitetura do Sistema

A macroarquitetura da assistente Fabi é composta por quatro componentes orquestrados em um fluxo de processamento contínuo:
* **Interface (Gradio):** Interação em linguagem natural com o usuário final.
* **Orquestrador (FastAPI):** Motor de coordenação que recebe a pergunta e as configurações de hiperparâmetros, despachando-as para o sistema RAG.
* **Recuperação e Banco de Dados (ChromaDB):** Realiza a busca híbrida no acervo vetorizado para extrair o contexto relevante.
* **Gerador (ChatGPT / OpenAI):** Sintetiza a resposta final ancorada exclusivamente nos documentos recuperados, mitigando o risco de alucinação.

## 🛠️ Stack Tecnológico
* **Linguagem & Orquestração:** Python 3.11, LangChain.
* **Backend & API:** FastAPI.
* **Frontend:** Gradio.
* **Banco de Vetores:** ChromaDB.
* **Modelos de Embedding e LLM:** text-embedding-3-small, GPT-4o, GPT-4o-mini.
* **Avaliação Automatizada:** Framework Ragas (paradigma LLM-as-a-Judge).
* **Análise Estatística:** R versão 4.3 (pacote ARTool).

## 📊 Metodologia e Otimização Experimental

O sistema não foi ajustado via empirismo, mas sim submetido a um Delineamento Fatorial Completo cruzando quatro fatores independentes (Estratégia de Chunking, Algoritmo de Busca, Top-K e Modelo LLM). O experimento gerou 3.840 observações independentes a partir de um conjunto de teste sintético. 

Devido à natureza não-paramétrica dos dados das métricas de RAG, a análise de significância estatística e as interações foram conduzidas utilizando o método de Transformação de Postos Alinhados (Aligned Rank Transform - ART).

### Configuração Ideal Recomendada
A calibração que demonstrou maximizar a qualidade técnica e a eficiência de custos no domínio jurídico do MPES consiste em:
* **Ingestão:** Segmentação Semântica (Percentil 95) ou Recursiva com janelas amplas (1000 caracteres).
* **Busca:** Fusão Híbrida (Vetorial + BM25) combinada via Reciprocal Rank Fusion (RRF).
* **Contexto:** Janela de 20 documentos recuperados (Top-K=20).
* **Modelo Gerador:** GPT-4o-mini.

## 👨‍💻 Autor
**Heitor C. Quartezani**
Estatístico e Cientista de Dados | Mestrando em Ciência da Computação (PPGI-UFES)
Focado no desenvolvimento de arquiteturas de IA escaláveis e na aplicação de Inteligência Artificial para otimização de sistemas públicos, como o desenvolvimento de chatbots estruturados com RAG.
*Orientador:* Prof. Dr. Diego Roberto Colombo Dias.
