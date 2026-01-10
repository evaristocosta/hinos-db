# 🎵 Hinos DB - Base de Dados de Hinos da ICM

![Python](https://img.shields.io/badge/Python-3.11.10-blue)
![Status](https://img.shields.io/badge/Status-Ativo-green)

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Support-yellow?logo=buy-me-a-coffee)](https://buymeacoffee.com/lucascosta74)

Repositório para armazenamento, processamento e análise de dados sobre hinos da Igreja Cristã Maranata (ICM). Este projeto mantém um banco de dados estruturado com informações sobre hinos, coletâneas, autores, categorias e suas relações, além de múltiplas aplicações para ETL, análise de dados, busca semântica e visualização interativa.

## 📋 Sobre o Projeto

Este projeto foi desenvolvido para organizar e analisar sistematicamente a coletânea de hinos da ICM. Ele oferece:

- **Banco de dados SQL** estruturado com informações sobre hinos, coletâneas, autores e categorias
- **Pipelines ETL** para extração de dados de slides em PowerPoint e análise de similaridade
- **Sistema de migrações** para versionamento do banco de dados
- **Análise exploratória de dados (EDA)** com técnicas de Ciência de Dados e NLP
- **Sistema RAG** para busca inteligente de hinos usando embeddings e LLM
- **Aplicações web interativas** com Streamlit para visualização e busca
- **Módulos compartilhados** para reutilização de código entre aplicações

## 🗂️ Estrutura do Repositório

```
hinos-db/
├── apps/                        # Aplicações do projeto
│   ├── analytics/               # Análise de dados
│   │   └── eda-01/              # 🌟 Análise Exploratória de Dados da Coletânea
│   │       ├── streamlit_app.py # Aplicação web interativa
│   │       ├── src/             # Código-fonte das análises
│   │       ├── notebooks/       # Notebooks de desenvolvimento
│   │       └── assets/          # Dados e banco de dados
│   ├── etl-similarity/          # Pipeline de análise de similaridade
│   │   ├── pipeline.py          # Pipeline completo
│   │   ├── extract_data.py      # Extração do banco
│   │   ├── similarities.py      # Cálculo de similaridades
│   │   ├── processes.py         # Processamento NLP
│   │   └── assets/              # Matrizes e logs
│   ├── etl-slides/              # Pipeline ETL para slides PowerPoint
│   │   ├── pipeline.py          # Pipeline completo
│   │   ├── pptx2txt.py          # Extrator de texto
│   │   ├── txt2json.py          # Conversor texto → JSON
│   │   ├── json2sql.py          # Conversor JSON → SQL
│   │   └── slides_adapt/        # Slides processados
│   ├── hymn-importer/           # Importação de novos hinos
│   │   ├── pipeline.ipynb       # Pipeline de adição
│   │   └── arquivos_hinos/      # Arquivos em Markdown
│   ├── hymn-rag/                # 🔍 Sistema de busca inteligente
│   │   ├── streamlit_app.py     # App de busca com RAG
│   │   ├── src/                 # Lógica do sistema RAG
│   │   └── assets/              # Índices e embeddings
│   └── shared/                  # Código compartilhado
│       ├── assets/              # Assets comuns
│       ├── models/              # Modelos compartilhados
│       ├── rag/                 # Utilitários RAG
│       └── similarity/          # Utilitários de similaridade
├── database/                    # Banco de dados e migrações
│   ├── migrations/              # Scripts SQL de migração
│   ├── schema/                  # Esquemas do banco
│   └── run_migrations.py        # Executor de migrações
├── docs/                        # Documentação adicional
├── requirements.txt             # Dependências Python
├── WARP.md                      # Guia para WARP terminal
└── README.md                    # Este arquivo
```

## 🗄️ Estrutura do Banco de Dados

O banco de dados possui as seguintes tabelas principais:

- **hino**: Informações principais dos hinos (título, texto, categoria, coletânea)
- **coletanea**: Coletâneas de hinos
- **categoria**: Categorias temáticas dos hinos
- **autor**: Autores e compositores
- **hino_autor**: Relação entre hinos e autores
- **autor_acao**: Tipo de contribuição do autor (letra, melodia, etc.)

## 📱 Aplicações

### 🌟 EDA-01 - Análise Exploratória de Dados

Aplicação web interativa com análises completas da coletânea usando Ciência de Dados e NLP:

- ✅ **Análise de Categorias**: Distribuição temática dos hinos
- ✅ **Análise de Títulos**: Padrões e características dos títulos
- ✅ **Análise Textual**: Palavras-chave e termos frequentes
- ✅ **Word Embeddings**: Representação vetorial de palavras
- ✅ **Sentence Embeddings**: Similaridade semântica entre hinos
- ✅ **Análise de Emoções**: Identificação de sentimentos nas letras
- ✅ **Sistema de Recomendação**: Seleção de hinos similares usando TOPSIS

```bash
cd apps/analytics/eda-01
streamlit run streamlit_app.py
```

📚 [README detalhado](apps/analytics/eda-01/README.md)

### 🔍 Hymn RAG - Busca Inteligente de Hinos

Sistema de busca semântica usando RAG (Retrieval-Augmented Generation):

- 🔎 **Busca Semântica**: Encontra hinos por significado, não apenas palavras-chave
- 🏷️ **Filtros Avançados**: Por categoria, coletânea e tema
- 🤖 **Respostas Contextualizadas**: Geradas por LLM com base nos hinos encontrados
- 📖 **Referências Bíblicas**: Relaciona hinos com passagens da Bíblia
- 🎨 **Interface Moderna**: UI responsiva e amigável

```bash
cd apps/hymn-rag
streamlit run streamlit_app.py
```

**Nota**: Requer token do Hugging Face para usar os modelos de LLM.

📚 [README detalhado](apps/hymn-rag/README.md)

### 🚀 Projeto em Desenvolvimento

Este é um **projeto em evolução contínua**! Futuros trabalhos ou ideias incluem:

- 📊 **Novas análises**: EDA-02, EDA-03 com diferentes enfoques
- 🤖 **Machine Learning**: Modelos preditivos e de classificação
- 📈 **Análise Avançada**: Evolução temporal, redes semânticas
- 🎼 **Análise Musical**: Integração com dados de melodias e harmonias

## 👨‍💻 Autor

**Lucas Piccioni Costa**

- 📧 Email: lucascosta74@gmail.com
- 📸 Instagram: [@lucas.costa74](https://www.instagram.com/lucas.costa74/)
- 💼 LinkedIn: [lucascosta74](https://www.linkedin.com/in/lucascosta74/)
- 🐙 GitHub: [evaristocosta](https://github.com/evaristocosta)

## 📄 Licença

Este projeto é de código aberto e está disponível sob a licença MIT.

---

**Nota**: Este repositório é mantido de forma independente e não possui afiliação oficial com a Igreja Cristã Maranata.
