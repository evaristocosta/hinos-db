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
│   │       └── assets/          # Dados e matrizes processadas
│   ├── etl-similarity/          # Pipeline de cálculo de similaridade semântica
│   │   ├── pipeline.py          # Execução sequencial da pipeline
│   │   ├── extract_data.py      # Extração dos dados do banco
│   │   ├── similarities.py      # Funções de cálculo de similaridade
│   │   ├── processes.py         # Processamento de NLP e embeddings
│   │   ├── config.py            # Modelos e parâmetros configuráveis
│   │   └── assets/              # Matrizes geradas e logs de execução
│   ├── etl-slides/              # Pipeline ETL para slides PowerPoint
│   │   ├── pptx2txt.py          # Extrator PPTX → TXT
│   │   ├── txt2json.py          # Conversor TXT → JSON estruturado
│   │   └── json2sql.py          # Conversor JSON → Migrações SQL
│   ├── hymn-importer/           # Importação e catalogação de novos hinos
│   │   ├── pipeline.ipynb       # Notebook de importação
│   │   └── arquivos_hinos/      # Letras em Markdown
│   ├── hymn-rag/                # 🔍 Sistema de busca inteligente e RAG
│   │   ├── streamlit_app.py     # Interface web em Streamlit
│   │   ├── src/                 # Utilitários RAG, Bíblia e categorias
│   │   ├── scripts/             # CLI query.py e generate_assets.py
│   │   └── assets/              # Vectorstore (ChromaDB) e chunks em cache
│   └── shared/                  # Módulos e modelos compartilhados
│       ├── assets/              # Stopwords e referências comuns
│       ├── models/              # Modelos (FastText cc.pt.300.bin)
│       ├── rag/                 # Artefatos RAG compartilhados
│       └── similarity/          # Matrizes de similaridade consolidadas
├── database/                    # Banco de dados central e ferramentas
│   ├── database.db              # Banco de dados SQLite
│   ├── migrations/              # Migrações SQL sequenciais (001 a 012+)
│   ├── schema/                  # Esquemas SQL, DBML e diagramas PlantUML
│   └── tools/                   # Scripts para migrações e geração de diagramas
├── docs/                        # Documentação complementar
├── requirements.txt             # Dependências Python globais
├── AGENTS.md                    # Guia para agentes de IA e desenvolvedores
└── README.md                    # Este arquivo
```

## 🗄️ Estrutura do Banco de Dados

O banco de dados SQLite (`database/database.db`) possui as seguintes tabelas principais:

- **hino**: Informações centrais do hino (título, texto bruto, texto processado com marcações, categoria, tom)
- **coletanea**: Coletâneas de hinos (ex: Coletânea de Igrejas, CIAS, Avulsos)
- **hino_coletanea**: Associação entre hino e coletânea, armazenando a numeração (`hino_numero`) em cada contexto
- **categoria**: Classificação temática dos hinos
- **autor**: Letristas, compositores e tradutores
- **hino_autor**: Relação N:N entre hinos e autores
- **autor_acao**: Tipo de contribuição do autor (letra, música, arranjo, tradução)

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

Para consultas via linha de comando ou regeneração de assets:

```bash
# Busca via CLI com streaming (usando Ollama localmente)
python scripts/query.py "Hinos sobre consolo e encorajamento"

# Regeneração do índice vetorial ChromaDB e cache de chunks
python scripts/generate_assets.py --force
```

📚 [README detalhado](apps/hymn-rag/README.md)

### ⚙️ ETL & Processamento de Dados

#### 1. Pipeline de Similaridade (`apps/etl-similarity`)
Calcula matrizes de similaridade léxica (TF-IDF), semântica (FastText & BERTimbau) e de sentimentos/emoções:

```bash
cd apps/etl-similarity
python pipeline.py
```

#### 2. Migrações e Gerenciamento do Banco (`database/tools`)
Aplica migrações sequenciais e reconstrói o banco SQLite e esquemas:

```bash
cd database/tools
# Recria database.db executando todas as migrações (requer sqlite3 no PATH)
python run_migrations.py

# Atualiza arquivos de esquema (SQL e PlantUML)
python generate_schema_sql.py
python generate_schema_puml.py
```

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
