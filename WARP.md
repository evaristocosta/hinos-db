# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**Hinos DB** is a database and analytics system for hymns from Igreja Cristã Maranata (ICM). The project combines ETL pipelines, NLP analysis, and interactive visualization to explore hymn data systematically.

**Language**: All code, comments, and documentation are primarily in Portuguese (Brazilian).

## Common Commands

### Database Management
```bash
# Run all database migrations (recreates database.db from scratch)
python database/run_migrations.py
```

**Note**: The migration script deletes the existing `database/database.db` file before applying migrations. Migrations are SQL files in `database/migrations/` executed in alphabetical order.

### ETL Pipelines

**ETL Slides** - Extract hymns from PowerPoint slides:
```bash
# Run full pipeline (PPTX → TXT → JSON → SQL)
python apps/etl-slides/pipeline.py
```

**ETL Similarity** - Generate similarity matrices and NLP features:
```bash
# Run full similarity pipeline with default settings
python apps/etl-similarity/pipeline.py

# Specify assets folder and collection ID
python apps/etl-similarity/pipeline.py --assets-folder path/to/assets --coletanea-id 1

# Enable debug logging
python apps/etl-similarity/pipeline.py --log-level DEBUG
```

This pipeline performs multiple sequential steps:
1. Extract hymns from database
2. Calculate title similarities (fuzzy matching)
3. Process tokens and n-grams
4. Calculate word TF-IDF similarities
5. Generate word embeddings (requires FastText model: `cc.pt.300.bin`)
6. Generate sentence embeddings (uses transformer model)
7. Analyze emotions (uses Portuguese BERT model)
8. Save final DataFrame with all features

**Output**: Similarity matrices and processed data saved to `apps/etl-similarity/assets/` (gitignored except stopwords)

### Analytics Application

**Streamlit EDA Dashboard**:
```bash
# Run the interactive analytics dashboard
cd apps/analytics/eda-01
streamlit run streamlit_app.py
```

The app includes exploratory data analysis with:
- Category distribution
- Title analysis
- Word frequency analysis
- Word and sentence embeddings visualization
- Emotion analysis
- TOPSIS-based hymn recommendation system

### Development Workflow

**Jupyter Notebooks**: The `apps/analytics/eda-01/notebooks/` directory contains development notebooks (`eda1_part1.ipynb` through `eda1_part7.ipynb`) used to prototype analyses before integration into the Streamlit app.

**Python Code Formatting**: The project uses `black` for code formatting (installed in requirements.txt).

## Architecture

### Database Schema

The SQLite database (`database/database.db`) uses the following core schema:

**Main Tables**:
- `hino`: Hymns with fields: id, coletanea_id, categoria_id, numero, nome, texto, texto_limpo, texto_estruturado, ano_composicao, tom, timestamps
- `coletanea`: Collections of hymns (nome, descricao, arquivo)
- `categoria`: Thematic categories (descricao)
- `autor`: Authors/composers (nome)
- `autor_acao`: Types of authorship contributions (acao) - e.g., letra, melodia
- `hino_autor`: Many-to-many relationship between hymns and authors with action type
- `hino_de`: German hymn translations (separate table structure)

**Key Relationships**:
- `hino.coletanea_id` → `coletanea.id`
- `hino.categoria_id` → `categoria.id`
- `hino_autor` joins `hino`, `autor`, and `autor_acao`

**Important**: `texto_limpo` contains cleaned text without control tags (CORO, BIS, etc.) used for NLP analysis.

### Directory Structure

```
hinos-db/
├── apps/                          # Application modules
│   ├── analytics/
│   │   └── eda-01/               # EDA Streamlit app
│   │       ├── src/              # Modular analysis pages
│   │       ├── notebooks/        # Development notebooks
│   │       ├── assets/           # Database copy + processed data
│   │       └── streamlit_app.py  # Main entry point
│   ├── etl-similarity/           # NLP similarity pipeline
│   │   ├── pipeline.py           # Main orchestration
│   │   ├── extract_data.py       # Database extraction
│   │   ├── processes.py          # NLP processing steps
│   │   ├── similarities.py       # Similarity calculations
│   │   ├── config.py             # Model configurations
│   │   └── utils.py              # Logging and tracking
│   ├── etl-slides/               # PowerPoint extraction pipeline
│   │   ├── pipeline.py           # Orchestrates pptx→txt→json→sql
│   │   ├── pptx2txt.py           # Extract text from PPTX
│   │   ├── txt2json.py           # Parse text to JSON
│   │   └── json2sql.py           # Insert JSON into database
│   └── hymn-importer/            # Manual hymn addition tools
│       ├── pipeline.ipynb        # Jupyter notebook for adding hymns
│       └── arquivos_hinos/       # Markdown files with hymn text
├── database/
│   ├── migrations/               # SQL migration files (numbered)
│   ├── run_migrations.py         # Migration runner
│   └── database.db               # SQLite database (gitignored)
├── docs/                         # Documentation
└── requirements.txt              # Python dependencies
```

### Pipeline Architecture

**ETL Similarity Pipeline** (`apps/etl-similarity/`):

This is the most complex component. It's designed as a sequential pipeline with detailed logging and progress tracking:

1. **Sequential Processing**: Each step depends on the previous step's output
2. **Artifact Saving**: Intermediate results saved as pickle files
3. **Error Handling**: Each step tracked with success/failure status
4. **Logging**: Detailed logs saved to `apps/etl-similarity/logs/`

**Key Design Pattern**: DataFrame transformation chain where each processing function:
- Takes a DataFrame as input
- Adds new columns with computed features
- Returns the enriched DataFrame
- Optionally saves similarity matrices to disk

**Model Dependencies**:
- FastText Portuguese model: `cc.pt.300.bin` (300-dim word vectors)
- Sentence Transformer: `rufimelo/Legal-BERTimbau-sts-base-ma-v2`
- Emotion Classifier: `pysentimiento/bert-pt-emotion`

**Analytics Architecture** (`apps/analytics/eda-01/`):

Uses Streamlit's multi-page app pattern:
- `streamlit_app.py`: Navigation setup
- `src/main.py`: Home page
- `src/loader.py`: Centralized data loading with `@st.cache_data`
- `src/*.py`: Individual analysis pages

**Caching Strategy**: All database queries and pickle loads use `@st.cache_data` decorator to avoid reloading on every interaction.

### Data Flow

1. **Input**: PowerPoint slides or Markdown files with hymn text
2. **ETL Slides**: Extract → Transform → Load into database
3. **Database**: Central SQLite repository
4. **ETL Similarity**: Extract from DB → NLP processing → Save features + similarity matrices
5. **Analytics**: Load processed data → Interactive visualization

**Important Note**: The `apps/analytics/eda-01/assets/database.db` is a copy of `database/database.db` (only the EDA database copy is version controlled per .gitignore rules).

## Code Conventions

### Database Access Pattern

Always use SQLAlchemy for database access:

```python
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine("sqlite:///../../database/database.db")
connection = engine.connect()
df = pd.read_sql_query(sql_query, connection)
```

### Text Cleaning Pattern

Hymn text uses specific formatting tags:
- `**Section**`: Bold sections (CORO, FINAL, BIS)
- Control tags: TODOS, M, H, T, BIS, VARÕES, SERVAS (literal indicators)
- Meta tags: CORO (2X), ÍNDICE, REPETIR O LOUVOR (removed in texto_limpo)

When processing text, always use `texto_limpo` column for NLP analysis as it has control tags removed.

### Similarity Matrix Storage

All similarity matrices stored as pickled pandas DataFrames with:
- Index and columns: hymn numbers
- Values: similarity scores (0-1 or 0-100 depending on metric)

Files follow naming pattern: `similarity_matrix_<type>.pkl`

### Portuguese NLP Considerations

- Stopwords: Use `apps/etl-similarity/assets/stopwords-br.txt` or `apps/analytics/eda-01/assets/stopwords-br.txt`
- Text preprocessing: Remove accents, lowercase, remove stopwords
- Models: Always use Portuguese-specific models (see config.py)

## Important Context

### Collection IDs

- `coletanea_id = 1`: Main hymnal collection (primary focus)
- Other collections: CIAs (Children, Intermediates, Adolescents), occasional hymns

Most analyses filter to `coletanea_id = 1` to focus on the main hymnal.

### Migration System

Migrations are **destructive**: `run_migrations.py` deletes `database.db` before reapplying all migrations. Each migration is idempotent SQL but the runner provides no rollback mechanism. Migrations are numbered sequentially (001, 002, etc.) and must maintain order.

### File Encoding

All text files use UTF-8 encoding. When reading/writing files, explicitly specify `encoding="utf-8"`.

### Asset Management

Large binary files are gitignored:
- FastText models (`cc.*.bin`)
- Similarity matrices and processed data in `apps/etl-similarity/assets/`
- PowerPoint files (`*.pptx`)

Only essential assets committed:
- Stopwords file
- EDA database copy
- Example hymn markdown files

### Path References

Code uses `Path(__file__).parent` for relative path resolution. This is critical because:
- Streamlit runs from different working directories
- Pipeline scripts may be called from repository root or app directory
- Database paths must resolve correctly regardless of execution context

## Windows-Specific Notes

This repository is developed on Windows. When working with:
- **Paths**: Use `Path` from `pathlib` for cross-platform compatibility
- **Shell Commands**: PowerShell is the primary shell
- **Line Endings**: Files use CRLF (`\r\n`) line endings
