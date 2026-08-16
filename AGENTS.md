# AGENTS.md

## Repository Overview
Python monorepo for storing, ETL processing, analyzing, and querying hymn data (ICM) via SQLite, Streamlit, NLP pipelines, and RAG.

## Project Structure & Entrypoints
- `database/`: Central SQLite database (`database.db`), sequential migration scripts (`migrations/*.sql`), and management tools (`tools/`).
- `apps/analytics/eda-01/`: Streamlit exploratory data analysis app.
  - Run: `streamlit run streamlit_app.py` (from `apps/analytics/eda-01`).
- `apps/hymn-rag/`: Streamlit RAG search UI and CLI query engine.
  - Web UI: `streamlit run streamlit_app.py` (from `apps/hymn-rag`).
  - CLI query: `python scripts/query.py "<query>"` (from `apps/hymn-rag`).
  - Asset generation: `python scripts/generate_assets.py` (from `apps/hymn-rag`).
- `apps/etl-similarity/`: NLP similarity matrices and feature extraction pipeline.
  - Run: `python pipeline.py` (from `apps/etl-similarity`).
- `apps/etl-slides/`: Slide extraction pipeline (`pptx2txt.py` -> `txt2json.py` -> `json2sql.py`).
- `apps/shared/`: Shared models (FastText `cc.pt.300.bin`), stopwords (`stopwords-br.txt`), and common similarity artifacts.

## Database & Schema Quirks
- **Schema Mapping (Migration 012+)**: `hino` does not contain `numero` or `coletanea_id`. Hymn numbering and collection membership are stored in `hino_coletanea`. Always join `hino_coletanea` (`hino_coletanea.hino_id = hino.id`) when filtering or displaying hymn numbers/collections.
- **Running Migrations**:
  - `database/tools/run_migrations.py` drops and rebuilds `database/database.db` from scratch by piping migrations to the `sqlite3` CLI (requires `sqlite3` in PATH).
  - Working directory must be `database/tools/` when running `run_migrations.py`, `generate_schema_sql.py`, or `generate_schema_puml.py`.
- **Schema Artifacts**: Update schema definition and diagram after modifying migrations:
  ```powershell
  # inside database\tools
  python generate_schema_sql.py; python generate_schema_puml.py
  ```

## App & Runtime Gotchas
- **Execution Working Directory**: Scripts and Streamlit apps rely on relative paths to their local `assets/` directories or relative parent directories. Always run commands from the specific application subdirectory rather than repo root.
- **RAG Vectorstore & Chunks**: `apps/hymn-rag` depends on `assets/chunks_cache.pkl` and `assets/vectorstore/chroma.sqlite3`. Regenerate them using `python scripts/generate_assets.py` inside `apps/hymn-rag` whenever hymns or database contents are modified.
- **Environment Variables**: `apps/hymn-rag` requires `.env` (or Streamlit secrets) containing `HUGGINGFACE_API_TOKEN` and optional Bible API keys (`ABIBLIADIGITAL_API_TOKEN`). CLI `query.py` requires Ollama running locally.
- **Hymn Text Markup**: `hino.texto_processado` uses custom XML-like structural tags (`<coro>`, `<bis>`, `<h>`, `<m>`, `<t>`, `<repetir>`, `<instrumentos>`, `<final>`). Keep or handle these tags when doing text transformations or tokenization.
