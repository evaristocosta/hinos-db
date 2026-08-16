#!/usr/bin/env python
"""
Script para gerar assets do sistema RAG (vectorstore e chunks cache)
Uso: python generate_assets.py [--verbose] [--force]
"""

import argparse
import sqlite3
import pickle
import shutil
from pathlib import Path
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tqdm import tqdm

# ===== CONFIGURAÇÕES =====
# Modelo de embeddings (sentence-transformers via HuggingFace - compatível com cloud)
HF_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def locate_database() -> Path:
    """Localiza o arquivo database.db"""
    candidates = [
        Path(__file__).parent.parent / "assets" / "database.db",
        Path.cwd() / "database" / "database.db",
        Path.cwd().parent / "database" / "database.db",
        Path.cwd().parent.parent / "database" / "database.db",
    ]
    db_path = next((p for p in candidates if p.exists()), None)
    if not db_path:
        raise FileNotFoundError(
            "database.db não encontrado. Verifique os caminhos candidatos."
        )
    return db_path


def load_hymns_from_db(db_path: Path, verbose: bool = False) -> List[Document]:
    """Carrega hinos do banco de dados"""
    if verbose:
        print(f"📂 Carregando hinos de: {db_path}")

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                h.id, 
                h.nome, 
                hc.hino_numero as numero, 
                h.texto_processado AS texto,
                h.categoria_id, 
                hc.coletanea_id
            FROM 
                hino h
                INNER JOIN hino_coletanea hc ON hc.hino_id = h.id
            WHERE 
                h.texto_processado IS NOT NULL
            ORDER BY h.id
            """)
        rows = cur.fetchall()

    docs = []
    iterator = tqdm(rows, desc="Carregando documentos") if verbose else rows
    for hid, nome, numero, texto, categoria_id, coletanea_id in iterator:
        if not texto:
            continue
        content = f"{nome or ''} ({numero or ''})\n\n{texto.strip()}"
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "hino_id": hid,
                    "nome": nome,
                    "numero": numero,
                    "categoria_id": categoria_id,
                    "coletanea_id": coletanea_id,
                },
            )
        )

    if verbose:
        print(f"✓ {len(docs)} documentos carregados")

    return docs


def create_chunks(
    docs: List[Document], cache_path: Path, verbose: bool = False
) -> List[Document]:
    """Cria chunks dos documentos e salva em cache"""
    if verbose:
        print("⚙️ Criando chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    iterator = tqdm(docs, desc="Criando chunks") if verbose else docs
    for doc in iterator:
        chunks.extend(splitter.split_documents([doc]))

    # Salva cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(chunks, f)

    if verbose:
        print(f"✓ {len(chunks)} chunks criados e salvos em: {cache_path}")

    return chunks


def create_vectorstore(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    vector_dir: Path,
    verbose: bool = False,
):
    """Cria vectorstore a partir dos chunks"""
    if verbose:
        print("⚙️ Criando vectorstore...")

    vector_dir.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma(
        embedding_function=embeddings, persist_directory=str(vector_dir)
    )

    batch_size = 64
    iterator = range(0, len(chunks), batch_size)
    if verbose:
        iterator = tqdm(iterator, desc="Indexando embeddings")

    for i in iterator:
        batch = chunks[i : i + batch_size]
        vectorstore.add_documents(batch)

    if verbose:
        print(f"✓ Vectorstore criado em: {vector_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Gera assets (vectorstore e chunks cache) para o sistema RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Modo verboso (mostra detalhes do processamento)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Força regeneração mesmo se os assets já existirem",
    )

    args = parser.parse_args()

    # Define caminhos
    base_path = Path(__file__).parent.parent / "assets"
    chunks_cache = base_path / "chunks_cache.pkl"
    vector_dir = base_path / "vectorstore"

    # Verifica se já existem
    if not args.force:
        if chunks_cache.exists() and (vector_dir / "chroma.sqlite3").exists():
            print("✅ Assets já existem!")
            print(f"   Chunks cache: {chunks_cache}")
            print(f"   Vectorstore: {vector_dir}")
            print("\nUse --force para regenerar")
            return
    else:
        # Remove assets antigos quando --force é usado
        if chunks_cache.exists():
            chunks_cache.unlink()
            if args.verbose:
                print(f"🗑️  Removido chunks cache antigo: {chunks_cache}")
        if vector_dir.exists():
            shutil.rmtree(vector_dir)
            if args.verbose:
                print(f"🗑️  Removido vectorstore antigo: {vector_dir}")

    print("🚀 Iniciando geração de assets...\n")

    # Localiza database
    db_path = locate_database()
    if args.verbose:
        print(f"📂 Database: {db_path}\n")

    # Inicializa embeddings
    if args.verbose:
        print(f"🤖 Inicializando embeddings: {HF_EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=HF_EMBED_MODEL,
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )

    # Carrega documentos
    docs = load_hymns_from_db(db_path, verbose=args.verbose)

    # Cria chunks
    chunks = create_chunks(docs, chunks_cache, verbose=args.verbose)

    # Cria vectorstore
    create_vectorstore(chunks, embeddings, vector_dir, verbose=args.verbose)

    print("\n✅ Assets gerados com sucesso!")
    print(f"   📦 {len(chunks)} chunks salvos")
    print(f"   🔍 Vectorstore indexado")
    print(f"\n📂 Localização dos assets:")
    print(f"   Chunks: {chunks_cache}")
    print(f"   Vectorstore: {vector_dir}")


if __name__ == "__main__":
    main()
