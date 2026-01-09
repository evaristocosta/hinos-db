#!/usr/bin/env python
"""
Script de busca RAG para a Coletânea de Hinos

NOTA: Este script requer que os assets (vectorstore e chunks cache) tenham sido
      gerados previamente. Execute 'python generate_assets.py' primeiro.

Uso: python query.py "sua consulta aqui" [--verbose]
"""
import argparse
import sqlite3
import pickle
import re

import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from fetch_bible import extract_bible_refs, fetch_bible_verses
from extract_categories import extract_filters_deterministic

# ===== CONFIGURAÇÕES =====
# Modelo de embeddings (sentence-transformers via HuggingFace - compatível com cloud)
HF_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_LLM_MODELS = [
    "gemma3:1b",
    "llama3.2:1b",
    "gemma3:4b",
]
LLM_TEMPERATURE = 0.1
MAX_RESULTS = 25
VECTOR_SEARCH_K = 10
VECTOR_FETCH_K = 25
BM25_K = 10


def _resolve_categorias(inputs: List[str], categorias_dict: dict) -> List[str]:
    """
    Resolve categorias por índice (1-based) ou nome.
    Retorna lista de nomes de categorias válidos (lowercase).
    """
    categorias_list = list(categorias_dict.keys())
    resolved = []

    for inp in inputs:
        inp = inp.strip()
        # Tenta como índice
        if inp.isdigit():
            idx = int(inp) - 1  # Converte para 0-based
            if 0 <= idx < len(categorias_list):
                resolved.append(categorias_list[idx])
            else:
                print(f"⚠️ Índice de categoria inválido: {inp}")
        else:
            # Tenta como nome (normalizado)
            inp_lower = inp.lower()
            if inp_lower in categorias_dict:
                resolved.append(inp_lower)
            else:
                print(f"⚠️ Categoria não encontrada: {inp}")

    return resolved


def _resolve_coletaneas(inputs: List[str], coletaneas_dict: dict) -> List[str]:
    """
    Resolve coletâneas por índice (1-based) ou nome.
    Retorna lista de nomes de coletâneas válidos (lowercase).
    """
    coletaneas_list = list(coletaneas_dict.keys())
    resolved = []

    for inp in inputs:
        inp = inp.strip()
        # Tenta como índice
        if inp.isdigit():
            idx = int(inp) - 1  # Converte para 0-based
            if 0 <= idx < len(coletaneas_list):
                resolved.append(coletaneas_list[idx])
            else:
                print(f"⚠️ Índice de coletânea inválido: {inp}")
        else:
            # Tenta como nome (normalizado)
            inp_lower = inp.lower()
            if inp_lower in coletaneas_dict:
                resolved.append(inp_lower)
            else:
                print(f"⚠️ Coletânea não encontrada: {inp}")

    return resolved


# ===== CLASSE PRINCIPAL =====
class HymnRAG:
    def __init__(self, verbose: bool = False, model: str = OLLAMA_LLM_MODELS[0]):
        self.verbose = verbose
        self.db_path = self._locate_database()
        self.vector_dir = Path(__file__).parent.parent / "assets" / "vectorstore"
        self.chunks_cache = Path(__file__).parent.parent / "assets" / "chunks_cache.pkl"
        self.stopwords_path = (
            Path(__file__).parent.parent / "assets" / "stopwords-br.txt"
        )

        # Carrega configurações do banco
        self._load_metadata()

        # Inicializa componentes
        self.embeddings = HuggingFaceEmbeddings(
            model_name=HF_EMBED_MODEL,
            model_kwargs={"device": "cpu", "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
        )
        self.llm = OllamaLLM(model=model, temperature=LLM_TEMPERATURE)

        # Carrega chunks e vectorstore (devem ter sido gerados previamente)
        self.chunks = self._load_chunks()
        self.vectorstore = self._load_vectorstore()

        # Configura retrievers
        self._setup_retrievers()

        # Configura chains
        self._setup_chains()

        if self.verbose:
            print("✅ Sistema inicializado com sucesso!")
            print(f"🤖 Modelo: {model}")

    def _locate_database(self) -> Path:
        candidates = [
            Path(__file__).parent.parent / "assets" / "database.db",
            Path.cwd() / "database" / "database.db",
            Path.cwd().parent / "database" / "database.db",
            Path.cwd().parent.parent / "database" / "database.db",
        ]
        db_path = next((p for p in candidates if p.exists()), None)
        if not db_path:
            raise FileNotFoundError("database.db não encontrado")
        if self.verbose:
            print(f"📂 Database: {db_path}")
        return db_path

    def _load_metadata(self):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()

            # Total de hinos
            cur.execute("SELECT count(*) FROM hino")
            self.total_hinos = cur.fetchone()[0]

            # Categorias
            cur.execute("SELECT id, descricao FROM categoria")
            self.categorias = {row[1].lower(): row[0] for row in cur.fetchall()}

            # Coletâneas
            cur.execute("SELECT id, nome FROM coletanea")
            self.coletaneas = {row[1].lower(): row[0] for row in cur.fetchall()}

        if self.verbose:
            print(f"📊 Total de hinos: {self.total_hinos}")

    def _load_chunks(self) -> List[Document]:
        """Carrega chunks do cache. Requer que os assets tenham sido gerados previamente."""
        if not self.chunks_cache.exists():
            raise FileNotFoundError(
                f"Cache de chunks não encontrado: {self.chunks_cache}\n"
                "Execute 'python generate_assets.py' primeiro para gerar os assets."
            )

        if self.verbose:
            print(f"💾 Carregando chunks do cache...")

        with open(self.chunks_cache, "rb") as f:
            chunks = pickle.load(f)

        if self.verbose:
            print(f"✓ {len(chunks)} chunks carregados")

        return chunks

    def _load_vectorstore(self) -> Chroma:
        """Carrega vectorstore. Requer que os assets tenham sido gerados previamente."""
        if (
            not self.vector_dir.exists()
            or not (self.vector_dir / "chroma.sqlite3").exists()
        ):
            raise FileNotFoundError(
                f"Vectorstore não encontrado: {self.vector_dir}\n"
                "Execute 'python generate_assets.py' primeiro para gerar os assets."
            )

        if self.verbose:
            print("💾 Carregando vectorstore...")

        return Chroma(
            embedding_function=self.embeddings,
            persist_directory=str(self.vector_dir),
        )

    def _setup_retrievers(self):
        # Vector retriever
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": VECTOR_SEARCH_K, "fetch_k": VECTOR_FETCH_K},
        )

        # BM25 retriever
        if self.chunks:
            # Carrega stopwords
            stopwords = set()
            if self.stopwords_path.exists():
                with open(self.stopwords_path, encoding="utf-8") as f:
                    stopwords = {
                        line.strip().strip('"')
                        for line in f
                        if line.strip() and not line.startswith("#")
                    }

            word_re = re.compile(r"\w+")

            def bm25_tokenizer(text: str):
                tokens = word_re.findall(text.lower())
                return [t for t in tokens if t not in stopwords]

            self.bm25_retriever = BM25Retriever.from_documents(
                self.chunks, preprocess_func=bm25_tokenizer
            )
            self.bm25_retriever.k = BM25_K
        else:
            self.bm25_retriever = None

    def _setup_chains(self):
        # Prompt de resposta
        answer_system = """
Você é um assistente que responde apenas com base nas opções de hinos fornecidas no contexto.
É preferível retornar mais de uma opção, pelo menos três, quando disponível, a não ser que requisitado diferente na pergunta.
Explique de maneira sucinta os motivos de selecionar tais hinos.
Cite os números dos hinos (se houver) e principalmente os títulos.
Se não souber, diga que não encontrou na base.
Responda SOMENTE em PORTUGUÊS DO BRASIL.
"""
        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", answer_system),
                (
                    "user",
                    "Pergunta original: {question}\n\nContexto:\n{context}\n\nResposta:",
                ),
            ]
        )

    def _format_docs(self, docs: List[Document]) -> str:
        parts = []
        for d in docs:
            parts.append(
                f"[{d.metadata.get('numero') or 'N/A'}] {d.metadata.get('nome')}\n{d.page_content}"
            )
        return "\n\n".join(parts)

    def _hybrid_retrieve_filtered(
        self, search_query: str, filters: dict
    ) -> List[Document]:
        # Coleta IDs de categorias/coletâneas
        categoria_ids = []
        if filters.get("categorias"):
            for cat_name in filters["categorias"]:
                cat_id = self.categorias.get(cat_name.lower())
                if cat_id:
                    categoria_ids.append(cat_id)

        coletanea_ids = []
        if filters.get("coletaneas"):
            for col_name in filters["coletaneas"]:
                col_id = self.coletaneas.get(col_name.lower())
                if col_id:
                    coletanea_ids.append(col_id)

        if self.verbose:
            print(f"🔍 Filtros: categorias={categoria_ids}, coletaneas={coletanea_ids}")

        # Helper para verificar filtros (intersecção)
        def matches_filters(doc) -> bool:
            if categoria_ids:
                if doc.metadata.get("categoria_id") not in categoria_ids:
                    return False
            if coletanea_ids:
                if doc.metadata.get("coletanea_id") not in coletanea_ids:
                    return False
            return True

        # Busca vetorial
        vec_docs = []
        if categoria_ids or coletanea_ids:
            seen_ids = set()

            if categoria_ids and coletanea_ids:
                for cat_id in categoria_ids:
                    for col_id in coletanea_ids:
                        try:
                            docs = self.vectorstore.similarity_search(
                                search_query,
                                k=15,
                                filter={"categoria_id": cat_id, "coletanea_id": col_id},
                            )
                            for doc in docs:
                                hid = doc.metadata.get("hino_id")
                                if hid not in seen_ids:
                                    seen_ids.add(hid)
                                    vec_docs.append(doc)
                        except:
                            pass
            elif categoria_ids:
                for cat_id in categoria_ids:
                    try:
                        docs = self.vectorstore.similarity_search(
                            search_query, k=15, filter={"categoria_id": cat_id}
                        )
                        for doc in docs:
                            hid = doc.metadata.get("hino_id")
                            if hid not in seen_ids:
                                seen_ids.add(hid)
                                vec_docs.append(doc)
                    except:
                        pass
            else:
                for col_id in coletanea_ids:
                    try:
                        docs = self.vectorstore.similarity_search(
                            search_query, k=15, filter={"coletanea_id": col_id}
                        )
                        for doc in docs:
                            hid = doc.metadata.get("hino_id")
                            if hid not in seen_ids:
                                seen_ids.add(hid)
                                vec_docs.append(doc)
                    except:
                        pass

            vec_docs = vec_docs[:10]

        else:
            vec_docs = self.vector_retriever.invoke(search_query)

        # BM25
        if self.bm25_retriever:
            bm25_docs = self.bm25_retriever.invoke(search_query)
            if categoria_ids or coletanea_ids:
                bm25_docs = [d for d in bm25_docs if matches_filters(d)]
        else:
            bm25_docs = []

        # Combina
        seen = set()
        combined = []
        for doc in vec_docs:
            hid = doc.metadata.get("hino_id")
            if hid not in seen:
                seen.add(hid)
                combined.append(doc)

        for doc in bm25_docs:
            hid = doc.metadata.get("hino_id")
            if hid not in seen and len(combined) < MAX_RESULTS:
                seen.add(hid)
                combined.append(doc)

        return combined[:MAX_RESULTS]

    def query_stream(
        self,
        question: str,
        auto_filters: bool = False,
        manual_categorias: List[str] = None,
        manual_coletaneas: List[str] = None,
    ):
        """Consulta com streaming de resposta - retorna (docs, bible_context, generator)"""
        # Extrai referências bíblicas
        bible_refs = extract_bible_refs(question)
        bible_context = fetch_bible_verses(bible_refs) if bible_refs else ""

        if self.verbose and bible_refs:
            print(f"📖 Referências bíblicas: {bible_refs}")
            if bible_context:
                print(f"📖 Texto bíblico extraído:\n{bible_context}")

        # Determina filtros
        filters = {}

        if manual_categorias or manual_coletaneas:
            filters = {
                "categorias": manual_categorias,
                "coletaneas": manual_coletaneas,
                "search_query": question,
                "matches_info": {"manual": True},
            }
        elif auto_filters:
            filters = extract_filters_deterministic(
                question, self.categorias, self.coletaneas
            )
        else:
            filters = {
                "categorias": None,
                "coletaneas": None,
                "search_query": question,
                "matches_info": {},
            }

        search_query = filters.get("search_query", question)

        # Enriquece com texto bíblico
        effective_query = search_query
        if bible_context:
            effective_query = search_query + "\n\n" + '"' + bible_context[:700] + '"'
            if self.verbose:
                print("🔎 Consulta enriquecida com texto bíblico")

        if self.verbose:
            print(f"📝 Query para busca: {effective_query}")

        # Busca hinos
        docs = self._hybrid_retrieve_filtered(effective_query, filters)

        if self.verbose:
            print(f"📚 {len(docs)} hinos encontrados")
            for doc in docs:
                print(f"  - [{doc.metadata.get('numero')}] {doc.metadata.get('nome')}")

        if not docs:

            def empty_generator():
                yield "❌ Nenhum hino encontrado com esses critérios."

            return empty_generator()

        # Formata contexto
        context = self._format_docs(docs)
        if bible_context:
            context = context + "\n\nTrechos bíblicos:\n" + '"' + bible_context + '"'

        filter_info = ""
        if filters.get("categorias") or filters.get("coletaneas"):
            filter_info = f"\nFiltros aplicados: Categorias={filters.get('categorias')}, Coletâneas={filters.get('coletaneas')}"

        # Gera resposta com streaming
        final_prompt = self.answer_prompt.format(
            question=question + filter_info, context=context
        )

        if self.verbose:
            print("💬 Gerando resposta...")

        # Retorna docs, bible_context e o generator
        def stream_generator():
            for chunk in self.llm.stream(final_prompt):
                yield chunk

        return stream_generator()


# ===== MAIN =====
def main():
    parser = argparse.ArgumentParser(
        description="Sistema RAG de busca na Coletânea de Hinos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python query.py "Hinos sobre unidade"
  python query.py "Hinos sobre graça" -f --verbose
  python query.py "Hinos que combinam com Isaías 4:6" -v
  python query.py "Hinos sobre salvação" --categorias 4 5
  python query.py "Louvores" --coletaneas 1 4
  python query.py "Hinos de consolo" --categorias "consolo e encorajamento"
        """,
    )
    parser.add_argument("query", type=str, help="Consulta/pergunta sobre hinos")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Modo verboso (mostra detalhes do processamento)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=OLLAMA_LLM_MODELS[0],
        choices=OLLAMA_LLM_MODELS,
        help="Modelo LLM a ser usado",
    )
    parser.add_argument(
        "-f",
        "--auto-filters",
        action="store_true",
        help="Habilita extração automática de filtros do prompt (padrão: desabilitado)",
    )
    parser.add_argument(
        "--categorias",
        nargs="+",
        type=str,
        help="Categorias para filtrar (por índice 1-based ou nome). Ex: 1 4 ou 'clamor'",
    )
    parser.add_argument(
        "--coletaneas",
        nargs="+",
        type=str,
        help="Coletâneas para filtrar (por índice 1-based ou nome)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Lista todos os modelos LLM disponíveis e sai",
    )
    parser.add_argument(
        "--list-categorias",
        action="store_true",
        help="Lista todas as categorias disponíveis e sai",
    )
    parser.add_argument(
        "--list-coletaneas",
        action="store_true",
        help="Lista todas as coletâneas disponíveis e sai",
    )

    args = parser.parse_args()

    # Inicializa sistema
    rag = HymnRAG(verbose=args.verbose, model=args.model)

    # Lista modelos se solicitado
    if args.list_models:
        print("\n🤖 Modelos LLM disponíveis:")
        for i, model in enumerate(OLLAMA_LLM_MODELS, 1):
            print(f"  {i}. {model}")
        print()
        return

    # Lista categorias/coletâneas se solicitado
    if args.list_categorias:
        print("\n📑 Categorias disponíveis:")
        for i, cat in enumerate(rag.categorias.keys(), 1):
            print(f"  {i}. {cat}")
        print()
        return

    if args.list_coletaneas:
        print("\n📚 Coletâneas disponíveis:")
        for i, col in enumerate(rag.coletaneas.keys(), 1):
            print(f"  {i}. {col}")
        print()
        return

    # Processa filtros manuais
    manual_categorias = None
    manual_coletaneas = None

    if args.categorias:
        manual_categorias = _resolve_categorias(args.categorias, rag.categorias)
        if not manual_categorias:
            print("❌ Nenhuma categoria válida fornecida")
            return

    if args.coletaneas:
        manual_coletaneas = _resolve_coletaneas(args.coletaneas, rag.coletaneas)
        if not manual_coletaneas:
            print("❌ Nenhuma coletânea válida fornecida")
            return

    # Executa consulta com streaming
    if args.verbose:
        print("\n" + "=" * 60)
        print(f"CONSULTA: {args.query}")
        print("=" * 60 + "\n")

    stream_generator = rag.query_stream(
        args.query,
        auto_filters=args.auto_filters,
        manual_categorias=manual_categorias,
        manual_coletaneas=manual_coletaneas,
    )

    if args.verbose:
        print("\n" + "=" * 60)
        print("RESPOSTA:")
        print("=" * 60)

    # Exibe resposta com streaming
    print()
    for chunk in stream_generator:
        print(chunk, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    main()
