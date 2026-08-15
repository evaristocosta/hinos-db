#!/usr/bin/env python
"""
Sistema RAG adaptado para usar Hugging Face Inference API
Usa vectorstore e chunks pré-calculados do repositório
"""

import os
import sqlite3
import pickle
import re
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv()

from huggingface_hub import InferenceClient

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.fetch_bible import (
    extract_bible_refs,
    fetch_bible_verses,
    fetch_bible_verses_bibliaapi,
)
from src.extract_categories import extract_filters_deterministic

# ===== CONFIGURAÇÕES =====
# Modelo de embeddings local (usa sentence-transformers)
# Modelo menor para economizar memória (~500MB vs ~1.7GB)
HF_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# HF_EMBED_MODEL = "mixedbread-ai/mxbai-embed-large-v1"  # Modelo maior - desabilitado por uso de memória

# Modelo LLM via Hugging Face InferenceClient
HF_LLM_MODEL = "openai/gpt-oss-20b"

# Configurações de busca
MAX_RESULTS = 25
VECTOR_SEARCH_K = 10
VECTOR_FETCH_K = 30
BM25_K = 15


# ===== CLASSE PRINCIPAL =====
class HymnRAG:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        print("🚀 [DEBUG] Iniciando HymnRAG.__init__()...")

        # --------------------------------------------------
        # 1. Localiza banco de dados
        # --------------------------------------------------
        print("🔍 [DEBUG] Passo 1/8: Localizando database...")
        try:
            self.db_path = self._locate_database()
            print(f"  ✓ database encontrado em: {self.db_path}")
        except Exception as e:
            print(f"  ❌ [ERRO] Falha ao localizar database: {type(e).__name__}: {e}")
            raise

        self.vector_dir = Path(__file__).parent.parent / "assets" / "vectorstore"
        self.chunks_cache = Path(__file__).parent.parent / "assets" / "chunks_cache.pkl"
        self.stopwords_path = (
            Path(__file__).parent.parent / "assets" / "stopwords-br.txt"
        )
        print(f"  ✓ vector_dir: {self.vector_dir}")
        print(f"  ✓ chunks_cache: {self.chunks_cache}")
        print(f"  ✓ stopwords_path: {self.stopwords_path}")

        # --------------------------------------------------
        # 2. Carrega metadados do banco
        # --------------------------------------------------
        print("🔍 [DEBUG] Passo 2/8: Carregando metadados...")
        try:
            self._load_metadata()
            print("  ✓ Metadados carregados com sucesso")
        except Exception as e:
            print(f"  ❌ [ERRO] Falha em _load_metadata: {type(e).__name__}: {e}")
            raise

        # --------------------------------------------------
        # 3. Inicializa embeddings locais
        # --------------------------------------------------
        print("🔍 [DEBUG] Passo 3/8: Inicializando HuggingFaceEmbeddings...")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=HF_EMBED_MODEL,
                model_kwargs={
                    "device": "cpu",
                    "trust_remote_code": True,
                },
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": 16,  # Reduz uso de memória durante encoding
                },
            )
            print(f"  ✓ Embeddings inicializados com modelo: {HF_EMBED_MODEL}")
        except Exception as e:
            print(
                f"  ❌ [ERRO] Falha ao inicializar embeddings: {type(e).__name__}: {e}"
            )
            raise

        # --------------------------------------------------
        # 4. Obtém token HF (Streamlit secrets → .env)
        # --------------------------------------------------
        print("🔍 [DEBUG] Passo 4/8: Obtendo HuggingFace API token...")
        self.hf_token = None
        try:
            import streamlit as st

            if hasattr(st, "secrets"):
                self.hf_token = st.secrets.get("HUGGINGFACE_API_TOKEN")
                if self.hf_token:
                    print("  ✓ Token obtido via Streamlit secrets")
        except ImportError:
            print("  ℹ️  streamlit não disponível, tentando variável de ambiente...")
        except Exception as e:
            print(f"  ⚠️  Erro ao acessar Streamlit secrets: {e}")

        if not self.hf_token:
            print("  🔑 Tentando HUGGINGFACE_API_TOKEN do ambiente...")
            self.hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
            if self.hf_token:
                print("  ✓ Token obtido via variável de ambiente")
            else:
                print(
                    "  ⚠️  HUGGINGFACE_API_TOKEN não encontrado. Configure como variável de ambiente ou Streamlit secret."
                )

        # --------------------------------------------------
        # 5. Inicializa InferenceClient
        # --------------------------------------------------
        print("🔍 [DEBUG] Passo 5/8: Criando InferenceClient...")
        try:
            self.hf_client = InferenceClient(token=self.hf_token)
            print("  ✓ InferenceClient criado com sucesso")
        except Exception as e:
            print(
                f"  ❌ [ERRO] Falha ao criar InferenceClient: {type(e).__name__}: {e}"
            )
            raise

        # --------------------------------------------------
        # 6. Carrega chunks
        # --------------------------------------------------
        print("🔍 [DEBUG] Passo 6/8: Carregando chunks do cache...")
        try:
            self.chunks = self._load_chunks()
            print(f"  ✓ {len(self.chunks)} chunks carregados")
        except Exception as e:
            print(f"  ❌ [ERRO] Falha ao carregar chunks: {type(e).__name__}: {e}")
            raise

        # --------------------------------------------------
        # 7. Carrega vectorstore
        # --------------------------------------------------
        print("🔍 [DEBUG] Passo 7/8: Carregando vectorstore...")
        try:
            self.vectorstore = self._load_vectorstore()
            print("  ✓ Vectorstore carregado com sucesso")
        except Exception as e:
            print(f"  ❌ [ERRO] Falha ao carregar vectorstore: {type(e).__name__}: {e}")
            raise

        # --------------------------------------------------
        # 8. Configura retrievers
        # --------------------------------------------------
        print("🔍 [DEBUG] Passo 8/8: Configurando retrievers...")
        try:
            self._setup_retrievers()
            print("  ✓ Retrievers configurados com sucesso")
        except Exception as e:
            print(
                f"  ❌ [ERRO] Falha ao configurar retrievers: {type(e).__name__}: {e}"
            )
            raise

        # BM25 será inicializado sob demanda (lazy loading)
        self._bm25_initialized = False

        if self.verbose:
            print("✅ Sistema inicializado com sucesso!")
            print(f"🤖 Modelo LLM: {HF_LLM_MODEL}")
            print(f"📦 Modelo Embeddings: {HF_EMBED_MODEL}")

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
        if self.verbose:
            print(f"  📂 Conectando ao banco: {self.db_path}")

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()

            cur.execute("SELECT count(*) FROM hino")
            self.total_hinos = cur.fetchone()[0]
            print(f"  📊 Total de hinos: {self.total_hinos}")

            cur.execute("SELECT id, nome as descricao FROM categoria")
            rows_cat = cur.fetchall()
            self.categorias = {}
            for row in rows_cat:
                cat_id, cat_desc = row
                if cat_desc is None:
                    print(
                        f"  ⚠️  [DEBUG] Categoria ID {cat_id} tem 'descricao' = None! Ignorando."
                    )
                    continue
                self.categorias[cat_desc.lower()] = cat_id
            print(f"  📂 {len(self.categorias)} categorias carregadas")

            cur.execute("SELECT id, nome FROM coletanea")
            rows_col = cur.fetchall()
            self.coletaneas = {}
            for row in rows_col:
                col_id, col_nome = row
                if col_nome is None:
                    print(
                        f"  ⚠️  [DEBUG] Coletânea ID {col_id} tem 'nome' = None! Ignorando."
                    )
                    continue
                self.coletaneas[col_nome.lower()] = col_id
            print(f"  📂 {len(self.coletaneas)} coletâneas carregadas")

        if self.verbose:
            print(f"📊 Total de hinos: {self.total_hinos}")

    def _load_chunks(self) -> List[Document]:
        """Carrega chunks PRÉ-CALCULADOS do cache"""
        if not self.chunks_cache.exists():
            raise FileNotFoundError(
                f"Cache de chunks não encontrado: {self.chunks_cache}\n"
                "Execute o query.py localmente primeiro para gerar o cache."
            )

        if self.verbose:
            print(f"💾 Carregando chunks do cache...")

        with open(self.chunks_cache, "rb") as f:
            chunks = pickle.load(f)

        if self.verbose:
            print(f"✓ {len(chunks)} chunks carregados")

        return chunks

    def _load_vectorstore(self) -> Chroma:
        """Carrega vectorstore PRÉ-CALCULADO"""
        if (
            not self.vector_dir.exists()
            or not (self.vector_dir / "chroma.sqlite3").exists()
        ):
            raise FileNotFoundError(
                f"Vectorstore não encontrado: {self.vector_dir}\n"
                "Execute o query.py localmente primeiro para gerar o vectorstore."
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

        # BM25 retriever será inicializado sob demanda (lazy loading)
        self.bm25_retriever = None

    def _init_bm25_if_needed(self):
        """Inicializa BM25 sob demanda para economizar memória"""
        if self._bm25_initialized or not self.chunks:
            return

        if self.verbose:
            print("💾 Inicializando BM25 retriever...")

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
        self._bm25_initialized = True

    def _call_hf_api_stream(self, prompt: str, max_tokens: int = 512):
        """Chama a API Hugging Face com streaming via InferenceClient"""
        if not self.hf_token:
            yield "❌ Token de API não configurado"
            return

        try:
            # Usa chat_completion com streaming
            messages = [{"role": "user", "content": prompt}]

            for chunk in self.hf_client.chat_completion(
                messages=messages,
                model=HF_LLM_MODEL,
                max_tokens=max_tokens,
                temperature=0.1,
                stream=True,
            ):
                if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        yield delta.content

        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "429" in error_msg:
                yield "❌ Rate limit atingido. Aguarde alguns segundos."
            elif "503" in error_msg or "loading" in error_msg:
                yield "❌ Modelo está carregando. Aguarde e tente novamente."
            elif "timeout" in error_msg:
                yield "❌ Timeout na requisição. Tente novamente."
            else:
                yield f"❌ Erro ao chamar API: {str(e)}"

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

        # BM25 - inicializa apenas quando necessário
        self._init_bm25_if_needed()
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
        """Consulta com streaming de resposta - retorna (docs, generator)"""
        # Extrai referências bíblicas
        bible_refs = extract_bible_refs(question)
        bible_context = fetch_bible_verses_bibliaapi(bible_refs) if bible_refs else ""

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

        effective_query = search_query
        if bible_context:
            effective_query = search_query + "\n\n" + '"' + bible_context[:700] + '"'
            if self.verbose:
                print("🔎 Consulta enriquecida com texto bíblico")

        if self.verbose:
            print(f"📝 Query para busca: {effective_query}")

        # Busca hinos diretamente com a query original
        docs = self._hybrid_retrieve_filtered(effective_query, filters)

        if self.verbose:
            print(f"📚 {len(docs)} hinos encontrados")

        if not docs:

            def empty_generator():
                yield "❌ Nenhum hino encontrado com esses critérios."

            return [], empty_generator()

        # Formata contexto
        context = self._format_docs(docs)
        if bible_context:
            context = context + "\n\nTrechos bíblicos:\n" + '"' + bible_context + '"'

        filter_info = ""
        if filters.get("categorias") or filters.get("coletaneas"):
            filter_info = f"\nFiltros aplicados: Categorias={filters.get('categorias')}, Coletâneas={filters.get('coletaneas')}"

        # Gera resposta com streaming
        prompt = f"""<s>[INST] Você é um assistente que responde apenas com base nas opções de hinos fornecidas no contexto.
É preferível retornar mais de uma opção, pelo menos três, quando disponível, a não ser que requisitado diferente na pergunta.
Explique de maneira sucinta os motivos de selecionar tais hinos.
Cite os números dos hinos (se houver) e principalmente os títulos.
Se não souber, diga que não encontrou na base.
Responda SOMENTE em PORTUGUÊS DO BRASIL.

Pergunta: {question}{filter_info}

Contexto:
{context}

Resposta: [/INST]"""

        # Retorna docs e o generator
        return docs, bible_context, self._call_hf_api_stream(prompt, max_tokens=None)
