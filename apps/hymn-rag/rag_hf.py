#!/usr/bin/env python
"""
Sistema RAG adaptado para usar Hugging Face Inference API
Usa vectorstore e chunks pré-calculados do repositório
"""
import os
import sqlite3
import pickle
import re
import unicodedata
import requests
from pathlib import Path
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ===== CONFIGURAÇÕES =====
# Modelo de embeddings local (usa sentence-transformers)
# HF_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"

# Modelo LLM via Hugging Face InferenceClient
# Usando modelo que funciona bem com text-generation
HF_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# Configurações de busca
MAX_RESULTS = 10
VECTOR_SEARCH_K = 8
VECTOR_FETCH_K = 20
BM25_K = 8

# Categorias e coletâneas (mantidas para compatibilidade)
CATEGORIAS = [
    "clamor",
    "invocação e comunhão",
    "dedicação",
    "morte, ressurreição e salvação",
    "consolo e encorajamento",
    "santificação e derramamento do espírito santo",
    "volta de jesus e eternidade",
    "louvor",
    "salmos de louvor",
    "grupo de louvor",
    "corinhos",
]
COLETANEAS = [
    "coletânea de louvores - igreja cristã maranata - edição 2018",
    "coletânea de crianças - igreja cristã maranata - edição 2019",
    "louvores avulsos de crianças, intermediários e adolescentes - edição 2022",
    "louvores avulsos",
    "louvores avulsos - manual",
    "louvores avulsos cias - manual",
]


# ===== UTILIDADES PARA REFERÊNCIAS BÍBLICAS =====
def _normalize_text(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


BIBLE_BOOK_MAP: Dict[str, str] = {
    "genesis": "Genesis",
    "exodo": "Exodus",
    "levitico": "Leviticus",
    "numeros": "Numbers",
    "deuteronomio": "Deuteronomy",
    "josue": "Joshua",
    "juizes": "Judges",
    "rute": "Ruth",
    "1samuel": "1 Samuel",
    "2samuel": "2 Samuel",
    "1reis": "1 Kings",
    "2reis": "2 Kings",
    "1cronicas": "1 Chronicles",
    "2cronicas": "2 Chronicles",
    "esdras": "Ezra",
    "neemias": "Nehemiah",
    "ester": "Esther",
    "jo": "Job",
    "salmos": "Psalms",
    "proverbios": "Proverbs",
    "eclesiastes": "Ecclesiastes",
    "cantico": "Song of Solomon",
    "canticos": "Song of Solomon",
    "cantares": "Song of Solomon",
    "isaias": "Isaiah",
    "jeremias": "Jeremiah",
    "lamentacoes": "Lamentations",
    "ezequiel": "Ezekiel",
    "daniel": "Daniel",
    "oseias": "Hosea",
    "joel": "Joel",
    "amos": "Amos",
    "obadias": "Obadiah",
    "jonas": "Jonah",
    "miqueias": "Micah",
    "naum": "Nahum",
    "habacuque": "Habakkuk",
    "sofonias": "Zephaniah",
    "ageu": "Haggai",
    "zacarias": "Zechariah",
    "malaquias": "Malachi",
    "mateus": "Matthew",
    "marcos": "Mark",
    "lucas": "Luke",
    "joao": "John",
    "atos": "Acts",
    "romanos": "Romans",
    "1corintios": "1 Corinthians",
    "2corintios": "2 Corinthians",
    "galatas": "Galatians",
    "efesios": "Ephesians",
    "filipenses": "Philippians",
    "colossenses": "Colossians",
    "1tessalonicenses": "1 Thessalonians",
    "2tessalonicenses": "2 Thessalonians",
    "1timoteo": "1 Timothy",
    "2timoteo": "2 Timothy",
    "tito": "Titus",
    "filemom": "Philemon",
    "hebreus": "Hebrews",
    "tiago": "James",
    "1pedro": "1 Peter",
    "2pedro": "2 Peter",
    "1joao": "1 John",
    "2joao": "2 John",
    "3joao": "3 John",
    "judas": "Jude",
    "apocalipse": "Revelation",
}

REF_RE = re.compile(
    r"(?i)([1-3]?\s?[A-Za-zÀ-ÿçãõâêôáéíóú]+(?:\s+dos\s+canticos|\s+de\s+canticos|\s+dos\s+reis|\s+cronicas|\s+corintios|\s+tessalonicenses|\s+pedro|\s+joao)*)\s+(\d{1,3})(?:[:\.](\d{1,3})(?:-(\d{1,3}))?)?"
)


def _normalize_book_key(book: str) -> str:
    return _normalize_text(book).replace(" ", "")


def extract_bible_refs(text: str) -> List[dict]:
    refs: List[dict] = []
    seen = set()
    for match in REF_RE.finditer(text or ""):
        book_raw = match.group(1)
        chapter = match.group(2)
        verse_start = match.group(3)
        verse_end = match.group(4)

        key = _normalize_book_key(book_raw)
        if key not in BIBLE_BOOK_MAP:
            continue

        api_book = BIBLE_BOOK_MAP[key]

        if verse_start is None:
            label = f"{book_raw.strip()} {chapter}"
            api_ref = f"{api_book} {chapter}"
        else:
            label = f"{book_raw.strip()} {chapter}:{verse_start}{('-' + verse_end) if verse_end else ''}"
            api_ref = f"{api_book} {chapter}:{verse_start}{('-' + verse_end) if verse_end else ''}"

        if api_ref in seen:
            continue
        seen.add(api_ref)
        refs.append(
            {
                "label": label,
                "api_ref": api_ref,
                "type": "chapter" if verse_start is None else "verse",
            }
        )
    return refs


def fetch_bible_verses(
    refs: List[dict], translation: str = "almeida", max_chars: int = 1200
) -> str:
    verses = []
    total_len = 0
    for ref in refs:
        try:
            resp = requests.get(
                f"https://bible-api.com/{requests.utils.requote_uri(ref['api_ref'])}",
                params={"translation": translation},
                timeout=8,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            text_parts = [v.get("text", "").strip() for v in data.get("verses", [])]
            verse_text = " ".join([t for t in text_parts if t])
            if not verse_text:
                continue

            is_chapter = ref.get("type") == "chapter"
            if is_chapter and len(verse_text) > 800:
                verse_text = verse_text[:800] + "..."

            snippet = f"{ref['label']} — {verse_text}"
            verses.append(snippet)
            total_len += len(snippet)
            if total_len >= max_chars:
                break
        except Exception:
            continue
    return "\n".join(verses)


# ===== EXTRAÇÃO DETERMINÍSTICA DE FILTROS =====
def _normalize_for_matching(text: str) -> str:
    """Normaliza texto para matching: remove acentos, converte a minúsculas, remove espaços extras."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    text_no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(text_no_accents.lower().split())


def _find_matches_in_text(query: str, candidates: list) -> list:
    """
    Busca candidatos dentro da query usando estratégia híbrida.
    Retorna lista de (candidate, score) ordenada por score.
    """
    from thefuzz import fuzz

    matches = []
    query_norm = _normalize_for_matching(query)
    query_tokens = set(query_norm.split())

    for candidate in candidates:
        candidate_norm = _normalize_for_matching(candidate)

        if candidate_norm in query_norm:
            matches.append((candidate, 1.0))
            continue

        candidate_tokens = set(candidate_norm.split())
        if len(candidate_tokens) == 0:
            continue

        common_tokens = query_tokens & candidate_tokens
        token_ratio = len(common_tokens) / len(candidate_tokens)

        if token_ratio >= 0.7:
            matches.append((candidate, token_ratio))
            continue

        if len(candidate_norm.split()) <= 4:
            ratio = fuzz.partial_ratio(candidate_norm, query_norm) / 100.0
            if ratio >= 0.6:
                matches.append((candidate, ratio * 0.8))

    return sorted(matches, key=lambda x: x[1], reverse=True)


def _extract_filters_deterministic(
    question: str, categorias_dict: dict, coletaneas_dict: dict
) -> dict:
    """Extrai filtros de forma determinística sem usar LLM."""
    question_lower = question.lower()
    found_categorias = []
    found_coletaneas = []

    categoria_names = list(categorias_dict.keys())
    cat_matches = _find_matches_in_text(question_lower, categoria_names)

    for cat_name, score in cat_matches:
        if score >= 0.7:
            found_categorias.append(cat_name)
            question_lower = question_lower.replace(cat_name.lower(), " ")

    coletanea_names = list(coletaneas_dict.keys())
    col_matches = _find_matches_in_text(question_lower, coletanea_names)

    for col_name, score in col_matches:
        if score >= 0.7:
            found_coletaneas.append(col_name)
            question_lower = question_lower.replace(col_name.lower(), " ")

    cleaned_query = " ".join(question_lower.split())

    return {
        "categorias": found_categorias if found_categorias else None,
        "coletaneas": found_coletaneas if found_coletaneas else None,
        "search_query": cleaned_query if cleaned_query else question,
        "matches_info": {
            "categorias_scores": cat_matches[:3],
            "coletaneas_scores": col_matches[:3],
        },
    }


# ===== CLASSE PRINCIPAL =====
class HymnRAG:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.db_path = self._locate_database()
        self.vector_dir = Path.cwd().parent / "shared" / "rag" / "vectorstore"
        self.chunks_cache = Path.cwd().parent / "shared" / "rag" / "chunks_cache.pkl"
        self.stopwords_path = (
            Path.cwd().parent / "shared" / "assets" / "stopwords-br.txt"
        )

        # Carrega configurações do banco
        self._load_metadata()

        # Inicializa embeddings locais
        self.embeddings = HuggingFaceEmbeddings(
            model_name=HF_EMBED_MODEL,
            model_kwargs={"device": "cpu", "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Obtém token da API Hugging Face
        load_dotenv()

        self.hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not self.hf_token:
            print(
                "⚠️ HUGGINGFACE_API_TOKEN não encontrado. Configure como variável de ambiente."
            )

        # Inicializa InferenceClient
        self.hf_client = InferenceClient(token=self.hf_token)

        # Carrega chunks e vectorstore PRÉ-CALCULADOS
        self.chunks = self._load_chunks()
        self.vectorstore = self._load_vectorstore()

        # Configura retrievers
        self._setup_retrievers()

        if self.verbose:
            print("✅ Sistema inicializado com sucesso!")
            print(f"🤖 Modelo LLM: {HF_LLM_MODEL}")
            print(f"📦 Modelo Embeddings: {HF_EMBED_MODEL}")

    def _locate_database(self) -> Path:
        candidates = [
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

            cur.execute("SELECT count(*) FROM hino")
            self.total_hinos = cur.fetchone()[0]

            cur.execute("SELECT id, descricao FROM categoria")
            self.categorias = {row[1].lower(): row[0] for row in cur.fetchall()}

            cur.execute("SELECT id, nome FROM coletanea")
            self.coletaneas = {row[1].lower(): row[0] for row in cur.fetchall()}

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

        # BM25 retriever
        if self.chunks:
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

    def _call_hf_api(self, prompt: str, max_tokens: int = 512) -> str:
        """Chama a API Hugging Face via InferenceClient oficial"""
        if not self.hf_token:
            return "❌ Token de API não configurado"

        try:
            # Usa chat_completion para melhor compatibilidade
            messages = [{"role": "user", "content": prompt}]

            response = self.hf_client.chat_completion(
                messages=messages,
                model=HF_LLM_MODEL,
                max_tokens=max_tokens,
                temperature=0.1,
            )

            # Extrai o conteúdo da resposta
            if hasattr(response, "choices") and len(response.choices) > 0:
                return response.choices[0].message.content.strip()
            else:
                return str(response).strip()

        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "429" in error_msg:
                return "❌ Rate limit atingido. Aguarde alguns segundos."
            elif "503" in error_msg or "loading" in error_msg:
                return "❌ Modelo está carregando. Aguarde e tente novamente."
            elif "timeout" in error_msg:
                return "❌ Timeout na requisição. Tente novamente."
            else:
                return f"❌ Erro ao chamar API: {str(e)}"

    def _rewrite_query(self, question: str) -> str:
        """Reescreve a consulta usando HF API"""
        prompt = f"""<s>[INST] Reescreva a consulta do usuário para busca em hinos, expandindo com sinônimos e 
termos relacionados, mantendo intenção e concisão. Mantenha as palavras que estiverem entre aspas. 
Não adicione explicações sobre as alterações na consulta.
Responda somente em português.

Consulta: {question} [/INST]"""

        return self._call_hf_api(prompt, max_tokens=128)

    def _generate_answer(self, question: str, rewritten: str, context: str) -> str:
        """Gera resposta usando HF API"""
        prompt = f"""<s>[INST] Você é um assistente que responde apenas com base nas opções de hinos fornecidas no contexto.
É preferível retornar mais de uma opção, pelo menos três, quando disponível.
Explique os motivos de selecionar tais hinos.
Cite números (se houver) e títulos.
Se não souber, diga que não está na base.
Responda somente em português.

Pergunta original: {question}

Consulta reescrita: {rewritten}

Contexto:
{context}

Resposta: [/INST]"""

        return self._call_hf_api(prompt, max_tokens=512)

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

            if len(vec_docs) == 0:
                vec_docs = self.vector_retriever.invoke(search_query)
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

    def query(
        self,
        question: str,
        auto_filters: bool = False,
        manual_categorias: List[str] = None,
        manual_coletaneas: List[str] = None,
    ) -> str:
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
            filters = _extract_filters_deterministic(
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

        if self.verbose:
            print(f"📝 Query para busca: {search_query}")

        # Reescreve consulta
        rewritten = self._rewrite_query(search_query)
        if self.verbose:
            print(f"📝 Consulta reescrita: {rewritten}")

        # Enriquece com texto bíblico
        effective_query = rewritten
        if bible_context:
            effective_query = rewritten + "\n\n" + '"' + bible_context[:700] + '"'
            if self.verbose:
                print("🔎 Consulta enriquecida com texto bíblico")

        # Busca hinos
        docs = self._hybrid_retrieve_filtered(effective_query, filters)

        if self.verbose:
            print(f"📚 {len(docs)} hinos encontrados")

        if not docs:
            return "❌ Nenhum hino encontrado com esses critérios."

        # Formata contexto
        context = self._format_docs(docs)
        if bible_context:
            context = context + "\n\nTrechos bíblicos:\n" + '"' + bible_context + '"'

        filter_info = ""
        if filters.get("categorias") or filters.get("coletaneas"):
            filter_info = f"\nFiltros: {filters}"

        # Gera resposta
        return self._generate_answer(question, rewritten + filter_info, context)
