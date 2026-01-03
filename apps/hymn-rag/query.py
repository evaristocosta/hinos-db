#!/usr/bin/env python
"""
Script de busca RAG para a Coletânea de Hinos
Uso: python query.py "sua consulta aqui" [--verbose]
"""
import argparse
import sqlite3
import pickle
import re
import unicodedata
import requests
from pathlib import Path
from typing import List, Dict

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from tqdm import tqdm


# ===== CONFIGURAÇÕES =====
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_LLM_MODELS = [
    "gemma3:1b",
    # "deepseek-r1:1.5b",
    "llama3.2:1b",
    "gemma3:4b",
]
LLM_TEMPERATURE = 0.1
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
MAX_RESULTS = 10
VECTOR_SEARCH_K = 8
VECTOR_FETCH_K = 20
BM25_K = 8
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

        # Se não há verso, é um capítulo inteiro
        if verse_start is None:
            label = f"{book_raw.strip()} {chapter}"
            api_ref = f"{api_book} {chapter}"
        else:
            # Com verso (pode ter range)
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

            # Para capítulos inteiros, trunca se for muito longo
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
    # Remove acentos
    nfkd = unicodedata.normalize("NFKD", text)
    text_no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Minúsculas e espaço único
    return " ".join(text_no_accents.lower().split())


def _find_matches_in_text(query: str, candidates: list) -> list:
    """
    Busca candidatos dentro da query usando estratégia híbrida:
    1. Substring match exato (normalizado) - score 1.0
    2. Partial token match - score baseado em tokens comuns
    3. Fuzzy matching (thefuzz) como fallback

    Retorna lista de (candidate, score) ordenada por score.
    """
    from thefuzz import fuzz

    matches = []
    query_norm = _normalize_for_matching(query)
    query_tokens = set(query_norm.split())

    for candidate in candidates:
        candidate_norm = _normalize_for_matching(candidate)

        # Estratégia 1: Substring match direto (melhor caso)
        if candidate_norm in query_norm:
            matches.append((candidate, 1.0))
            continue

        # Estratégia 2: Match por tokens (bom para variações de ordem)
        candidate_tokens = set(candidate_norm.split())
        if len(candidate_tokens) == 0:
            continue

        # Calcula overlap de tokens
        common_tokens = query_tokens & candidate_tokens
        token_ratio = len(common_tokens) / len(candidate_tokens)

        # Se a maioria dos tokens da categoria está na query, considera match
        if token_ratio >= 0.7:
            matches.append((candidate, token_ratio))
            continue

        # Estratégia 3: Fuzzy matching como fallback (para typos)
        # Usa partial_ratio que é ideal para encontrar substring fuzzy
        if len(candidate_norm.split()) <= 4:
            ratio = fuzz.partial_ratio(candidate_norm, query_norm) / 100.0
            # Threshold mais alto para fuzzy, pois já falhou nos outros métodos
            if ratio >= 0.6:
                matches.append((candidate, ratio * 0.8))  # Penaliza fuzzy match

    # Ordena por score (descendente)
    return sorted(matches, key=lambda x: x[1], reverse=True)


def _extract_filters_deterministic(
    question: str, categorias_dict: dict, coletaneas_dict: dict
) -> dict:
    """
    Extrai filtros de forma determinística sem usar LLM.

    Estratégia:
    1. Busca categorias/coletâneas contidas na query (substring + token matching)
    2. Remove as referências encontradas da query
    3. Retorna filtros e query limpa
    """
    question_lower = question.lower()
    found_categorias = []
    found_coletaneas = []

    # Extrai possíveis categorias
    categoria_names = list(categorias_dict.keys())
    cat_matches = _find_matches_in_text(question_lower, categoria_names)

    # Aceita matches com score >= 0.7 (alta confiança)
    for cat_name, score in cat_matches:
        if score >= 0.7:
            found_categorias.append(cat_name)
            # Remove a referência da pergunta (para limpeza)
            question_lower = question_lower.replace(cat_name.lower(), " ")

    # Extrai possíveis coletâneas
    coletanea_names = list(coletaneas_dict.keys())
    col_matches = _find_matches_in_text(question_lower, coletanea_names)

    for col_name, score in col_matches:
        if score >= 0.7:
            found_coletaneas.append(col_name)
            # Remove a referência da pergunta
            question_lower = question_lower.replace(col_name.lower(), " ")

    # Limpa a query removendo espaços extras
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
        self.vector_dir = Path.cwd().parent / "shared" / "rag" / "vectorstore"
        self.chunks_cache = Path.cwd().parent / "shared" / "rag" / "chunks_cache.pkl"
        self.stopwords_path = (
            Path.cwd().parent / "shared" / "assets" / "stopwords-br.txt"
        )

        # Carrega configurações do banco
        self._load_metadata()

        # Inicializa componentes
        self.embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
        self.llm = OllamaLLM(model=model, temperature=LLM_TEMPERATURE)

        # Carrega chunks e vectorstore
        self.chunks = self._load_or_create_chunks()
        self.vectorstore = self._load_or_create_vectorstore()

        # Configura retrievers
        self._setup_retrievers()

        # Configura chains
        self._setup_chains()

        if self.verbose:
            print("✅ Sistema inicializado com sucesso!")
            print(f"🤖 Modelo: {model}")

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
            print(f"📑 Categorias: {list(self.categorias.keys())}")
            print(f"📚 Coletâneas: {list(self.coletaneas.keys())}")

    def _load_or_create_chunks(self) -> List[Document]:
        if self.chunks_cache.exists():
            if self.verbose:
                print(f"💾 Carregando chunks do cache...")
            with open(self.chunks_cache, "rb") as f:
                chunks = pickle.load(f)
            if self.verbose:
                print(f"✓ {len(chunks)} chunks carregados")
            return chunks

        if self.verbose:
            print("⚙️ Criando chunks do zero...")

        # Carrega hinos
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, nome, numero, texto_limpo AS texto, 
                       categoria_id, coletanea_id
                FROM hino
                WHERE texto_limpo IS NOT NULL
                ORDER BY id
                """,
            )
            rows = cur.fetchall()

        # Cria documentos
        docs = []
        iterator = tqdm(rows, desc="Criando documentos") if self.verbose else rows
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

        # Cria chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = []
        iterator = tqdm(docs, desc="Criando chunks") if self.verbose else docs
        for doc in iterator:
            chunks.extend(splitter.split_documents([doc]))

        # Salva cache
        with open(self.chunks_cache, "wb") as f:
            pickle.dump(chunks, f)

        if self.verbose:
            print(f"✓ {len(chunks)} chunks criados e salvos")

        return chunks

    def _load_or_create_vectorstore(self) -> Chroma:
        if self.vector_dir.exists() and (self.vector_dir / "chroma.sqlite3").exists():
            if self.verbose:
                print("💾 Carregando vectorstore...")
            return Chroma(
                embedding_function=self.embeddings,
                persist_directory=str(self.vector_dir),
            )

        if self.verbose:
            print("⚙️ Criando vectorstore...")

        vectorstore = Chroma(
            embedding_function=self.embeddings, persist_directory=str(self.vector_dir)
        )

        batch_size = 64
        iterator = range(0, len(self.chunks), batch_size)
        if self.verbose:
            iterator = tqdm(iterator, desc="Indexando")

        for i in iterator:
            batch = self.chunks[i : i + batch_size]
            vectorstore.add_documents(batch)

        if self.verbose:
            print("✓ Vectorstore criado")

        return vectorstore

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
        # Chain de reescrita
        rewrite_system = """
Reescreva a consulta do usuário para busca em hinos, expandindo com sinônimos e termos relacionados, 
mantendo intenção e concisão. Mantenha as palavras que estiverem entre aspas. 
Não adicione explicações sobre as alterações na consulta.
"""
        rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", rewrite_system),
                ("user", "Consulta: {question}"),
            ]
        )
        self.rewrite_chain = rewrite_prompt | self.llm | StrOutputParser()

        # Prompt de resposta
        answer_system = """
Você é um assistente que responde apenas com base nas opções de hinos fornecidas 
no contexto.
É preferível retornar mais de uma opção, pelo menos três, quando disponível.
Explique os motivos de selecionar tais hinos. 
Cite números (se houver) e títulos.
Se não souber, diga que não está na base.
"""
        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", answer_system),
                (
                    "user",
                    "Pergunta original: {question}\n\nConsulta reescrita: {rewritten}\n\nContexto:\n{context}\n\nResposta:",
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
                print(f"📥 Texto bíblico: {bible_context[:240]}...")

        # Determina filtros a aplicar
        filters = {}

        # Filtros manuais têm prioridade
        if manual_categorias or manual_coletaneas:
            filters = {
                "categorias": manual_categorias,
                "coletaneas": manual_coletaneas,
                "search_query": question,
                "matches_info": {"manual": True},
            }
            if self.verbose:
                print(
                    f"🛠 Usando filtros manuais: categorias={manual_categorias}, coletaneas={manual_coletaneas}"
                )
        # Se auto_filters está habilitado, extrai automaticamente
        elif auto_filters:
            filters = _extract_filters_deterministic(
                question, self.categorias, self.coletaneas
            )
            if self.verbose and (
                filters.get("categorias") or filters.get("coletaneas")
            ):
                print(
                    f"🛠 Filtros automáticos detectados: categorias={filters.get('categorias')}, coletaneas={filters.get('coletaneas')}"
                )
                print(f"   (scores: {filters['matches_info']})")
        else:
            # Sem filtros
            filters = {
                "categorias": None,
                "coletaneas": None,
                "search_query": question,
                "matches_info": {},
            }
            if self.verbose:
                print(f"🛠 Sem filtros (auto_filters=False)")

        search_query = filters.get("search_query", question)

        if self.verbose:
            print(f"📝 Query para busca: {search_query}")

        # Reescreve consulta
        rewritten = self.rewrite_chain.invoke({"question": search_query})
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
            print(f"📚 {len(docs)} hinos encontrados:")
            for doc in docs:
                print(
                    f"  - [{doc.metadata.get('numero') or 'N/A'}] {doc.metadata.get('nome')}"
                )

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
        final_prompt = self.answer_prompt.format(
            question=question, rewritten=rewritten + filter_info, context=context
        )

        if self.verbose:
            print("💬 Gerando resposta...")

        return self.llm.invoke(final_prompt)


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

    # Executa consulta
    if args.verbose:
        print("\n" + "=" * 60)
        print(f"CONSULTA: {args.query}")
        print("=" * 60 + "\n")

    resposta = rag.query(
        args.query,
        auto_filters=args.auto_filters,
        manual_categorias=manual_categorias,
        manual_coletaneas=manual_coletaneas,
    )

    if args.verbose:
        print("\n" + "=" * 60)
        print("RESPOSTA:")
        print("=" * 60)

    print(f"\n{resposta}\n")


if __name__ == "__main__":
    main()
