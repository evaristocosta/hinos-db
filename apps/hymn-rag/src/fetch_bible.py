import unicodedata
import requests
from typing import List
import re
import os
from dotenv import load_dotenv

load_dotenv()

try:
    from bible_constants import BIBLE_BOOK_MAP, BIBLE_BOOK_PATTERNS
except ImportError:
    from utils.bible_constants import BIBLE_BOOK_MAP, BIBLE_BOOK_PATTERNS
except Exception:
    raise


# ===== UTILIDADES PARA REFERÊNCIAS BÍBLICAS =====
def _normalize_text(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _normalize_book_key(book: str) -> str:
    return _normalize_text(book).replace(" ", "")


def extract_bible_refs(text: str) -> List[dict]:
    """
    Extrai referências bíblicas do texto usando busca baseada em lista de livros conhecidos.
    Busca padrões como: "1 Samuel 2", "João 3:16", "Gênesis 1:1-3"
    """
    if not text:
        return []

    refs: List[dict] = []
    seen = set()
    text_normalized = _normalize_text(text)

    # Procura cada padrão de livro no texto
    for patterns in BIBLE_BOOK_PATTERNS:
        for pattern in patterns:
            pattern_normalized = _normalize_text(pattern)

            # Busca todas as ocorrências do livro no texto
            start_idx = 0
            while True:
                idx = text_normalized.find(pattern_normalized, start_idx)
                if idx == -1:
                    break

                # Verifica se é uma palavra completa (não parte de outra palavra)
                before_ok = idx == 0 or not text_normalized[idx - 1].isalnum()
                after_idx = idx + len(pattern_normalized)
                after_ok = (
                    after_idx >= len(text_normalized)
                    or not text_normalized[after_idx].isalnum()
                )

                if not (before_ok and after_ok):
                    start_idx = idx + 1
                    continue

                # Pega o texto original (não normalizado) do livro
                book_original = text[idx:after_idx].strip()

                # Procura por números após o nome do livro
                remaining = text[after_idx:].lstrip()

                # Regex para capturar capítulo e versículos após o livro
                chapter_verse_match = re.match(
                    r"^(\d{1,3})(?:[:\.](\d{1,3})(?:-(\d{1,3}))?)?", remaining
                )

                if chapter_verse_match:
                    chapter = chapter_verse_match.group(1)
                    verse_start = chapter_verse_match.group(2)
                    verse_end = chapter_verse_match.group(3)

                    # Mapeia o livro para o código da API
                    key = _normalize_book_key(pattern)
                    if key not in BIBLE_BOOK_MAP:
                        start_idx = idx + 1
                        continue

                    api_book = BIBLE_BOOK_MAP[key]

                    # Cria as referências
                    if verse_start is None:
                        label = [f"{book_original} {chapter}"]
                        api_ref = [f"{api_book}/{chapter}"]
                    else:
                        if verse_end is not None and int(verse_end) > int(verse_start):
                            label = []
                            api_ref = []
                            for v in range(int(verse_start), int(verse_end) + 1):
                                label.append(f"{book_original} {chapter}:{v}")
                                api_ref.append(f"{api_book}/{chapter}/{v}")
                        else:
                            label = [f"{book_original} {chapter}:{verse_start}"]
                            api_ref = [f"{api_book}/{chapter}/{verse_start}"]

                    # Adiciona à lista se não for duplicado
                    for l, a in zip(label, api_ref):
                        if a not in seen:
                            seen.add(a)
                            refs.append(
                                {
                                    "label": l,
                                    "api_ref": a,
                                    "type": (
                                        "chapter" if verse_start is None else "verse"
                                    ),
                                }
                            )

                start_idx = idx + 1

    return refs


def fetch_bible_verses(refs: List[dict], max_chars: int = 1200) -> str:
    """Busca versículos usando a API abibliadigital.com.br."""
    verses = []
    total_len = 0
    for ref in refs:
        try:
            resp = requests.get(
                f"https://www.abibliadigital.com.br/api/verses/acf/{requests.utils.requote_uri(ref['api_ref'])}",
                headers={
                    "Authorization": f"Bearer {os.getenv('ABIBLIADIGITAL_API_TOKEN')}"
                },
                timeout=8,
            )
            if resp.status_code != 200:
                continue

            data = resp.json()
            is_chapter = ref.get("type") == "chapter"
            if not is_chapter:
                verse_text = data.get("text", "").strip()
            else:
                verses_data = data.get("verses", [])
                text_parts = [v.get("text", "").strip() for v in verses_data]
                verse_text = " ".join([t for t in text_parts if t])

            if not verse_text:
                continue

            if len(verse_text) > 800:
                verse_text = verse_text[:800] + "..."

            snippet = f"{ref['label']} — {verse_text}"
            verses.append(snippet)
            total_len += len(snippet)
            if total_len >= max_chars:
                break
        except Exception:
            continue

    return "\n".join(verses)


def fetch_bible_verses_bibliaapi(refs: List[dict], max_chars: int = 1200) -> str:
    """
    Método alternativo: busca versículos usando a API bibliaapi.com.br.

    Endpoints usados:
      - Capítulo: GET /versions/acf/books/{book}/chapters/{chapter}
      - Versículo: GET /versions/acf/books/{book}/chapters/{chapter}/verses/{verse}
    """
    base_url = "https://bibliaapi.com.br/api/v2"
    token = os.getenv("BIBLIAAPI_API_KEY") or os.getenv("ABIBLIADIGITAL_API_TOKEN")

    verses = []
    total_len = 0

    for ref in refs:
        try:
            # api_ref tem formato "livro/capitulo" ou "livro/capitulo/versiculo"
            parts = ref["api_ref"].split("/")
            if len(parts) < 2:
                continue

            book_slug = parts[0]
            chapter = parts[1]
            verse = parts[2] if len(parts) > 2 else None

            is_chapter = ref.get("type") == "chapter"
            # caso especifico dessa api: "atos"
            if book_slug == "at":
                book_slug = "atos"

            if verse and not is_chapter:
                # Versículo específico
                url = f"{base_url}/versions/acf/books/{book_slug}/chapters/{chapter}/verses/{verse}"
            else:
                # Capítulo completo
                url = f"{base_url}/versions/acf/books/{book_slug}/chapters/{chapter}"

            resp = requests.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                timeout=8,
            )

            if resp.status_code != 200:
                continue

            data = resp.json()

            # A resposta pode vir direta ou envelopada em "data"
            payload = data.get("data", data)

            if is_chapter or verse is None:
                # Capítulo completo → extrair todos os versículos
                verses_raw = payload.get("verses", [])
                if not verses_raw and isinstance(payload, list):
                    verses_raw = payload
                text_parts = []
                for v in verses_raw:
                    txt = v.get("text", "").strip()
                    if txt:
                        text_parts.append(txt)
                verse_text = " ".join(text_parts)
            else:
                # Versículo único
                verse_text = payload.get("text", "").strip()

            if not verse_text:
                continue

            if len(verse_text) > 800:
                verse_text = verse_text[:800] + "..."

            snippet = f"{ref['label']} — {verse_text}"
            verses.append(snippet)
            total_len += len(snippet)
            if total_len >= max_chars:
                break
        except Exception:
            continue

    return "\n".join(verses)
