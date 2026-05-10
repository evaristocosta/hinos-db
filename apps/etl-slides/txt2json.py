"""Convert slide TXT dumps to structured JSON-ready lists.

This module reads `slides_txt/*.txt`, parses shapes and text, applies
tagging rules and merges bis/brace shapes into a single text block per slide.
Refactored for clarity and testability.
"""

from glob import glob
import json
import logging
import math
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Regex / constants
AUTO_SHAPE_RE = re.compile(r"_AUTO_SHAPE_([A-Z_]+)")
FREEFORM_RE = re.compile(r"_FREEFORM_")

CORS_RE = re.compile(
    r"\s*CORO(:)?\s*(\(BIS\)|\(?\d\s*X\)?)?(:)?\s*\n", re.IGNORECASE | re.MULTILINE
)
INSTRUMENTOS_RE = re.compile(r"\n\s*INSTRUMENTOS\s*(\n|$)", re.IGNORECASE)
FINAL_RE = re.compile(r"(\n|^)\s*FINAL\s*(:)?\s*", re.IGNORECASE)
VAROES_RE = re.compile(r"\(?VARÕES\)?\s*(\n|$)", re.IGNORECASE)
SERVAS_RE = re.compile(r"\(?SERVAS\)?", re.IGNORECASE)
H_RE = re.compile(r"\(H\)|^\s*H\s*$", re.IGNORECASE)
S_RE = re.compile(r"\(S\)|^\s*S\s*$", re.IGNORECASE)
M_RE = re.compile(r"\(M\)|^\s*M\s*$", re.IGNORECASE)
T_RE = re.compile(r"\(T\)|^\s*T\s*$", re.IGNORECASE)
REPETIR_RE = re.compile(
    r"REPETIR\s+(A\s+1a\s+ESTROFE\s+\dX|O\s+LOUVOR|O\s+HINO)", re.IGNORECASE
)
REPETIR_NO_FINAL_RE = re.compile(r"\((\d+)X\s+NO\s+FINAL\)", re.IGNORECASE)
REPETIR_X_VEZES_RE = re.compile(r"\(?(\d+)X\)?", re.IGNORECASE)
BIS_RE = re.compile(
    r"\(?BIS(\s+NO\s+FINAL|\s*\dX)?\)?\s*$", re.IGNORECASE | re.MULTILINE
)
START_BIS_TAG_RE = re.compile(r"^<bis(\s+modo=\".*\")?>\s*(\n|$)")


def find_txt_files(path: str = "slides_txt") -> List[str]:
    return glob(str(Path(path) / "*.txt"))


def parse_txt_file(file_txt: str) -> List[Dict]:
    """Parse a single TXT dump into a list of praises (each with slides and shapes).

    Returns the same structure used previously: list[ {"slides": [ {"slide": n, "shapes": [...]}, ...] }, ...]
    """
    LOGGER.info("Parsing %s", file_txt)
    with open(file_txt, "r", encoding="utf-8") as fh:
        text = fh.read()

    praises_out: List[Dict] = []
    praises_raw = text.split("__END__")

    for praise in tqdm(praises_raw, desc="Processing praises", total=len(praises_raw)):
        praise_struct = {"slides": []}

        for slide_blob in praise.split("SLIDE_"):
            slide_num = slide_blob.split("\n")[0].strip()
            if not slide_num:
                continue

            slide_struct: Dict = {"slide": slide_num, "shapes": []}

            for shape_blob in slide_blob.split("\nSHAPE_"):
                record = False
                text_inside = ""
                shape_struct: Dict = {}

                for line in shape_blob.split("\n"):
                    if "SHAPE_" in line:
                        m_auto = AUTO_SHAPE_RE.search(line)
                        if m_auto:
                            shape_struct["shape"] = "AUTO_SHAPE"
                            shape_struct["auto_shape"] = m_auto.group(1)
                        if FREEFORM_RE.search(line):
                            shape_struct["shape"] = "FREEFORM"

                    if "HEIGHT_" in line:
                        try:
                            shape_struct["height"] = int(line.split("_")[1])
                        except Exception:
                            pass
                    if "TOP_" in line:
                        try:
                            shape_struct["top"] = int(line.split("_")[1])
                        except Exception:
                            pass

                    if "END_TEXT" in line:
                        record = False

                    if record:
                        text_inside += line + "\n"
                    else:
                        if text_inside:
                            shape_struct["text"] = text_inside[:-1]

                    if "START_TEXT" in line:
                        text_inside = ""
                        record = True

                if shape_struct and text_inside.strip().lower() not in ["índice"]:
                    slide_struct["shapes"].append(shape_struct)

            if slide_struct["shapes"]:
                praise_struct["slides"].append(slide_struct)

        if praise_struct["slides"]:
            praises_out.append(praise_struct)

    return praises_out


def txt2list(path: str = "slides_txt") -> List[Dict]:
    files_txt = find_txt_files(path)
    LOGGER.info("Found %d files", len(files_txt))
    all_praises: List[Dict] = []
    for f in files_txt:
        all_praises.extend(parse_txt_file(f))
    LOGGER.info("Total praises parsed: %d", len(all_praises))
    return all_praises


# ----- Tagging and processing -----


def _normalize_tag_value(value: str) -> str:
    value = re.sub(r"[()]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip().upper()
    return value


def tagify_text(text: str) -> str:
    if not text:
        return text
    tagged = text.replace("\t\n", "\n").replace("\t", "\n")

    def repl_coro(m: re.Match) -> str:
        rep = m.group(2)
        if rep:
            rep = _normalize_tag_value(rep)
            return f'<coro tipo="{rep}">\n'
        return "<coro>\n"

    tagged = CORS_RE.sub(repl_coro, tagged)
    tagged = REPETIR_RE.sub(
        lambda m: f'<repetir tipo="{_normalize_tag_value(m.group(1))}x">', tagged
    )
    tagged = INSTRUMENTOS_RE.sub("\n<instrumentos>\n", tagged)
    tagged = FINAL_RE.sub("\n<final>", tagged)

    def repl_bis(m: re.Match) -> str:
        modo = m.group(1)
        if modo:
            modo = _normalize_tag_value(modo)
            return f'<bis tipo="{modo}">'
        return "<bis>"

    tagged = BIS_RE.sub(repl_bis, tagged)
    tagged = REPETIR_X_VEZES_RE.sub(lambda m: f'<bis tipo="{m.group(1)}x">', tagged)
    tagged = VAROES_RE.sub("<h>\n", tagged)
    tagged = SERVAS_RE.sub("<m>", tagged)
    tagged = H_RE.sub("<h>", tagged)
    tagged = S_RE.sub("<m>", tagged)
    tagged = M_RE.sub("<m>", tagged)
    tagged = T_RE.sub("<t>", tagged)
    tagged = REPETIR_NO_FINAL_RE.sub(
        lambda m: f'<repetir tipo="no-final,{m.group(1)}x">', tagged
    )

    return tagged


def apply_tags_to_all_praises(praises: List[Dict]) -> None:
    for praise in praises:
        for slide in praise.get("slides", []):
            for shape in slide.get("shapes", []):
                original = shape.get("text")
                if not original:
                    continue
                shape["text"] = original
                shape["text_processado"] = tagify_text(original)


def get_texts(d: object, key: str = "text_processado") -> Iterable[str]:
    if isinstance(d, dict):
        for k, v in d.items():
            if k == key and isinstance(v, str):
                yield v
            else:
                yield from get_texts(v, key=key)
    elif isinstance(d, list):
        for v in d:
            yield from get_texts(v, key=key)


def return_possible_title(texts: List[str]) -> tuple[Optional[str], Optional[str]]:
    possible = [t for t in texts if "<" not in t and t.strip() != ""]
    if not possible:
        return None, None
    min_string = min(possible, key=len)
    title = min_string.upper().replace("\n", " ").strip()
    title = re.sub(r"\s+", " ", title)
    return title, min_string


def has_bis_tag(shape: Dict, pattern: re.Pattern) -> bool:
    text = shape.get("text_processado", "")
    return bool(text) and pattern.search(text) is not None


def build_merged_shape_for_bis(slide: Dict, pattern: re.Pattern) -> Optional[Dict]:
    shapes = slide.get("shapes", [])
    if not any(has_bis_tag(s, pattern) for s in shapes):
        return None

    textos_processados: List[str] = []
    textos: List[str] = []
    height_texto = 0
    top_texto = 0
    height_brace: List[int] = []
    top_brace: List[int] = []
    modo_bis = None
    bis_tag = "<bis>"

    for shape in shapes:
        is_brace = (
            shape.get("shape") == "AUTO_SHAPE"
            and shape.get("auto_shape") == "RIGHT_BRACE"
        ) or (shape.get("shape") == "FREEFORM")
        if is_brace:
            height_brace.append(shape.get("height", 0))
            top_brace.append(shape.get("top", 0))

        if "text_processado" in shape and not has_bis_tag(shape, pattern):
            height_texto += shape.get("height", 0)
            top_texto += shape.get("top", 0)
            textos_processados.append(shape["text_processado"])
            textos.append(shape.get("text", ""))

        if has_bis_tag(shape, pattern):
            m = pattern.search(shape["text_processado"])  # type: ignore[arg-type]
            modo_bis = m.group(1) if m is not None else None
            bis_tag = f"<bis{modo_bis}>" if modo_bis else "<bis>"

    texto_processado_unico = "".join(textos_processados)
    texto_unico = "".join(textos)

    qtde_linhas_texto = len(texto_processado_unico.split("\n"))
    height_linha = height_texto / qtde_linhas_texto if qtde_linhas_texto > 0 else 0

    qtde_linhas_brace = [
        math.ceil(h / height_linha if height_linha > 0 else 0) for h in height_brace
    ]
    linha_inicio_brace = [
        int(round(((top - top_texto) / height_linha if height_linha > 0 else 0), 0))
        for top in top_brace
    ]

    linhas = texto_processado_unico.split("\n")

    for linha_inicio, linhas_a_marcar in zip(linha_inicio_brace, qtde_linhas_brace):
        for idx, line in enumerate(linhas):
            if idx >= linha_inicio and linhas_a_marcar > 0:
                if bis_tag not in line:
                    linhas[idx] = line + bis_tag
                linhas_a_marcar -= 1

    texto_processado_unico = "\n".join(linhas)

    return {
        "shape": "AUTO_SHAPE",
        "auto_shape": "RECTANGLE",
        "text_processado": texto_processado_unico,
        "text": texto_unico,
    }


def set_text_full(texts: List[str]) -> str:
    texts_full = [line for line in texts if line.strip() != ""]
    texts_full = "\n\n".join(texts_full)
    texts_full = texts_full.replace('"', "'")
    texts_full = re.sub(r"[^\S\r\n]+", " ", texts_full).strip()
    return texts_full


def return_number(title: Optional[str]) -> str:
    if title:
        for word in title.split(" "):
            m = re.search(r"^\d+", word)
            if m:
                return m.group()
    return "null"


def list2json(all_praises: List[Dict]) -> List[Dict]:
    """Apply tagging and produce the final list of hymns (new_structure).

    Returns new_structure list ready for JSON serialization.
    """
    LOGGER.info("Applying tags and building final structure")
    apply_tags_to_all_praises(all_praises)

    # extract titles from first slide
    for praise in all_praises:
        first_slide_shapes = (
            praise.get("slides", [])[0].get("shapes", [])
            if praise.get("slides")
            else []
        )
        first_slide_texts = list(get_texts(first_slide_shapes, "text_processado"))
        title, unchanged_title = return_possible_title(first_slide_texts)
        LOGGER.debug("Title: %s, raw: %s", title, unchanged_title)
        praise.setdefault("title", title)
        praise["slides"][0]["shapes"] = [
            s for s in first_slide_shapes if s.get("text") != unchanged_title
        ]

    bis_pattern = re.compile(r"^<bis(\s+modo=\".*\")?>\s*(\n|$)")
    for praise in all_praises:
        for slide in praise.get("slides", []):
            merged = build_merged_shape_for_bis(slide, bis_pattern)
            if merged is not None:
                slide["shapes"] = [merged]

    new_structure: List[Dict] = []
    for praise in all_praises:
        texts_processado = list(get_texts(praise, "text_processado"))
        texts = list(get_texts(praise, "text"))
        if not texts_processado:
            continue
        titulo = praise.get("title", "null")
        numero = "null"
        m = re.match(r"^(\d+)\s+-\s+", titulo or "")
        if m:
            numero = m.group(1)
            titulo = titulo[len(numero) + 3 :].strip()

        texts_processado_full = set_text_full(texts_processado)
        texts_full = set_text_full(texts)

        new_structure.append(
            {
                "numero": numero,
                "nome": titulo or "null",
                "texto_processado": texts_processado_full,
                "texto": texts_full,
            }
        )

    return new_structure


def write_json(data: List[Dict], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    LOGGER.info("Wrote JSON to %s", p)


def process_file(file_txt: str, out_dir: str = "slides_json") -> Optional[Path]:
    """Parse a single TXT file and write a JSON file with same stem into out_dir.

    Returns the path written or None on error.
    """
    try:
        praises = parse_txt_file(file_txt)
        new_structure = list2json(praises)
        out_path = Path(out_dir) / (Path(file_txt).stem + ".json")
        write_json(new_structure, str(out_path))
        return out_path
    except Exception:
        LOGGER.exception("Error processing %s", file_txt)
        return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = ArgumentParser(description="Convert slides_txt to structured JSON list")
    parser.add_argument(
        "--src", default="slides_txt", help="Source folder for txt files"
    )
    parser.add_argument(
        "--out-dir",
        default="slides_json",
        help="Output directory for per-TXT JSON files",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.quiet:
        LOGGER.setLevel(logging.WARNING)

    files = find_txt_files(args.src)
    LOGGER.info("Processing %d files to %s", len(files), args.out_dir)
    written = []
    for f in files:
        p = process_file(f, args.out_dir)
        if p is not None:
            written.append(str(p))
    print(json.dumps({"written_files": written}, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
