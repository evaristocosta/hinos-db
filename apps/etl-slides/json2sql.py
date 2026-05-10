"""Convert JSON hymn files into SQL migration files.

Each input JSON file produces one SQL file in the output directory.
"""

from argparse import ArgumentParser
from glob import glob
import json
import logging
from pathlib import Path
from typing import Iterable, List

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


INSERT_COLUMNS = "numero, nome, nome_pt, texto, texto_processado, coletanea_id, idioma, date_insert, date_update"


def find_json_files(input_pattern: str) -> List[Path]:
    return [Path(path) for path in glob(input_pattern)]


def load_json_file(file_path: Path) -> list:
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def escape_sql_literal(value: str) -> str:
    return value.replace("'", "''").replace("\n", "\\n")


def format_numero(value) -> str:
    if value is None:
        return "NULL"
    value_str = str(value).strip()
    if value_str.isdigit():
        return int(value_str)
    return "NULL"


def build_insert_statement(hino: dict, coletanea_id: int) -> str:
    numero = format_numero(hino.get("numero"))
    nome = escape_sql_literal(str(hino.get("nome", "")))
    texto = escape_sql_literal(str(hino.get("texto", "")))
    texto_processado = escape_sql_literal(str(hino.get("texto_processado", "")))

    return (
        f"INSERT INTO hino ({INSERT_COLUMNS}) VALUES ("
        f"{numero}, "
        f"'{nome}', "
        f"'{nome}', "
        f"'{texto}', "
        f"'{texto_processado}', "
        f"{coletanea_id}, "
        f"'PT-BR', "
        f"CURRENT_TIMESTAMP, CURRENT_TIMESTAMP"
        f");\n"
    )


def build_output_name(file_path: Path, inicio: int, index: int) -> str:
    prefix = str(index + inicio).zfill(3)
    stem = file_path.stem.lower().replace(" ", "-")
    return f"{prefix}-add-{stem}.sql"


def write_sql_file(
    output_path: Path, file_name: str, louvores: list, coletanea_id: int
) -> Path:
    output_path.mkdir(parents=True, exist_ok=True)
    destination = output_path / file_name

    with destination.open("w", encoding="utf-8") as handle:
        for hino in louvores:
            handle.write(build_insert_statement(hino, coletanea_id))

    return destination


def json2sql(
    inicio: int = 3,
    input_pattern: str = "slides_json\\*.json",
    output_path: str = "..\\..\\database\\migrations",
) -> List[Path]:
    LOGGER.info("Starting json2sql conversion")
    files_json = find_json_files(input_pattern)
    LOGGER.info("Files found: %s", [str(path) for path in files_json])

    written_files: List[Path] = []
    output_dir = Path(output_path)

    for index, file_path in enumerate(files_json):
        file_name = build_output_name(file_path, inicio, index)
        louvores = load_json_file(file_path)
        written_files.append(write_sql_file(output_dir, file_name, louvores, index + 1))
        LOGGER.info("Wrote %s", written_files[-1])

    return written_files


def main(argv: Iterable[str] | None = None) -> int:
    parser = ArgumentParser(description="Convert JSON hymn files to SQL migrations")
    parser.add_argument(
        "--inicio",
        type=int,
        default=3,
        help="Starting sequence number for migration files",
    )
    parser.add_argument(
        "--input-pattern",
        default="slides_json\\*.json",
        help="Glob pattern for input JSON files",
    )
    parser.add_argument(
        "--output-path",
        default="..\\..\\database\\migrations",
        help="Directory for generated SQL files",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    json2sql(
        inicio=args.inicio,
        input_pattern=args.input_pattern,
        output_path=args.output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
