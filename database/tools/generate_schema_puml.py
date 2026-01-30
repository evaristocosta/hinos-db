# generate_db_schema_puml.py
import sqlite3
from pathlib import Path
from typing import Dict, List

DB_PATH = Path("..") / "database.db"
OUTPUT = Path("..") / "schema" / "db-schema.puml"


def get_tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    """
    )
    return [r[0] for r in cur.fetchall()]


def get_columns(conn: sqlite3.Connection, table: str):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info('{table}')")
    cols = []
    for cid, name, col_type, notnull, dflt_value, pk in cur.fetchall():
        cols.append(
            {
                "name": name,
                "type": col_type or "",
                "notnull": bool(notnull),
                "pk": bool(pk),
                "default": dflt_value,
            }
        )
    return cols


def get_foreign_keys(conn: sqlite3.Connection, table: str):
    cur = conn.cursor()
    cur.execute(f"PRAGMA foreign_key_list('{table}')")
    fks = []
    for (
        _id,
        seq,
        ref_table,
        from_col,
        to_col,
        on_update,
        on_delete,
        match,
    ) in cur.fetchall():
        fks.append(
            {
                "from_table": table,
                "from_col": from_col,
                "to_table": ref_table,
                "to_col": to_col,
            }
        )
    return fks


def generate_puml(tables_info, foreign_keys) -> str:
    lines = []
    lines.append("@startuml")
    lines.append("hide circle")
    lines.append("skinparam class {")
    lines.append("  BackgroundColor White")
    lines.append("  BorderColor Black")
    lines.append("}")
    lines.append("")

    # Entidades
    for table, cols in tables_info.items():
        lines.append(f"class {table} {{")
        for col in cols:
            flags = []
            if col["pk"]:
                flags.append("PK")
            if col["notnull"] and not col["pk"]:
                flags.append("NN")
            flag_str = f" <<{','.join(flags)}>>" if flags else ""
            col_type = f" : {col['type']}" if col["type"] else ""
            lines.append(f"  {col['name']}{col_type}{flag_str}")
        lines.append("}")
        lines.append("")

    # Relacionamentos (FK)
    # Exemplo: Order }o--|| Customer : "customer_id"
    for fk in foreign_keys:
        from_t = fk["from_table"]
        to_t = fk["to_table"]
        label = fk["from_col"]
        lines.append(f"{from_t} }}o--|| {to_t} : {label}")

    lines.append("")
    lines.append("@enduml")
    return "\n".join(lines)


def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        tables = get_tables(conn)
        tables_info: Dict[str, List[Dict]] = {}
        foreign_keys: List[Dict] = []
        for t in tables:
            tables_info[t] = get_columns(conn, t)
            foreign_keys.extend(get_foreign_keys(conn, t))

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        puml = generate_puml(tables_info, foreign_keys)
        OUTPUT.write_text(puml, encoding="utf-8")
        print(f"PlantUML ERD salvo em {OUTPUT}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
