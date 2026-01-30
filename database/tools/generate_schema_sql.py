# generate_schema_sql.py
import sqlite3
from pathlib import Path

DB_PATH = Path("..") / "database.db"
OUTPUT = Path("..") / "schema" / "db-schema.sql"


def dump_schema(db_path: Path) -> str:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT type, name, tbl_name, sql
            FROM sqlite_master
            WHERE sql IS NOT NULL
              AND name NOT LIKE 'sqlite_%'
            ORDER BY type, name
        """
        )
        statements = []
        for row in cur.fetchall():
            sql = row["sql"].strip()
            # Garante que termina com ponto e vírgula
            if not sql.endswith(";"):
                sql += ";"
            statements.append(sql)
        return "\n\n".join(statements) + "\n"
    finally:
        conn.close()


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    schema_sql = dump_schema(DB_PATH)
    OUTPUT.write_text(schema_sql, encoding="utf-8")
    print(f"Schema salvo em {OUTPUT}")


if __name__ == "__main__":
    main()
