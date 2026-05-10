import argparse
import subprocess
from pathlib import Path

DB_PATH = Path("..") / "database.db"
MIGRATIONS_PATH = Path("..") / "migrations"


def run_migrations(
    db_path: Path = DB_PATH, migrations_path: Path = MIGRATIONS_PATH
) -> None:
    """Run SQLIte migrations using CMD."""
    if db_path.exists():
        db_path.unlink()

    migration_files = sorted(migrations_path.glob("*.sql"))
    for migration in migration_files:
        print(f"Running migration: {migration}")
        subprocess.run(f"sqlite3 {db_path} < {migration}", shell=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SQLite migrations")
    parser.add_argument(
        "--db-path", default=str(DB_PATH), help="Path to SQLite DB file"
    )
    parser.add_argument(
        "--migrations-path",
        default=str(MIGRATIONS_PATH),
        help="Path to migrations directory",
    )
    args = parser.parse_args()

    run_migrations(
        db_path=Path(args.db_path), migrations_path=Path(args.migrations_path)
    )


if __name__ == "__main__":
    main()
