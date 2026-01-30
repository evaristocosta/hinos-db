import subprocess
import glob
import os
from pathlib import Path

DB_PATH = Path("..") / "database.db"
MIGRATIONS_PATH = Path("..") / "migrations"


def run_migrations():
    """Run SQLIte migrations using CMD."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    migration_files = glob.glob(str(MIGRATIONS_PATH / "*.sql"))
    for migration in migration_files:
        print(f"Running migration: {migration}")
        subprocess.run(f"sqlite3 {DB_PATH} < {migration}", shell=True)


if __name__ == "__main__":
    run_migrations()
