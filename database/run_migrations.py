import subprocess
import glob
import os


def run_migrations():
    """Run SQLIte migrations using CMD."""
    # remove old database file if it exists
    if os.path.exists("database.db"):
        os.remove("database.db")

    migration_files = glob.glob("migrations/*.sql")
    for migration in migration_files:
        print(f"Running migration: {migration}")
        subprocess.run(f"sqlite3 database.db < {migration}", shell=True)


if __name__ == "__main__":
    run_migrations()
