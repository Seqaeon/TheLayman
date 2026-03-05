import os
from pathlib import Path
db_url = os.environ.get("DATABASE_URL")
is_postgres = db_url and db_url.startswith("postgres")
print(f"URL: {db_url}, is_postgres: {is_postgres}")
