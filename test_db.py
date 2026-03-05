import urllib.parse
from the_layman.database.store import Store
import os

os.environ["DATABASE_URL"] = "postgresql://neondb_owner:password123@ep-cool-butterfly-a5z3z.us-east-2.aws.neon.tech/neondb?sslmode=require"
try:
    s = Store()
    print("Store initialized")
except Exception as e:
    print(f"Error: {e}")
