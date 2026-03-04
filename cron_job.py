import os
import sys
from pathlib import Path

# Add project root to sys path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from the_layman.database.store import Store
from the_layman.pipeline.daily_feed import generate_daily_feed


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    db_path = base_dir / "cache" / "the_layman.db"
    
    print(f"Connecting to DB at: {db_path}")
    store = Store(db_path)
    
    # Run the generator
    generate_daily_feed(store)
