import sqlite3
from pathlib import Path

# Fix for McKinsey PRO V4.2+: Migrating existing SQLite schema
db_path = Path("data/analytics.db")
if db_path.exists():
    print(f"Migrating database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check current columns
    cursor.execute("PRAGMA table_info(conversations)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if "transcript" not in columns:
        print("Adding 'transcript' column...")
        cursor.execute("ALTER TABLE conversations ADD COLUMN transcript TEXT")
        print("Migration complete.")
    else:
        print("'transcript' column already exists.")
        
    conn.commit()
    conn.close()
else:
    print("No database found at data/analytics.db - it will be created with the new schema on next run.")
