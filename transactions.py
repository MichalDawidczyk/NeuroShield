import sqlite3

def initialize_db():
    conn = sqlite3.connect("transactions.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            user_id TEXT PRIMARY KEY,
            location TEXT,
            time VARCHAR(20)
        )
    """)
    conn.commit()