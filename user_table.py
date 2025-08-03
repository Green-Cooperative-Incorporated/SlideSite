import sqlite3

conn = sqlite3.connect(r'C:\Users\carso\Documents\GitHub\SlideSite\image_db_new.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')
cursor.execute('''ALTER TABLE users ADD COLUMN verified BOOLEAN DEFAULT 0;
 ''')
conn.commit()
conn.close()
