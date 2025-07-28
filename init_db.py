import sqlite3
import os

# Directory where your PNG files are stored
PNG_FOLDER = r'c:\Users\carso\Documents\slide_images'  # or any folder on your system

# Connect to database (creates it if it doesn't exist)
conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS png_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        description TEXT,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')

# Loop through PNG files in the directory
for filename in os.listdir(PNG_FOLDER):
    if filename.lower().endswith('.png'):
        filepath = os.path.join(PNG_FOLDER, filename)

        # You can add smarter logic here to generate descriptions
        description = f"Auto-added file: {filename}"

        # Avoid duplicates by checking first (optional)
        cursor.execute('SELECT 1 FROM png_files WHERE filename = ?', (filename,))
        if cursor.fetchone() is None:
            cursor.execute('INSERT INTO png_files (filename, description) VALUES (?, ?)', (filename, description))
            print(f"Inserted: {filename}")
        else:
            print(f"Skipped (already exists): {filename}")

# Finalize DB changes
conn.commit()
conn.close()

print("All PNG files in the folder have been processed.")
