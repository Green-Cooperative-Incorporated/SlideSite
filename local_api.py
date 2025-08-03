from flask import Flask, request, send_file, jsonify
import os
import sqlite3
import numpy as np
from dotenv import load_dotenv

load_dotenv()
SECRET_KEY = os.getenv('SECRET_KEY')  # Loaded from your local .env

def set_db_size_limit(db_path='database_new.db', max_gb=10):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    page_size = cur.execute("PRAGMA page_size").fetchone()[0]  # typically 4096
    max_pages = (max_gb * 1024 * 1024 * 1024) // page_size
    cur.execute(f"PRAGMA max_page_count = {max_pages}")
    conn.commit()
    conn.close()

# Call this once at startup
set_db_size_limit()
set_db_size_limit('image_database.db')

app = Flask(__name__)
DB_PATH = 'database_new.db'  # Your local SQLite DB
png_FOLDER = 'png_files'  # Folder where your .png files live
from flask import send_file
from io import BytesIO
@app.route('/get-secret-key')
def get_secret_key():
    token = request.args.get('auth')  # simple auth
    if token != os.getenv("API_TOKEN"):
        return "Unauthorized", 401

    return jsonify({'SECRET_KEY': SECRET_KEY})
@app.route('/get-mail-password')
def get_mail_password():
    token = request.args.get('auth')
    if token != os.getenv("API_TOKEN"):
        return "Unauthorized", 401

    return jsonify({'MAIL_PASSWORD': os.getenv('MAIL_PASSWORD')})
@app.route('/get-png')
def get_png():
    filename = request.args.get('filename')
    user_email = request.args.get('email')

    if not filename or not filename.endswith('.png') or not user_email:
        return "Invalid request", 400

    try:
        conn = sqlite3.connect('image_database.db')
        cur = conn.cursor()
        cur.execute(
            "SELECT image_data FROM user_images WHERE filename = ? AND user_email = ?",
            (filename, user_email)
        )
        row = cur.fetchone()
        conn.close()

        if not row:
            return "File not found for this user", 404

        image_blob = row[0]
        return send_file(
            BytesIO(image_blob),
            mimetype='image/png',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        return f"Server error: {str(e)}", 500


@app.route('/list-png')
def list_png():
    files = [f for f in os.listdir(png_FOLDER) if f.endswith('.png')]
    return jsonify(files)

@app.route('/query-db')
def query_db():
    query = request.args.get('query')
    if not query:
        return "Missing SQL query", 400
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(query)
        result = cur.fetchall()
        conn.close()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
from flask import request, jsonify

@app.route('/store-user-image', methods=['POST'])
@app.route('/store-user-image', methods=['POST'])
def store_user_image():
    try:
        data = request.form
        user_email = data.get('user_email')
        filename = data.get('filename')
        image_file = request.files.get('file')

        if not user_email or not filename or not image_file:
            return "Missing required fields", 400

        image_data = image_file.read()

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        # Overwrite if entry for user exists
        cur.execute('''
            INSERT INTO user_images (user_email, filename, image_data)
            VALUES (?, ?, ?)
            ON CONFLICT(user_email) DO UPDATE SET
                filename=excluded.filename,
                image_data=excluded.image_data,
                uploaded_at=CURRENT_TIMESTAMP
        ''', (user_email, filename, image_data))
        conn.commit()
        conn.close()

        return jsonify({"status": "success", "filename": filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    os.makedirs(png_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=8080)
