from flask import Flask, request, send_file, jsonify
import os
import sqlite3
import numpy as np
from dotenv import load_dotenv

load_dotenv()
SECRET_KEY = os.getenv('SECRET_KEY')  # Loaded from your local .env

@app.route('/get-secret-key')
def get_secret_key():
    token = request.args.get('auth')  # simple auth
    if token != os.getenv("SECRET_KEY"):
        return "Unauthorized", 401

    return jsonify({'SECRET_KEY': SECRET_KEY})

app = Flask(__name__)
DB_PATH = 'database_new.db'  # Your local SQLite DB
png_FOLDER = 'png_files'  # Folder where your .png files live
from flask import send_file
from io import BytesIO

@app.route('/get-png')
def get_png():
    filename = request.args.get('filename')
    if not filename or not filename.endswith('.png'):
        return "Invalid filename", 400

    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT image_data FROM png_files WHERE filename = ?", (filename,))
        row = cur.fetchone()
        conn.close()

        if not row:
            return "File not found in database", 404

        # Convert BLOB to file-like object and send
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

if __name__ == '__main__':
    os.makedirs(png_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=8080)
