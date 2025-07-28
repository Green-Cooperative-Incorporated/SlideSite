from flask import Flask, request, send_file, jsonify
import os
import sqlite3
import numpy as np

app = Flask(__name__)
DB_PATH = 'database.db'  # Your local SQLite DB
png_FOLDER = 'png_files'  # Folder where your .png files live

@app.route('/get-png')
def get_png():
    filename = request.args.get('filename')
    if not filename or not filename.endswith('.png'):
        return "Invalid filename", 400
    
    filepath = os.path.join(png_FOLDER, filename)
    if not os.path.exists(filepath):
        return "File not found", 404
    
    return send_file(filepath, as_attachment=True)

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
