from flask import Flask, request, send_file, jsonify
import os
import sqlite3
import numpy as np
from dotenv import load_dotenv
from PIL import Image
import openslide
from io import BytesIO
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
set_db_size_limit('image_db_new.db')

app = Flask(__name__)
DB_PATH = 'image_db_new.db'  # Your local SQLite DB
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
        conn = sqlite3.connect('image_db_new.db')
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
from werkzeug.security import check_password_hash, generate_password_hash

@app.route('/check-user-login', methods=['POST'])
def check_user_login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'success': False, 'error': 'Missing credentials'}), 400

    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        conn.close()

        if row and check_password_hash(row[0], password):
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Invalid credentials'}), 401

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/register-user', methods=['POST'])
def register_user():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'success': False, 'error': 'Missing fields'}), 400

    try:
        hashed_pw = generate_password_hash(password)

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username, password, verified) VALUES (?, ?, 0)", (username, hashed_pw))
        conn.commit()
        conn.close()

        return jsonify({'success': True})
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'error': 'Username already exists'}), 409
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/upload-slide', methods=['POST'])
def upload_slide():
    try:
        user_email = request.form.get('user_email')
        slide_file = request.files.get('file')

        if not user_email or not slide_file:
            return "Missing data", 400

        # Check for supported file type
        filename = slide_file.filename
        if not filename.lower().endswith(('.svs', '.tif', '.tiff')):
            return "Invalid slide format", 400

        # Save slide to temp directory
        temp_path = os.path.join("temp_slides", filename)
        os.makedirs("temp_slides", exist_ok=True)
        slide_file.save(temp_path)

        # Generate thumbnail using OpenSlide
        slide = openslide.open_slide(temp_path)
        thumbnail = slide.get_thumbnail((2048, 2048))

        # Save thumbnail to bytes
        img_bytes = BytesIO()
        thumbnail.save(img_bytes, format='PNG')
        img_data = img_bytes.getvalue()

        # Save thumbnail to DB, 1 per user
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO user_images (user_email, filename, image_data)
            VALUES (?, ?, ?)
            ON CONFLICT(user_email) DO UPDATE SET
                filename=excluded.filename,
                image_data=excluded.image_data,
                uploaded_at=CURRENT_TIMESTAMP
        ''', (user_email, filename, img_data))
        conn.commit()
        conn.close()

        return jsonify({"status": "success", "message": "Slide uploaded and thumbnail saved."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/check-user-verified', methods=['GET'])
def check_user_verified():
    email = request.args.get('email')
    if not email:
        return jsonify({'error': 'Missing email'}), 400

    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT verified FROM users WHERE username = ?", (email,))
        row = cur.fetchone()
        conn.close()

        if row and row[0] == 1:
            return jsonify({'verified': True})
        else:
            return jsonify({'verified': False})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/verify-user', methods=['POST'])
def verify_user():
    data = request.json
    email = data.get('email')

    if not email:
        return jsonify({'error': 'Missing email'}), 400

    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("UPDATE users SET verified = 1 WHERE username = ?", (email,))
        conn.commit()
        conn.close()
        return jsonify({'status': 'verified'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-user-image-name')
def get_user_image_name():
    user_email = request.args.get('email')
    if not user_email:
        return "Missing email", 400

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT filename FROM user_images WHERE user_email = ?", (user_email,))
    row = cur.fetchone()
    conn.close()

    if row:
        return jsonify({'filename': row[0]})
    return "No image found", 404

if __name__ == '__main__':
    os.makedirs(png_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=8080)
