from flask import Flask, render_template, request, redirect, url_for, flash
import os
import requests
from flask import send_file, Response
from werkzeug.security import check_password_hash
import sqlite3
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for flash messages
import json


def load_local_api_url():
    try:
        with open(os.path.join(os.path.dirname(__file__), 'ngrok_url.json')) as f:
            return json.load(f)["ngrok_url"]
    except:
        return "https://fallback-url.com"

LOCAL_API = load_local_api_url()

# Dummy user for demonstration
USER_CREDENTIALS = {
    'username': 'admin',
    'password': 'password123'
}

@app.route('/')
def home():
    return redirect(url_for('login'))

from werkzeug.security import check_password_hash

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('database_new.db')
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        conn.close()

        if row and check_password_hash(row[0], password):
            return redirect(url_for('slidesite'))
        else:
            flash('Invalid credentials. Please try again.')
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/slidesite')
def slidesite():
    return render_template('slidesite.html')

from werkzeug.security import generate_password_hash
from flask import render_template, request, redirect, url_for, flash

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']

        if not username or not password:
            flash("Both fields are required.")
            return redirect(url_for('register'))

        hashed_pw = generate_password_hash(password)

        conn = sqlite3.connect('database_new.db')
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
            conn.commit()
            flash("Account created successfully! You can now log in.")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already exists.")
            return redirect(url_for('register'))
        finally:
            conn.close()

    return render_template('register.html')



@app.route('/download/<filename>')
def download_image(filename):
    try:
        # Fetch file from local API
        response = requests.get(f'{LOCAL_API}/get-png', params={'filename': filename}, stream=True)

        if response.status_code != 200:
            return f"Error fetching file: {response.text}", response.status_code

        # Relay the file directly to the user
        return Response(
            response.iter_content(chunk_size=8192),
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "image/png"
            },
            direct_passthrough=True
        )

    except Exception as e:
        return f"Download failed: {str(e)}", 500
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
