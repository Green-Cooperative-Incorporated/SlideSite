from flask import Flask, render_template, request, redirect, url_for, flash
import os
import requests
from flask import send_file, Response
from werkzeug.security import check_password_hash
import sqlite3

import json
from flask_mail import Mail, Message

from itsdangerous import URLSafeTimedSerializer
from dotenv import load_dotenv
from werkzeug.security import check_password_hash

from werkzeug.security import generate_password_hash
from flask import render_template, request, redirect, url_for, flash
from flask_mail import Message
from werkzeug.security import generate_password_hash
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
# Token generation and confirmation
def get_serializer():
    return URLSafeTimedSerializer(app.secret_key)

def generate_verification_token(email):
    serializer = get_serializer()
    return serializer.dumps(email, salt='email-confirmation-salt')

def confirm_token(token, expiration=3600):  # 1 hour
    serializer = get_serializer()
    try:
        return serializer.loads(token, salt='email-confirmation-salt', max_age=expiration)
    except Exception:
        return None

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'greencooperativeinc@gmail.com'
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = 'greencooperativeinc@gmail.com'

mail = Mail(app)


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


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        hashed_pw = generate_password_hash(password)

        conn = sqlite3.connect('database_new.db')
        cur = conn.cursor()
        try:
            # insert with verified = 0
            cur.execute("INSERT INTO users (username, password, verified) VALUES (?, ?, 0)", (username, hashed_pw))
            conn.commit()

            # generate email verification token
            token = generate_verification_token(username)
            confirm_url = url_for('confirm_email', token=token, _external=True)

            # send email
            msg = Message("Confirm Your Account", recipients=[username])
            msg.body = f"Hi, please verify your account by clicking this link:\n\n{confirm_url}"
            mail.send(msg)

            flash("Account created! Check your email to verify.")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username/email already exists.")
            return redirect(url_for('register'))
        finally:
            conn.close()

    return render_template('register.html')

@app.route('/confirm/<token>')
def confirm_email(token):
    email = confirm_token(token)
    if not email:
        return render_template('confirm_email.html', success=False)

    conn = sqlite3.connect('database_new.db')
    cur = conn.cursor()
    cur.execute("UPDATE users SET verified = 1 WHERE username = ?", (email,))
    conn.commit()
    conn.close()

    return render_template('confirm_email.html', success=True)


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
