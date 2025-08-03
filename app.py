from flask import Flask, render_template, request, redirect, url_for, flash
import os
import requests
from flask import send_file, Response
from werkzeug.security import check_password_hash
import sqlite3

import json
from flask_mail import Mail, Message

from itsdangerous import URLSafeTimedSerializer

from werkzeug.security import check_password_hash

from werkzeug.security import generate_password_hash
from flask import render_template, request, redirect, url_for, flash
from flask_mail import Message
from werkzeug.security import generate_password_hash
from werkzeug.utils import secure_filename

app = Flask(__name__)
from flask import session
def load_local_api_url():
    try:
        with open(os.path.join(os.path.dirname(__file__), 'ngrok_url.json')) as f:
            return json.load(f)["ngrok_url"]
    except:
        return "https://fallback-url.com"

LOCAL_API = load_local_api_url()
def fetch_secret_key():
    try:
        res = requests.get(
            f"{LOCAL_API}/get-secret-key",
            params={'auth': 'super_secure_token_123'}
        )
        if res.status_code == 200:
            return res.json()['SECRET_KEY']
        else:
            print("Failed to fetch SECRET_KEY:", res.status_code)
    except Exception as e:
        print("Error fetching SECRET_KEY:", e)
    return 'fallback-dev-secret'

def fetch_mail_password():
    try:
        res = requests.get(
            f"{LOCAL_API}/get-mail-password",
            params={'auth': 'super_secure_token_123'}
        )
        if res.status_code == 200:
            return res.json().get('MAIL_PASSWORD')
        else:
            print("Failed to fetch MAIL_PASSWORD:", res.status_code)
    except Exception as e:
        print("Error fetching MAIL_PASSWORD:", e)
    return None
app.secret_key = fetch_secret_key()

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
app.config['MAIL_PASSWORD'] = fetch_mail_password() or 'fallback-password'
app.config['MAIL_DEFAULT_SENDER'] = 'greencooperativeinc@gmail.com'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB

mail = Mail(app)




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
            session['user_email'] = username  # track logged-in user
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

    user_email = session.get('user_email')
    if not user_email:
        return "Unauthorized", 401

    try:
        response = requests.get(
            f'{LOCAL_API}/get-png',
            params={'filename': filename, 'email': user_email},
            stream=True
        )

        if response.status_code != 200:
            return f"Error fetching file: {response.text}", response.status_code

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
from flask import session

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    user_email = session.get('user_email')
    if not user_email:
        return "Unauthorized", 401

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not file.filename.lower().endswith('.png'):
            flash("Please upload a valid PNG file.")
            return redirect(url_for('upload'))

        filename = secure_filename(file.filename)

        # Send file to local API
        try:
            res = requests.post(
                f"{LOCAL_API}/store-user-image",
                data={
                    'user_email': user_email,
                    'filename': filename
                },
                files={'file': (filename, file.stream, file.mimetype)}
            )
            if res.status_code == 200:
                flash("File uploaded successfully.")
            else:
                flash(f"Upload failed: {res.text}")
        except Exception as e:
            flash(f"Error connecting to local API: {e}")

        return redirect(url_for('upload'))

    return render_template('upload.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
