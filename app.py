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
def landing():
    return render_template('landing.html')




@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        try:
            res = requests.post(f"{LOCAL_API}/check-user-login", json={
                "username": username,
                "password": password
            })

            if res.status_code == 200 and res.json().get('success'):
                session['user_email'] = username
                return redirect(url_for('slidesite'))
            else:
                flash(res.json().get('error', 'Login failed.'))
        except Exception as e:
            flash(f"Error contacting login API: {e}")
        return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/slidesite')
def slidesite():
    upload_success = request.args.get('upload_success', 'false').lower() == 'true'
    return render_template('slidesite.html', upload_success=upload_success)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']

        try:
            res = requests.post(f"{LOCAL_API}/register-user", json={
                "username": username,
                "password": password
            })

            if res.status_code == 200 and res.json().get('success'):
                # Generate email token and send mail
                token = generate_verification_token(username)
                confirm_url = url_for('confirm_email', token=token, _external=True)

                msg = Message("Confirm Your Account", recipients=[username])
                msg.body = f"Hi, please verify your account:\n\n{confirm_url}"
                mail.send(msg)

                flash("Account created! Check your email to verify.")
                return redirect(url_for('login'))
            else:
                flash(res.json().get('error', 'Registration failed.'))

        except Exception as e:
            flash(f"Error contacting registration API: {e}")
        return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/confirm/<token>')
def confirm_email(token):
    email = confirm_token(token)
    if not email:
        return render_template('confirm_email.html', success=False)

    # ✅ Make sure this updates the right table (local DB)
    try:
        res = requests.post(f"{LOCAL_API}/verify-user", json={"email": email})
        if res.status_code == 200:
            return render_template('confirm_email.html', success=True)
        else:
            return render_template('confirm_email.html', success=False)
    except Exception as e:
        return f"Verification failed: {str(e)}", 500

@app.route('/my-image')
def my_image():
    user_email = session.get('user_email')
    if not user_email:
        return "Unauthorized", 401

    try:
        res = requests.get(f"{LOCAL_API}/get-user-image-name", params={'email': user_email})
        if res.status_code == 200:
            filename = res.json().get('filename')
            if filename:
                return redirect(url_for('download_image', filename=filename))
        return "No image found.", 404
    except Exception as e:
        return f"Error: {str(e)}", 500

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


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    user_email = session.get('user_email')
    if not user_email:
        return "Unauthorized", 401

    if request.method == 'POST':
        file = request.files.get('file')

        if not file or not file.filename.lower().endswith(('.svs', '.tif', '.tiff')):
            flash("Please upload a valid .svs or .tif file.")
            return render_template('upload.html', upload_success=False)

        filename = secure_filename(file.filename)

        try:
            res = requests.post(
                f"{LOCAL_API}/upload-slide",
                data={'user_email': user_email},
                files={'file': (filename, file.stream, file.mimetype)}
            )

            if res.status_code == 200:
                flash("Slide uploaded and thumbnail saved.")

                # ✅ Send confirmation email
                try:
                    msg = Message(
                        "SlideSite - Thumbnail Ready",
                        recipients=[user_email]
                    )
                    msg.body = f"""Hi,

Your slide ({filename}) has been successfully uploaded, and your thumbnail has been generated.

You can download it from your dashboard here:
{url_for('download_image', _external=True)}

Thank you for using SlideSite!
"""
                    mail.send(msg)
                except Exception as e:
                    flash(f"Upload succeeded, but email failed to send: {e}")

                return "Upload complete", 200  
            else:
                flash(f"Upload failed: {res.text}")
        except Exception as e:
            flash(f"Error sending file to local API: {e}")

        return render_template('upload.html', upload_success=False)

    return render_template('upload.html', upload_success=False)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
