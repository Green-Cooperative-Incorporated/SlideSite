from flask import Flask, render_template, request, redirect, url_for, flash
import os
import requests
from flask import send_file, Response
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for flash messages
LOCAL_API = "http://localhost:5000"  # Replace with your ngrok URL when exposed
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

        if username == USER_CREDENTIALS['username'] and password == USER_CREDENTIALS['password']:
            return redirect(url_for('slidesite'))
        else:
            flash('Invalid credentials. Please try again.')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/slidesite')
def slidesite():
    return render_template('slidesite.html')




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
