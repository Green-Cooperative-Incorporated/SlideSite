from flask import Flask, render_template, request, redirect, url_for, flash
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for flash messages

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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
