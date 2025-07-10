from flask import Flask, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return 'Welcome to SlideSite Backend!'

@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify(message="Hello World")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
