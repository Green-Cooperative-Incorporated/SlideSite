from flask import Flask, jsonify
from flask_cors import CORS
import os
app = Flask(__name__)
CORS(app)  # Allows frontend on a different port to access this backend

@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify(message="Hello World")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # fallback to 5000 locally
    app.run(host='0.0.0.0', port=port, debug=True)
