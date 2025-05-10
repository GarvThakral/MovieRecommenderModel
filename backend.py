from flask import Flask, jsonify, request
from model import recommend
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    movieName = request.args.get('query') 
    print(movieName, "heres the name")
    return jsonify(message=recommend(movieName))

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))  # Default to 5000 locally
    app.run(host='0.0.0.0', port=port, debug=False)  # Bind to 0.0.0.0, disable debug