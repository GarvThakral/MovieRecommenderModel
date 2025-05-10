from flask import Flask ,jsonify,request
from model import recommend
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
@app.route('/')
def hello_world():
    movieName = request.args.get('query') 
    print(movieName,"heres the name")
    return jsonify(message = recommend(movieName))
if __name__ == '__main__':
    app.run(debug=True)
recommend("Batman Begins")