from flask import Flask, render_template, request, jsonify
from model.naive_bayes import predict_text

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['user_input']
    prediction = predict_text(text)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
