from flask import Flask, render_template, request, jsonify
import pickle
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download
from pathlib import Path
from naive_bayes import NaiveBayes

# Descargar recursos NLTK si es necesario
download('punkt')
download('stopwords')

# === CONFIGURACIÓN Y CARGA DEL MODELO ===
MODEL_PATH = Path(__file__).resolve().parent / 'model' / 'naive_bayes_model.pkl'

# Cargar modelo entrenado
with open(MODEL_PATH, 'rb') as f:
    modelo = pickle.load(f)

# === PREPROCESAMIENTO ===
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(f"[{re.escape(string.punctuation)}]", "", texto)
    tokens = word_tokenize(texto)
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    return tokens

# === FUNCIÓN DE INFERENCIA ===
def predict_text(text):
    tokens = limpiar_texto(text)
    resultado = modelo.predecir(tokens)
    if resultado == 'pos':
        return 'Positivo'
    elif resultado == 'neg':
        return 'Negativo'
    elif resultado == 'neu':
        return 'Neutro'
    return 'Desconocido'

# === FLASK APP ===
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    texto = request.form['texto']
    resultado = predict_text(texto)
    return jsonify({'resultado': resultado})

if __name__ == '__main__':
    app.run(debug=True)
