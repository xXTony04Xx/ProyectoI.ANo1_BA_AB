from flask import Flask, render_template, request, jsonify
import pickle
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from model.train import cargar_datos
from naiveBayes import NaiveBayes


# Descargar recursos NLTK
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

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

def evaluar_modelo(df, porcentaje_test=0.2):
    print(f"\nEvaluando modelo con {porcentaje_test*100:.0f}% de los datos como test...\n")

    df['tokens'] = df['text'].apply(limpiar_texto)
    X_train, X_test, y_train, y_test = train_test_split(df['tokens'], df['label'], test_size=porcentaje_test, random_state=42)

    nb = NaiveBayes()
    nb.entrenar(X_train, y_train)

    y_pred = [nb.predecir(tokens) for tokens in X_test]
    print(classification_report(y_test, y_pred, digits=4))

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

@app.route('/evaluate', methods=['POST'])
def evaluate():
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    # Obtener porcentaje
    porcentaje = float(request.form['split'])
    
    # Cargar datos y evaluar
    df = cargar_datos()  # Asegúrate de importar esta función
    df['tokens'] = df['text'].apply(limpiar_texto)
    X_train, X_test, y_train, y_test = train_test_split(df['tokens'], df['label'], test_size=porcentaje, random_state=42)

    nb = NaiveBayes()
    nb.entrenar(X_train, y_train)
    y_pred = [nb.predecir(tokens) for tokens in X_test]

    report = classification_report(y_test, y_pred, digits=4)
    return jsonify({'report': report})

if __name__ == '__main__':
    app.run(debug=True)
    