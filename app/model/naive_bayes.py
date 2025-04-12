import pickle
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download
from pathlib import Path

# Descargar recursos si no están disponibles
download('punkt')
download('stopwords')

# Ruta al modelo
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / 'naive_bayes_model.pkl'

# Cargar modelo
with open(MODEL_PATH, 'rb') as f:
    modelo = pickle.load(f)

# Preprocesamiento igual al usado en entrenamiento
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(f"[{re.escape(string.punctuation)}]", "", texto)
    tokens = word_tokenize(texto)
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    return tokens

# Función de inferencia
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
