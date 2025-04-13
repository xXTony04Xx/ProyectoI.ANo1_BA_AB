import pandas as pd
import re
import string
import pickle
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import download
from pathlib import Path
from nltk.tokenize import word_tokenize
from app.naive_bayes import NaiveBayes

# Descargar recursos de NLTK si no existen
download('punkt')
download('stopwords')

# === CONFIGURACIÓN ===
DATASET_PATH = Path(__file__).resolve().parent.parent / 'static' / 'sentiment140.csv'
MODEL_PATH = Path(__file__).resolve().parent / 'naive_bayes_model.pkl'

# === PREPROCESAMIENTO ===
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(f"[{re.escape(string.punctuation)}]", "", texto)
    tokens = word_tokenize(texto, preserve_line=True)  # <- AÑADIDO preserve_line
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    return tokens

# === CARGAR DATASET ===
def cargar_datos():
    df = pd.read_csv(DATASET_PATH, encoding='latin-1', header=None)

    # Mostrar cuántas columnas tiene el archivo
    print(f"Columnas detectadas en el archivo: {df.shape[1]}")
    print(df.head())

    if df.shape[1] >= 6:
        df = df[[0, 5]]
        df.columns = ['label', 'text']
    elif df.shape[1] == 2:
        df.columns = ['label', 'text']
    else:
        raise ValueError("❌ El archivo no tiene el formato esperado. Asegúrate de que contenga columnas para 'label' y 'text'.")

    df['label'] = df['label'].map({0: 'neg', 2: 'neu', 4: 'pos'})
    return df

# === ENTRENAMIENTO DE NAIVE BAYES ===
class NaiveBayes:
    def __init__(self):
        self.clases = ['pos', 'neg', 'neu']
        self.vocabulario = set()
        self.prior = {}
        self.freq_palabra = {}
        self.total_palabras_clase = {}

    def entrenar(self, X, y):
        self.freq_palabra = {c: defaultdict(int) for c in self.clases}
        self.total_palabras_clase = {c: 0 for c in self.clases}
        self.prior = {c: 0 for c in self.clases}

        total_docs = len(y)
        for texto, etiqueta in zip(X, y):
            self.prior[etiqueta] += 1
            for palabra in texto:
                self.freq_palabra[etiqueta][palabra] += 1
                self.total_palabras_clase[etiqueta] += 1
                self.vocabulario.add(palabra)

        for c in self.clases:
            self.prior[c] /= total_docs

    def predecir(self, tokens):
        resultados = {}
        V = len(self.vocabulario)
        for c in self.clases:
            log_prob = 0
            for palabra in tokens:
                frecuencia = self.freq_palabra[c][palabra]
                prob = (frecuencia + 1) / (self.total_palabras_clase[c] + V)
                log_prob += pd.np.log(prob)
            resultados[c] = pd.np.log(self.prior[c]) + log_prob
        return max(resultados, key=resultados.get)

# === ENTRENAMIENTO ===
def main():
    print("Cargando datos...")
    df = cargar_datos()

    print("Preprocesando...")
    df['tokens'] = df['text'].apply(limpiar_texto)

    print("Entrenando modelo...")
    nb = NaiveBayes()
    nb.entrenar(df['tokens'], df['label'])

    print(f"Guardando modelo en {MODEL_PATH}")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(nb, f)

    print("✅ Entrenamiento completado.")

if __name__ == "__main__":
    main()
