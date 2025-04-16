import pandas as pd
import re
import string
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import download
from pathlib import Path
from nltk.tokenize import word_tokenize
from naiveBayes import NaiveBayes

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
    tokens = word_tokenize(texto, preserve_line=True)
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    return tokens

# === CARGAR DATASET ===
def cargar_datos():
    df = pd.read_csv(DATASET_PATH, encoding='latin-1', header=None)

    if df.shape[1] >= 6:
        df = df[[0, 5]]
        df.columns = ['label', 'text']
    else:
        raise ValueError("El archivo no tiene el formato esperado.")

    df['label'] = df['label'].map({0: 'neg', 2: 'neu', 4: 'pos'})
    return df

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

    print("✅ ✅ ✅")

if __name__ == "__main__":
    main()
