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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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

    print("Dividiendo datos 70% entrenamiento / 30% prueba...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['tokens'], df['label'], test_size=0.3, random_state=42
    )

    print("Entrenando modelo...")
    nb = NaiveBayes()
    nb.entrenar(X_train, y_train)

    print("Evaluando modelo...")
    y_pred = [nb.predecir(tokens) for tokens in X_test]
    print("\n=== Métricas del Modelo ===")
    print(classification_report(y_test, y_pred, digits=4))

    print(f"Guardando modelo en {MODEL_PATH}")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(nb, f)

    print("✅ Entrenamiento completo")

if __name__ == "__main__":
    main()
