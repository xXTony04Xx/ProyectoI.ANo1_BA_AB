import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

def limpiar(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+", "", texto)       # eliminar URLs
    texto = re.sub(r"@\w+", "", texto)          # eliminar menciones
    texto = re.sub(r"#\w+", "", texto)          # eliminar hashtags
    texto = re.sub(r"[^a-záéíóúüñ\s]", "", texto)  # mantener solo letras y espacios
    texto = re.sub(r"\s+", " ", texto).strip()  # quitar espacios extra

    # Eliminar stopwords
    stop_words = set(stopwords.words('spanish'))
    palabras = texto.split()
    texto = " ".join([p for p in palabras if p not in stop_words])

    return texto
