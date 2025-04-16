import pandas as pd
from deep_translator import GoogleTranslator

# Cargar dataset TweetEval ya en CSV
df = pd.read_csv("tweet_eval_sentiment.csv")

# Traducir solo las primeras N filas (por ejemplo, 5000 para pruebas)
limite = 20000
df = df.head(limite)

# Traducir columna 'text' al español
traductor = GoogleTranslator(source='auto', target='es')

def traducir(texto):
    try:
        return traductor.translate(texto)
    except:
        return texto  # si falla, dejar el original

print("Traduciendo tweets... esto puede tardar.")
df["text_es"] = df["text"].apply(traducir)

# Guardar nuevo archivo traducido
df.to_csv("tweet_eval_translated.csv", index=False, encoding="utf-8")
print("✅ Dataset traducido guardado como 'tweet_eval_translated.csv'")
