from datasets import load_dataset
import pandas as pd

# 1. Cargar dataset de HuggingFace
print("Descargando dataset TweetEval (sentiment)...")
dataset = load_dataset("tweet_eval", "sentiment")

# 2. Unir train, test y validation
df_train = pd.DataFrame(dataset['train'])
df_test = pd.DataFrame(dataset['test'])
df_val = pd.DataFrame(dataset['validation'])
df = pd.concat([df_train, df_test, df_val], ignore_index=True)

# 3. Convertir etiquetas numéricas a texto
# 0 = negativo, 1 = neutro, 2 = positivo
mapeo = {0: "negativo", 1: "neutro", 2: "positivo"}
df["label"] = df["label"].map(mapeo)

# 4. Guardar como CSV
df.to_csv("tweet_eval_sentiment.csv", index=False, encoding="utf-8")
print("✅ Dataset guardado como 'tweet_eval_sentiment.csv' con:")
print(df["label"].value_counts())
