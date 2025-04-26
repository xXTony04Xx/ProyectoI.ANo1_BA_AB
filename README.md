===========================================
          PROYECTO NAÏVE BAYES
===========================================

Clasificador de texto usando el algoritmo
Naïve Bayes implementado desde cero.

-------------------------------------------
 INSTRUCCIONES
-------------------------------------------

1. Instala los paquetes necesarios:
   > pip install -r requirements.txt

2. Ejecuta el servidor:
   > python app/main.py

3. Abre tu navegador en:
   > http://127.0.0.1:5000

-------------------------------------------
 ¿CÓMO FUNCIONA?
-------------------------------------------

- PREPROCESAMIENTO:
  * Convierte texto a minúsculas.
  * Elimina puntuación y URLs.
  * Tokeniza y limpia con NLTK.

- ENTRENAMIENTO:
  * Calcula frecuencias de palabras por clase.
  * Guarda totales y probabilidades iniciales.

- PREDICCIÓN:
  * Usa Laplace smoothing.
  * Calcula probabilidades logarítmicas.
  * Devuelve la clase más probable.

-------------------------------------------
 ESTRUCTURA DEL PROYECTO
-------------------------------------------

  app/
  ├── main.py             ← Servidor Flask
  ├── train.py            ← Entrenamiento
  ├── naiveBayes.py       ← Algoritmo NB
  ├── model/
  │   └── naive_bayes_model.pkl
  └── templates/
      └── index.html      ← Interfaz web

  static/
  └── sentiment140.csv    ← Dataset

  requirements.txt         ← Dependencias
  README.txt               ← Este archivo

-------------------------------------------
 ARCHIVOS IMPORTANTES
-------------------------------------------

- main.py           → Inicia servidor y usa el modelo
- naiveBayes.py     → Implementación del clasificador
- train.py          → Entrenamiento y carga del dataset
- index.html        → Página web para ingresar texto
- naive_bayes_model.pkl → Modelo entrenado (pickle)
- sentiment140.csv  → Dataset de tweets con etiquetas

-------------------------------------------
 DATASET
-------------------------------------------

Usamos el dataset Sentiment140 con:
  - 0 → Negativo (neg)
  - 2 → Neutro   (neu)
  - 4 → Positivo (pos)

Debe estar ubicado en:
  static/sentiment140.csv

Los datos fueron extraidos del siguiente
Dataset:
  https://www.kaggle.com/datasets/kazanova/sentiment140

-------------------------------------------
 DEPENDENCIAS
-------------------------------------------

Incluidas en el archivo requirements.txt:

- flask
- nltk
- pandas
- numpy
- scikit-learn

Instala con:
> pip install -r requirements.txt

-------------------------------------------
 EJECUCIÓN
-------------------------------------------

1. Asegúrate de que el archivo de datos `sentiment140.csv`
   esté en la carpeta `static/`.

2. Entrena el modelo con:
   > python app/train.py

   Esto generará el archivo `naive_bayes_model.pkl`
   dentro del directorio `app/model/`.

3. Ejecuta el servidor Flask:
   > python app/main.py

4. Abre tu navegador en:
   > http://127.0.0.1:5000

5. Escribe un texto en el formulario y haz clic en enviar.
   Verás el resultado del análisis de sentimiento.

===========================================
