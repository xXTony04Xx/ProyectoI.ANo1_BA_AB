from naive_bayes import NaiveBayesClassifier
from preprocess import limpiar
from utils import cargar_dataset

def main():
    # 1. Cargar datos con distribución específica (más neutros)
    print("Cargando dataset con énfasis en tweets neutros...")
    textos, etiquetas = cargar_dataset(
        "tweet_eval_sentiment.csv", 
        limites={"positivo": 10000, "negativo": 10000, "neutro": 20000}
    )

    # Mostrar distribución de clases y ejemplos representativos
    print("\nDistribución de clases en el dataset de entrenamiento:")
    from collections import Counter
    conteo = Counter(etiquetas)
    for clase, cantidad in conteo.items():
        print(f"{clase.capitalize()}: {cantidad} tweets ({cantidad/sum(conteo.values()):.1%})")

    # Encontrar ejemplos representativos de cada clase
    def encontrar_ejemplo(etiqueta_buscada):
        for i, etiqueta in enumerate(etiquetas):
            if etiqueta == etiqueta_buscada:
                return textos[i]
        return "No encontrado"

    print("\nEjemplos de cada clase:")
    print(f"Positivo: {encontrar_ejemplo('positivo')}")
    print(f"Negativo: {encontrar_ejemplo('negativo')}")
    print(f"Neutro:   {encontrar_ejemplo('neutro')}")

    # 2. Entrenar el modelo con ajuste para neutros
    print("\nEntrenando modelo con ajuste para clase neutra...")
    modelo = NaiveBayesClassifier(alpha=1.0, class_weights={"neutro": 1.5})  # Peso extra para neutros
    modelo.entrenar(textos, etiquetas)

    # 3. Pruebas con énfasis en casos neutros
    print("\nModelo entrenado. Probando con casos límite:")
    test_samples = [
        ("I love this product, it's amazing!", "positivo"),
        ("This is the worst experience ever", "negativo"),
        ("The service was okay, nothing special", "neutro"),
        ("I am very happy with my purchase!", "positivo"),
        ("I do not recommend this product, very poor quality", "negativo"),
        ("I'm going to watch a movie and then have dinner.", "neutro"),
        ("This place is okay, but nothing impressive.", "neutro"),
        ("The meeting was at 3pm in the main office", "neutro"),  # Neutral objetivo
        ("It's neither good nor bad", "neutro"),  # Neutral explícito
        ("Just sharing some thoughts", "neutro"),  # Neutral implícito
        ("Not great, not terrible", "neutro")  # Neutral mixto
    ]

    correct = 0
    for sample, expected in test_samples:
        texto_limpio = limpiar(sample)
        prediccion = modelo.predecir(texto_limpio)
        result = "ok" if prediccion == expected else "X"
        print(f"{result} Texto: '{sample[:50]}' → Predicción: {prediccion} (Esperado: {expected})")
        if result == "ok":
            correct += 1

    print(f"\nPrecisión en pruebas: {correct/len(test_samples):.1%}")

    # 4. Modo interactivo con análisis de confianza
    print("\n Modo interactivo (escribe 'salir' para terminar)")
    print("El modelo ahora mostrará la confianza en cada predicción:")
    while True:
        entrada = input("\n Escribe tu mensaje aca: ").strip()
        if entrada.lower() == "salir":
            break
        if not entrada:
            continue
            
        texto_limpio = limpiar(entrada)
        prediccion, confianza = modelo.predecir_con_confianza(texto_limpio)
        
        # Ajustar umbral para neutralidad
        if max(confianza.values()) < 0.6:  # Si ninguna clase tiene alta confianza
            prediccion = "neutro"
            print("🔍 El modelo no está seguro - clasificando como neutro por defecto")
        
        print(f"\nSe ha detectado: {prediccion.upper()}")
        print("Porcentajes:")
        for clase, prob in sorted(confianza.items(), key=lambda x: -x[1]):
            print(f"  {clase.capitalize()}: {prob:.1%}")

if __name__ == "__main__":
    main()