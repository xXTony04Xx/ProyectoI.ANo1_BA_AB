def predict_text(text):
    # Aquí luego llamas tu modelo real
    if "feliz" in text.lower():
        return "Positivo"
    elif "triste" in text.lower():
        return "Negativo"
    else:
        return "Neutro"
