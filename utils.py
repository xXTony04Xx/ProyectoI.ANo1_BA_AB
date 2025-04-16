import csv
from collections import Counter
from preprocess import limpiar

def cargar_dataset(ruta, limites={"positivo": 10000, "negativo": 10000, "neutro": 20000}):
    
    textos = []
    etiquetas = []
    contador = {"positivo": 0, "negativo": 0, "neutro": 0}
    
    try:
        with open(ruta, encoding="utf-8") as f:
            lector = csv.DictReader(f)
            
            for fila in lector:
                if "text" not in fila or "label" not in fila:
                    continue
                
                etiqueta = fila["label"].lower()  # Normalizar a minúsculas
                texto = fila["text"]
                
                # Solo procesar las etiquetas que nos interesan
                if etiqueta not in contador:
                    continue
                
                # Si ya alcanzamos el límite para esta clase, saltar
                if contador[etiqueta] >= limites.get(etiqueta, 0):
                    continue
                
                # Procesar y agregar el tweet
                texto_limpio = limpiar(texto)
                textos.append(texto_limpio)
                etiquetas.append(etiqueta)
                contador[etiqueta] += 1
                
                # Mostrar progreso cada 1000 tweets
                if sum(contador.values()) % 1000 == 0:
                    print(f"Procesados: {sum(contador.values())} tweets...")
                
                # Terminar si todas las clases alcanzaron sus límites
                if all(contador[cls] >= limites[cls] for cls in limites):
                    break
        
        print("\n✅ Dataset cargado correctamente.")
        print("Distribución de clases final:")
        for clase in contador:
            print(f"{clase.capitalize()}: {contador[clase]}")
        print(f"Total: {len(textos)} tweets\n")
        
    except FileNotFoundError:
        print(f"⚠️ Archivo no encontrado: {ruta}")
    except Exception as e:
        print(f"❌ Error leyendo el archivo: {e}")
    
    return textos, etiquetas