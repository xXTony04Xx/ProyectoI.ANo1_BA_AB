import math
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self, alpha=1.0, class_weights=None):
        """
        Inicializa el clasificador Naive Bayes.
        
        Args:
            alpha (float): Parámetro de suavizado (Laplace)
            class_weights (dict): Pesos para cada clase (ej. {'positivo': 1.2, 'neutro': 1.5})
        """
        self.alpha = alpha
        self.class_weights = class_weights if class_weights else {}
        self.reset()
    
    def reset(self):
        """Reinicia el estado del modelo"""
        self.class_counts = defaultdict(int)
        self.class_word_counts = defaultdict(lambda: defaultdict(int))
        self.class_total_words = defaultdict(int)
        self.vocabulary = set()
        self.classes = set()
    
    def entrenar(self, textos, etiquetas):
        """Entrena el modelo con los textos y etiquetas proporcionados"""
        self.reset()
        
        # Contar palabras por clase
        for texto, etiqueta in zip(textos, etiquetas):
            self.class_counts[etiqueta] += 1
            self.classes.add(etiqueta)
            
            palabras = texto.split()
            for palabra in palabras:
                self.class_word_counts[etiqueta][palabra] += 1
                self.class_total_words[etiqueta] += 1
                self.vocabulary.add(palabra)
        
        # Aplicar pesos de clases si existen
        if self.class_weights:
            for clase, peso in self.class_weights.items():
                if clase in self.class_counts:
                    self.class_counts[clase] = int(self.class_counts[clase] * peso)
    
    def _calcular_log_probabilidad_clase(self, clase):
        """Calcula el logaritmo de la probabilidad a priori de una clase"""
        total_docs = sum(self.class_counts.values())
        class_count = self.class_counts[clase]
        return math.log(class_count / total_docs)
    
    def _calcular_log_probabilidad_palabra(self, palabra, clase):
        """Calcula el logaritmo de P(palabra|clase)"""
        word_count = self.class_word_counts[clase].get(palabra, 0)
        total_words = self.class_total_words[clase]
        vocab_size = len(self.vocabulary)
        
        return math.log(
            (word_count + self.alpha) / 
            (total_words + self.alpha * vocab_size)
        )
    
    def predecir(self, texto):
        """Predice la clase más probable para el texto dado"""
        palabras = texto.split()
        mejor_clase = None
        mejor_log_prob = -float('inf')
        
        for clase in self.classes:
            log_prob = self._calcular_log_probabilidad_clase(clase)
            
            for palabra in palabras:
                log_prob += self._calcular_log_probabilidad_palabra(palabra, clase)
            
            # Aplicar peso de clase si existe
            if clase in self.class_weights:
                log_prob += math.log(self.class_weights[clase])
            
            if log_prob > mejor_log_prob:
                mejor_log_prob = log_prob
                mejor_clase = clase
        
        return mejor_clase
    
    def predecir_con_confianza(self, texto):
        """Predice la clase con las probabilidades normalizadas"""
        palabras = texto.split()
        log_probs = {}
        
        # Calcular log probabilidades
        for clase in self.classes:
            log_prob = self._calcular_log_probabilidad_clase(clase)
            
            for palabra in palabras:
                log_prob += self._calcular_log_probabilidad_palabra(palabra, clase)
            
            # Aplicar peso de clase
            if clase in self.class_weights:
                log_prob += math.log(self.class_weights[clase])
            
            log_probs[clase] = log_prob
        
        # Convertir log probs a probabilidades normalizadas
        max_log_prob = max(log_probs.values())
        probs = {
            clase: math.exp(log_prob - max_log_prob)
            for clase, log_prob in log_probs.items()
        }
        
        # Normalizar
        sum_probs = sum(probs.values())
        confianzas = {
            clase: prob / sum_probs
            for clase, prob in probs.items()
        }
        
        mejor_clase = max(confianzas.items(), key=lambda x: x[1])[0]
        return mejor_clase, confianzas