from collections import defaultdict
import math

class NaiveBayes:
    def __init__(self):
        self.clases = ['pos', 'neg', 'neu']
        self.vocabulario = set()
        self.prior = {}
        self.freq_palabra = {}
        self.total_palabras_clase = {}

    def entrenar(self, X, y):
        self.freq_palabra = {c: defaultdict(int) for c in self.clases}
        self.total_palabras_clase = {c: 0 for c in self.clases}
        self.prior = {c: 0 for c in self.clases}

        total_docs = len(y)
        for texto, etiqueta in zip(X, y):
            self.prior[etiqueta] += 1
            for palabra in texto:
                self.freq_palabra[etiqueta][palabra] += 1
                self.total_palabras_clase[etiqueta] += 1
                self.vocabulario.add(palabra)

        for c in self.clases:
            self.prior[c] /= total_docs

    def predecir(self, tokens):
        resultados = {}
        V = len(self.vocabulario)
        for c in self.clases:
            log_prob = 0
            for palabra in tokens:
                frecuencia = self.freq_palabra[c][palabra]
                prob = (frecuencia + 1) / (self.total_palabras_clase[c] + V)
                log_prob += math.log(prob)
            resultados[c] = math.log(self.prior[c]) + log_prob
        return max(resultados, key=resultados.get)
