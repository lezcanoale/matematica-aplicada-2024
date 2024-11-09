
# Importamos las bibliotecas necesarias
import pandas as pd
import numpy as np
import re
import time
import nltk
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from fpdf import FPDF

# Descargamos el lexicón de VADER (sólo si es la primera vez que se ejecuta el código)
nltk.download("vader_lexicon")

# ========================
# Sección 1: Preparación del Dataset
# ========================
# Aquí se carga y procesa el conjunto de datos Sentiment140 para limpieza de texto básica.

# Cargamos el archivo de datos 'test_data.csv' 
datos = pd.read_csv('test_data.csv')

# Función para limpiar el texto
def limpiar_texto(texto):
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)  # Eliminamos caracteres especiales
    texto = re.sub(r'\b\w\b', '', texto)       # Eliminamos palabras de una letra
    return texto.strip().lower()               # Convertimos a minúsculas y quitamos espacios

datos['texto_limpio'] = datos['sentence'].apply(limpiar_texto)

# ========================
# Sección 2: Análisis de Sentimientos
# ========================
# Utilizamos VADER de la librería NLTK para obtener los puntajes positivos y negativos de cada oración.

analizador_sentimientos = SentimentIntensityAnalyzer()
datos['puntaje_positivo'] = datos['texto_limpio'].apply(lambda x: analizador_sentimientos.polarity_scores(x)['pos'])
datos['puntaje_negativo'] = datos['texto_limpio'].apply(lambda x: analizador_sentimientos.polarity_scores(x)['neg'])

# ========================
# Sección 3: Fuzzificación
# ========================
# Definimos las variables difusas y sus funciones de membresía para los puntajes de sentimiento.

positivo = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'positivo')
negativo = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'negativo')
sentimiento = ctrl.Consequent(np.arange(-1, 1.1, 0.1), 'sentimiento')

# Definimos las funciones de membresía
positivo['bajo'] = fuzz.trimf(positivo.universe, [0, 0, 0.5])
positivo['medio'] = fuzz.trimf(positivo.universe, [0, 0.5, 1])
positivo['alto'] = fuzz.trimf(positivo.universe, [0.5, 1, 1])

negativo['bajo'] = fuzz.trimf(negativo.universe, [0, 0, 0.5])
negativo['medio'] = fuzz.trimf(negativo.universe, [0, 0.5, 1])
negativo['alto'] = fuzz.trimf(negativo.universe, [0.5, 1, 1])

sentimiento['negativo'] = fuzz.trimf(sentimiento.universe, [-1, -1, 0])
sentimiento['neutral'] = fuzz.trimf(sentimiento.universe, [-0.5, 0, 0.5])
sentimiento['positivo'] = fuzz.trimf(sentimiento.universe, [0, 1, 1])

# ========================
# Sección 4: Reglas Difusas
# ========================
regla1 = ctrl.Rule(positivo['alto'] & negativo['bajo'], sentimiento['positivo'])
regla2 = ctrl.Rule(positivo['medio'] & negativo['bajo'], sentimiento['positivo'])
regla3 = ctrl.Rule(positivo['bajo'] & negativo['alto'], sentimiento['negativo'])
regla4 = ctrl.Rule(positivo['medio'] & negativo['medio'], sentimiento['neutral'])
regla5 = ctrl.Rule(positivo['bajo'] & negativo['bajo'], sentimiento['neutral'])

# Creamos el sistema de control
control_sentimiento = ctrl.ControlSystem([regla1, regla2, regla3, regla4, regla5])
# Simulamos el sistema de control
simulacion_sentimiento = ctrl.ControlSystemSimulation(control_sentimiento)

# ========================
# Sección 5: Defuzzificación
# ========================
resultados_fuzzificados, etiquetas_sentimiento, tiempos_fuzz, tiempos_defuzz, tiempos_totales = [], [], [], [], []

# Aplicamos la fuzzificación y defuzzificación en cada entrada
for i, fila in datos.iterrows():
    # Fuzzificación
    inicio_fuzz = time.time()
    simulacion_sentimiento.input['positivo'] = fila['puntaje_positivo']
    simulacion_sentimiento.input['negativo'] = fila['puntaje_negativo']
    tiempo_fuzz = time.time() - inicio_fuzz

    # Defuzzificación
    inicio_defuzz = time.time()
    simulacion_sentimiento.compute()
    tiempo_defuzz = time.time() - inicio_defuzz
    tiempo_total = tiempo_fuzz + tiempo_defuzz

    # Obtenemos el valor defuzzificado y el resultado de la inferencia
    valor_fuzzificado = simulacion_sentimiento.output['sentimiento']
    resultados_fuzzificados.append(valor_fuzzificado)
    etiqueta_sentimiento = 'Positivo' if valor_fuzzificado > 0.5 else ('Negativo' if valor_fuzzificado < -0.5 else 'Neutral')
    etiquetas_sentimiento.append(etiqueta_sentimiento)

    # Guardamos los tiempos y resultados
    tiempos_fuzz.append(round(tiempo_fuzz, 8))
    tiempos_defuzz.append(round(tiempo_defuzz, 8))
    tiempos_totales.append(round(tiempo_total, 8))

# Guardamos los resultados en el DataFrame
datos['resultado_fuzzificado'] = resultados_fuzzificados
datos['sentimiento_defuzzificado'] = etiquetas_sentimiento
datos['tiempo_fuzzificacion'] = tiempos_fuzz
datos['tiempo_defuzzificacion'] = tiempos_defuzz
datos['tiempo_total'] = tiempos_totales

# ========================
# Sección 6: Resultados y Exportación
# ========================
resumen_sentimientos = datos['sentimiento_defuzzificado'].value_counts()
tiempo_total_ejecucion = datos['tiempo_total'].sum()
tiempo_promedio_ejecucion = tiempo_total_ejecucion / len(datos)

# Renombramos columnas y exportamos resultados finales
datos['etiqueta_original'] = datos['sentimiento_defuzzificado']
datos.rename(columns={
    'sentence': 'Texto Original',
    'etiqueta_original': 'Etiqueta Original',
    'puntaje_positivo': 'Puntaje Positivo',
    'puntaje_negativo': 'Puntaje Negativo',
    'resultado_fuzzificado': 'Valor Inferido',
    'tiempo_fuzzificacion': 'Tiempo Fuzzificación',
    'tiempo_defuzzificacion': 'Tiempo Defuzzificación',
    'tiempo_total': 'Tiempo Ejecución'
}, inplace=True)

# Reordenamos las columnas según lo requerido
datos[['Texto Original', 'Etiqueta Original', 'Puntaje Positivo', 'Puntaje Negativo',
       'Valor Inferido', 'Tiempo Fuzzificación', 'Tiempo Defuzzificación', 'Tiempo Ejecución']].to_csv('resultados_finales.csv', index=False)

print(f"Total de oraciones procesadas: {len(datos)}")
print(f"Total de positivos: {resumen_sentimientos.get('Positivo', 0)}")
print(f"Total de negativos: {resumen_sentimientos.get('Negativo', 0)}")
print(f"Total de neutrales: {resumen_sentimientos.get('Neutral', 0)}")
print(f"Tiempo promedio de ejecución: {tiempo_promedio_ejecucion:.8f} s")
