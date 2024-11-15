# Predicción de Compra de Cliente en Comercio Electrónico

Este proyecto utiliza **modelos de aprendizaje automático** para predecir si un cliente realizará una compra en una tienda en línea. Se basa en **modelos de clasificación supervisada** y en la evaluación de métricas de rendimiento para seleccionar el modelo más adecuado.

## Objetivo del Proyecto

El objetivo de este proyecto es:
1. **Predecir la probabilidad de compra** de un cliente en función de sus características demográficas y su comportamiento en el sitio web.
2. **Evaluar el rendimiento de varios modelos de clasificación** como el Árbol de Decisión, Random Forest, XGBoost, SVM y Regresión Logística.
3. **Seleccionar el mejor modelo** en términos de precisión, sensibilidad, especificidad, F1 Score y AUC, y recomendar el más adecuado según el contexto.

## Características del Dataset Sintético

El dataset simulado incluye las siguientes variables:
- **age**: Edad del cliente.
- **income**: Ingreso mensual del cliente.
- **gender**: Género del cliente (Male/Female).
- **web_visits**: Número de visitas al sitio web en el último mes.
- **purchased**: Variable objetivo (1 = realizó una compra, 0 = no realizó una compra).

La clase objetivo (`purchased`) está balanceada en 50% para asegurar un análisis justo de los modelos.

## Modelos de Clasificación Utilizados

El proyecto emplea cinco modelos de clasificación:
1. **Árbol de Decisión** - Método sencillo y fácil de interpretar.
2. **Random Forest** - Conjunto de árboles de decisión que mejora la precisión y robustez.
3. **XGBoost** - Algoritmo avanzado de boosting que optimiza el rendimiento.
4. **SVM (Máquina de Soporte Vectorial)** - Algoritmo potente para clasificación binaria.
5. **Regresión Logística** - Método clásico para problemas de clasificación.

Cada modelo es evaluado con métricas de clasificación y se elige el mejor con base en el F1 Score.

## Evaluación y Métricas

Las métricas utilizadas para evaluar el rendimiento de los modelos son:
- **Precisión**: Proporción de predicciones positivas correctas.
- **Sensibilidad (Recall)**: Proporción de verdaderos positivos capturados.
- **Especificidad**: Proporción de verdaderos negativos correctamente clasificados.
- **F1 Score**: Equilibrio entre precisión y sensibilidad.
- **AUC**: Área bajo la curva ROC, mide la capacidad de discriminación del modelo.

## Recomendación de Modelo

La recomendación final del modelo depende del contexto:
- **Alta sensibilidad**: Si el objetivo es detectar todos los compradores potenciales, elige un modelo con alta sensibilidad.
- **Alta especificidad y precisión**: Si se necesita reducir falsos positivos, selecciona un modelo con alta especificidad.

## Requisitos

Asegúrate de tener instalados los siguientes paquetes en R:

```r
install.packages(c("caret", "xgboost", "e1071", "randomForest", "pROC", "dplyr"))
```
## Uso del Código
1. Preparación del Dataset

## Generar un dataset sintético
```r

set.seed(123)
n <- 10000
data <- data.frame(
  age = round(runif(n, 18, 70)),
  income = round(runif(n, 20000, 120000)),
  gender = factor(sample(c("Male", "Female"), n, replace = TRUE)),
  web_visits = round(runif(n, 0, 20)),
  purchased = factor(sample(c(0, 1), n, replace = TRUE, prob = c(0.5, 0.5))) 
)
```

2. Entrenamiento y Evaluación de Modelos
r
Copiar código

## Dividir el dataset en conjuntos de entrenamiento y prueba
```r

trainIndex <- createDataPartition(data$purchased, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]
```
## Entrenar modelos y evaluar su rendimiento

```r

saveRDS(mejor_modelo, file = "mejor_modelo.rds")
```
3. Comparación de Resultados e Interpretación Dinámica
El código proporciona una comparación de resultados entre los modelos, mostrando las métricas de precisión, sensibilidad, especificidad, F1 Score y AUC. También genera una interpretación dinámica que comenta sobre el rendimiento de cada modelo basado en sus métricas, para ayudar a seleccionar el más adecuado.

4. Guardar el Mejor Modelo
El modelo con el mejor F1 Score se guarda en un archivo .rds para su uso futuro:
```r

saveRDS(mejor_modelo, file = "mejor_modelo.rds")
```

Ejemplo de Resultados
La salida incluirá una comparación de métricas para cada modelo, como el siguiente ejemplo:
```r

yaml
```


Comparación de Resultados

Árbol de Decisión:
Precisión: 88.9 %
Sensibilidad: 75.0 %
Especificidad: 66.0 %
F1 Score: 81.59 %
AUC: 0.75

Random Forest:
Precisión: 64.83 %
Sensibilidad: 83.79 %
Especificidad: 22.83 %
F1 Score: 76.65 %
AUC: 0.82

XGBoost:
Precisión: 65.43 %
Sensibilidad: 86.41 %
Especificidad: 58.97 %
F1 Score: 77.50 %
AUC: 0.85

Además, la interpretación dinámica del modelo ayudará a comprender el rendimiento de cada uno y a elegir el más adecuado para el contexto de negocio.
Interpretación y Recomendación Final

La recomendación final del modelo se basa en el análisis de sus métricas y en el contexto en el que se usará:

XGBoost: Alta sensibilidad, ideal para detectar la mayoría de los compradores.
Árbol de Decisión: Alta precisión, adecuado para reducir falsos positivos.
El mejor modelo en términos de F1 Score se selecciona automáticamente para ser guardado y utilizado.

Contacto
Para preguntas o sugerencias sobre este proyecto, no dudes en contactarme.


