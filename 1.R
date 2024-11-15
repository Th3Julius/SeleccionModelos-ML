install.packages(c("caret", "xgboost", "e1071", "randomForest", "pROC"))
library(caret)
library(xgboost)
library(e1071)
library(randomForest)
library(pROC)
library(dplyr)

# Generar un dataset sintético 
set.seed(123)
n <- 10000
data <- data.frame(
  age = round(runif(n, 18, 70)),
  income = round(runif(n, 20000, 120000)),
  gender = factor(sample(c("Male", "Female"), n, replace = TRUE)),
  web_visits = round(runif(n, 0, 20)),
  purchased = factor(sample(c(0, 1), n, replace = TRUE, prob = c(0.5, 0.5))) # Cambiar probabilidad para balancear
)

# Dividir datos en train y test
set.seed(123)
trainIndex <- createDataPartition(data$purchased, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

train_dummy <- model.matrix(purchased ~ ., data = trainData)[, -1]
test_dummy <- model.matrix(purchased ~ ., data = testData)[, -1]
train_labels <- as.numeric(as.character(trainData$purchased))
test_labels <- as.numeric(as.character(testData$purchased))


evaluar_modelo <- function(predicciones, labels) {
  pred_binario <- as.factor(ifelse(predicciones > 0.5, 1, 0)) # Umbral de 0.5
  labels <- as.factor(labels)
  
  levels(pred_binario) <- levels(labels)
  
  cm <- confusionMatrix(pred_binario, labels, positive = "1")
  precision <- cm$byClass["Pos Pred Value"]
  sensibilidad <- cm$byClass["Sensitivity"]
  especificidad <- cm$byClass["Specificity"]
  f1 <- cm$byClass["F1"]
  return(list(Precision = precision, Sensibilidad = sensibilidad, Especificidad = especificidad, F1 = f1))
}


# 1. Árbol de Decisión
modelo_arbol <- train(purchased ~ ., data = trainData, method = "rpart")
pred_arbol <- predict(modelo_arbol, testData, type = "prob")[,2]
resultado_arbol <- evaluar_modelo(pred_arbol, test_labels)

# 2. Random Forest
modelo_rf <- randomForest(purchased ~ ., data = trainData, importance = TRUE)
pred_rf <- predict(modelo_rf, testData, type = "prob")[,2]
resultado_rf <- evaluar_modelo(pred_rf, test_labels)

# 3. XGBoost
modelo_xgb <- xgboost(data = as.matrix(train_dummy), label = train_labels, nrounds = 100, objective = "binary:logistic", verbose = 0)
pred_xgb <- predict(modelo_xgb, as.matrix(test_dummy))
resultado_xgb <- evaluar_modelo(pred_xgb, test_labels)

# 4. SVM
modelo_svm <- svm(purchased ~ ., data = trainData, probability = TRUE)
pred_svm <- attr(predict(modelo_svm, testData, probability = TRUE), "probabilities")[,2]
resultado_svm <- evaluar_modelo(pred_svm, test_labels)

# 5. Regresión Logística
modelo_log <- train(purchased ~ ., data = trainData, method = "glm", family = "binomial")
pred_log <- predict(modelo_log, testData, type = "prob")[,2]
resultado_log <- evaluar_modelo(pred_log, test_labels)

# Curvas ROC y AUC 
roc_arbol <- roc(test_labels, pred_arbol)
roc_rf <- roc(test_labels, pred_rf)
roc_xgb <- roc(test_labels, pred_xgb)
roc_svm <- roc(test_labels, pred_svm)
roc_log <- roc(test_labels, pred_log)

# Comparación de Resultados
cat("Comparación de Resultados\n")
modelos <- list("Árbol de Decisión" = resultado_arbol, "Random Forest" = resultado_rf, "XGBoost" = resultado_xgb, 
                "SVM" = resultado_svm, "Regresión Logística" = resultado_log)
rocs <- list("Árbol de Decisión" = roc_arbol, "Random Forest" = roc_rf, "XGBoost" = roc_xgb, 
             "SVM" = roc_svm, "Regresión Logística" = roc_log)

for (modelo in names(modelos)) {
  cat("\n", modelo, ":\n")
  cat("Precisión:", round(modelos[[modelo]]$Precision * 100, 2), "%\n")
  cat("Sensibilidad:", round(modelos[[modelo]]$Sensibilidad * 100, 2), "%\n")
  cat("Especificidad:", round(modelos[[modelo]]$Especificidad * 100, 2), "%\n")
  cat("F1 Score:", round(modelos[[modelo]]$F1 * 100, 2), "%\n")
  cat("AUC:", round(auc(rocs[[modelo]]), 2), "\n")
}
# Comparación de Resultados
cat("Comparación de Resultados\n")
modelos <- list("Árbol de Decisión" = resultado_arbol, "Random Forest" = resultado_rf, "XGBoost" = resultado_xgb, 
                "SVM" = resultado_svm, "Regresión Logística" = resultado_log)
rocs <- list("Árbol de Decisión" = roc_arbol, "Random Forest" = roc_rf, "XGBoost" = roc_xgb, 
             "SVM" = roc_svm, "Regresión Logística" = roc_log)

for (modelo in names(modelos)) {
  cat("\n", modelo, ":\n")
  cat("Precisión:", round(modelos[[modelo]]$Precision * 100, 2), "%\n")
  cat("Sensibilidad:", round(modelos[[modelo]]$Sensibilidad * 100, 2), "%\n")
  cat("Especificidad:", round(modelos[[modelo]]$Especificidad * 100, 2), "%\n")
  cat("F1 Score:", round(modelos[[modelo]]$F1 * 100, 2), "%\n")
  cat("AUC:", round(auc(rocs[[modelo]]), 2), "\n")
}


# Comparación de Resultados
cat("Comparación de Resultados\n")
modelos <- list("Árbol de Decisión" = resultado_arbol, "Random Forest" = resultado_rf, "XGBoost" = resultado_xgb, 
                "SVM" = resultado_svm, "Regresión Logística" = resultado_log)
rocs <- list("Árbol de Decisión" = roc_arbol, "Random Forest" = roc_rf, "XGBoost" = roc_xgb, 
             "SVM" = roc_svm, "Regresión Logística" = roc_log)

for (modelo in names(modelos)) {
  cat("\n", modelo, ":\n")
  cat("Precisión:", round(modelos[[modelo]]$Precision * 100, 2), "%\n")
  cat("Sensibilidad:", round(modelos[[modelo]]$Sensibilidad * 100, 2), "%\n")
  cat("Especificidad:", round(modelos[[modelo]]$Especificidad * 100, 2), "%\n")
  cat("F1 Score:", round(modelos[[modelo]]$F1 * 100, 2), "%\n")
  cat("AUC:", round(auc(rocs[[modelo]]), 2), "\n")
}

#segunda interpretacion
cat("\nInterpretación de los Resultados:\n\n")

for (modelo in names(modelos)) {
  cat(modelo, ":\n")
  
  precision <- modelos[[modelo]]$Precision * 100
  sensibilidad <- modelos[[modelo]]$Sensibilidad * 100
  especificidad <- modelos[[modelo]]$Especificidad * 100
  f1_score <- modelos[[modelo]]$F1 * 100
  
  # Comentario basado en precisión
  if (precision >= 85) {
    cat("Este modelo muestra una alta precisión (", round(precision, 2), "%), lo que indica que la mayoría de sus predicciones positivas son correctas.\n", sep = "")
  } else {
    cat("La precisión de este modelo es moderada (", round(precision, 2), "%). Esto indica que algunas de sus predicciones positivas son incorrectas.\n", sep = "")
  }
  
  # Comentario basado en sensibilidad
  if (sensibilidad >= 85) {
    cat("Este modelo tiene una alta sensibilidad (", round(sensibilidad, 2), "%), lo que significa que es eficaz en detectar la mayoría de los compradores.\n", sep = "")
  } else {
    cat("La sensibilidad de este modelo es baja (", round(sensibilidad, 2), "%), lo que significa que no detecta todos los compradores.\n", sep = "")
  }
  
  # Comentario basado en especificidad
  if (especificidad >= 85) {
    cat("Este modelo también muestra una alta especificidad (", round(especificidad, 2), "%), indicando que clasifica correctamente la mayoría de los no compradores.\n", sep = "")
  } else {
    cat("La especificidad de este modelo es baja (", round(especificidad, 2), "%), indicando una mayor cantidad de falsos positivos.\n", sep = "")
  }
  
  # Comentario basado en F1 Score
  if (f1_score >= 80) {
    cat("El F1 Score de este modelo es alto (", round(f1_score, 2), "%), indicando un buen equilibrio entre precisión y sensibilidad.\n", sep = "")
  } else {
    cat("El F1 Score de este modelo es moderado (", round(f1_score, 2), "%), sugiriendo que podría mejorarse en términos de equilibrio entre precisión y sensibilidad.\n", sep = "")
  }
  
  cat("\n")
}

# Recomendación final basada en el rendimiento global
cat("Recomendación Final Basada en el Rendimiento:\n")
mejor_modelo_nombre <- names(modelos)[which.max(sapply(modelos, function(x) x$F1))]
cat("El modelo con el mejor F1 Score es ", mejor_modelo_nombre, ", lo que sugiere que es el más adecuado para un equilibrio entre precisión y sensibilidad.\n", sep = "")
cat("Sin embargo, la elección del modelo puede ajustarse según el contexto:\n")
cat("- Si el objetivo es detectar todos los compradores posibles (minimizar falsos negativos), elige un modelo con alta sensibilidad.\n")
cat("- Si la prioridad es reducir los falsos positivos, selecciona un modelo con alta precisión y especificidad.\n\n")

# Selección y guardado del mejor modelo basado en F1 Score
mejor_f1 <- max(sapply(modelos, function(x) x$F1))
mejor_modelo_nombre <- names(modelos)[which(sapply(modelos, function(x) x$F1) == mejor_f1)][1]
print(paste("El mejor modelo es:", mejor_modelo_nombre))

mejor_modelo <- switch(
  mejor_modelo_nombre,
  "Árbol de Decisión" = modelo_arbol,
  "Random Forest" = modelo_rf,
  "XGBoost" = modelo_xgb,
  "SVM" = modelo_svm,
  "Regresión Logística" = modelo_log
)
saveRDS(mejor_modelo, file = "mejor_modelo.rds")
print("El mejor modelo ha sido guardado en 'mejor_modelo.rds'.")

