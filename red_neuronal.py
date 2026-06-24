import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)

from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# 0. CONFIGURACIÓN DE GRÁFICAS
# ============================================================

plt.ion()


# ============================================================
# 1. FIJAR SEMILLAS
# ============================================================

SEED = 123

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. LECTURA DE DATOS
# ============================================================

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

nombres = [
    "ID", "Diagnosis",
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

dat = pd.read_csv(url, header=None, names=nombres)

print("\nPrimeras filas:")
print(dat.head())

print("\nDimensión del dataset:")
print(dat.shape)

print("\nDistribución total de Diagnosis:")
print(dat["Diagnosis"].value_counts())
print(dat["Diagnosis"].value_counts(normalize=True) * 100)


# ============================================================
# 3. CARGAR IDS DEL TEST USADO EN R
# ============================================================

test_ids = pd.read_csv("test_ids_breast_cancer.csv")

test_ids = test_ids["ID"].values

test_mask = dat["ID"].isin(test_ids)

test = dat[test_mask].copy()
train_val = dat[~test_mask].copy()

print("\nTamaño train_val:")
print(train_val.shape)

print("\nTamaño test:")
print(test.shape)

print("\nDistribución test usada en R:")
print(test["Diagnosis"].value_counts())
print(test["Diagnosis"].value_counts(normalize=True) * 100)

print("\nComprobación:")
print("Total test:", len(test))
print("Benignos test:", sum(test["Diagnosis"] == "B"))
print("Malignos test:", sum(test["Diagnosis"] == "M"))


# ============================================================
# 4. PREPARACIÓN DE VARIABLES
# ============================================================

X_train_val = train_val.drop(columns=["ID", "Diagnosis"])
y_train_val = train_val["Diagnosis"].map({"B": 0, "M": 1})

X_test = test.drop(columns=["ID", "Diagnosis"])
y_test = test["Diagnosis"].map({"B": 0, "M": 1})

print("\nCodificación de y en test:")
print(pd.crosstab(test["Diagnosis"], y_test))


# ============================================================
# 5. SEPARACIÓN TRAIN / VALIDATION
# ============================================================

# El test ya está fijado por R.
# Ahora solo separamos validación dentro del conjunto de entrenamiento.

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.20,
    random_state=SEED,
    stratify=y_train_val
)

print("\nTamaño train red neuronal:")
print(X_train.shape)

print("\nTamaño validation red neuronal:")
print(X_val.shape)

print("\nTamaño test final:")
print(X_test.shape)

print("\nDistribución train:")
print(y_train.value_counts(normalize=True) * 100)

print("\nDistribución validation:")
print(y_val.value_counts(normalize=True) * 100)

print("\nDistribución test:")
print(y_test.value_counts(normalize=True) * 100)


# ============================================================
# 6. ESCALADO DE VARIABLES
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 7. CALLBACK PARA GUARDAR MÉTRICAS DE TEST POR ÉPOCA
# ============================================================

class TestMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_test, y_test):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.test_loss = []
        self.test_accuracy = []
        self.test_precision = []
        self.test_recall = []
        self.test_auc = []

    def on_epoch_end(self, epoch, logs=None):
        results = self.model.evaluate(
            self.X_test,
            self.y_test,
            verbose=0,
            return_dict=True
        )

        self.test_loss.append(results["loss"])
        self.test_accuracy.append(results["accuracy"])
        self.test_precision.append(results["precision"])
        self.test_recall.append(results["recall"])
        self.test_auc.append(results["auc"])


test_callback = TestMetricsCallback(X_test_scaled, y_test)


# ============================================================
# 8. DEFINICIÓN DE LA RED NEURONAL
# ============================================================

model = models.Sequential()

model.add(layers.Dense(
    32,
    activation="relu",
    input_shape=(X_train_scaled.shape[1],)
))

model.add(layers.Dropout(0.20))

model.add(layers.Dense(
    16,
    activation="relu"
))

model.add(layers.Dropout(0.20))

model.add(layers.Dense(
    8,
    activation="relu"
))

model.add(layers.Dense(
    1,
    activation="sigmoid"
))

model.summary()


# ============================================================
# 9. COMPILACIÓN DEL MODELO
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)


# ============================================================
# 10. ENTRENAMIENTO
# ============================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=200,
    batch_size=16,
    validation_data=(X_val_scaled, y_val),
    callbacks=[early_stop, test_callback],
    verbose=1
)


# ============================================================
# 11. GRÁFICA DE FUNCIÓN DE PÉRDIDA
# ============================================================

plt.figure()
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.plot(test_callback.test_loss, label="Test loss")
plt.xlabel("Época")
plt.ylabel("Binary crossentropy")
plt.title("Evolución de la función de pérdida")
plt.legend()
plt.grid(True)
plt.savefig("loss_red_neuronal_train_val_test.png", dpi=300, bbox_inches="tight")
plt.show(block=False)
plt.pause(0.1)


# ============================================================
# 12. GRÁFICA DE ACCURACY
# ============================================================

plt.figure()
plt.plot(history.history["accuracy"], label="Train accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.plot(test_callback.test_accuracy, label="Test accuracy")
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.title("Evolución de la accuracy")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_red_neuronal_train_val_test.png", dpi=300, bbox_inches="tight")
plt.show(block=False)
plt.pause(0.1)


# ============================================================
# 13. EVALUACIÓN FINAL EN TEST
# ============================================================

test_results = model.evaluate(
    X_test_scaled,
    y_test,
    verbose=0,
    return_dict=True
)

print("\nResultados finales en TEST:")
for name, value in test_results.items():
    print(f"{name}: {value:.4f}")


# ============================================================
# 14. PREDICCIONES
# ============================================================

y_prob = model.predict(X_test_scaled).ravel()

threshold = 0.5

y_pred = (y_prob >= threshold).astype(int)


# ============================================================
# 15. MATRIZ DE CONFUSIÓN
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nMatriz de confusión:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benigno", "Maligno"]
)

disp.plot()
plt.title("Matriz de confusión - Red neuronal")
plt.xlabel("Predicción")
plt.ylabel("Valor real")
plt.savefig("matriz_confusion_red_neuronal.png", dpi=300, bbox_inches="tight")
plt.show(block=False)
plt.pause(0.1)


# ============================================================
# 16. MÉTRICAS MANUALES
# ============================================================

# cm =
# [[VN, FP],
#  [FN, VP]]

VN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
VP = cm[1, 1]

accuracy = accuracy_score(y_test, y_pred)

sensibilidad = recall_score(
    y_test,
    y_pred,
    pos_label=1
)

especificidad = VN / (VN + FP)

precision = precision_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    y_prob
)

print("\nMétricas finales:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Sensibilidad malignos: {sensibilidad * 100:.2f}%")
print(f"Especificidad benignos: {especificidad * 100:.2f}%")
print(f"Precisión malignos: {precision * 100:.2f}%")
print(f"F1-score malignos: {f1 * 100:.2f}%")
print(f"AUC: {auc:.4f}")

print("\nErrores:")
print(f"Falsos positivos: {FP}")
print(f"Falsos negativos: {FN}")


# ============================================================
# 17. FUNCIÓN DE COSTE
# ============================================================

coste_FP = 10
coste_FN = 500

funcion_coste = coste_FP * FP + coste_FN * FN

print("\nFunción de coste:")
print(f"Coste = {coste_FP} * FP + {coste_FN} * FN")
print(f"Coste = {coste_FP} * {FP} + {coste_FN} * {FN}")
print(f"Coste total = {funcion_coste}")


# ============================================================
# 18. TABLA RESUMEN DEL MODELO
# ============================================================

tabla_red = pd.DataFrame({
    "Modelo": ["Red neuronal MLP"],
    "Accuracy": [accuracy * 100],
    "Sensibilidad_M": [sensibilidad * 100],
    "Especificidad_B": [especificidad * 100],
    "Precision_M": [precision * 100],
    "F1_M": [f1 * 100],
    "AUC": [auc],
    "FP": [FP],
    "FN": [FN],
    "Funcion_coste": [funcion_coste]
})

print("\nTabla resumen:")
print(tabla_red)


# ============================================================
# 19. GUARDAR RESULTADOS
# ============================================================

tabla_red.to_csv(
    "resultados_red_neuronal.csv",
    index=False,
    sep=";",
    decimal=","
)

model.save("modelo_red_neuronal_breast_cancer.keras")

print("\nArchivos guardados:")
print("- loss_red_neuronal_train_val_test.png")
print("- accuracy_red_neuronal_train_val_test.png")
print("- matriz_confusion_red_neuronal.png")
print("- resultados_red_neuronal.csv")
print("- modelo_red_neuronal_breast_cancer.keras")


# ============================================================
# 20. MANTENER GRÁFICAS ABIERTAS
# ============================================================

input("\nCódigo ejecutado completo. Pulsa ENTER para cerrar el programa y las gráficas...")