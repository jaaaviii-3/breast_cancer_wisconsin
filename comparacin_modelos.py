import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. DATOS DE LOS MODELOS
# ============================================================

modelos = [
    "LDA",
    "PCA + LDA 3 CP",
    "PCA + LDA 5 CP",
    "PCA + LDA 7 CP",
    "PCA + LDA 14 CP",
    "Árbol completo",
    "Random Forest",
    "Red neuronal"
]

accuracy = [
    95.61404,
    93.85965,
    94.73684,
    94.73684,
    94.73684,
    92.98246,
    94.73684,
    100.00000
]

sensibilidad = [
    98.36066,
    100.00000,
    100.00000,
    100.00000,
    100.00000,
    93.44262,
    95.08197,
    100.00000
]

especificidad = [
    92.45283,
    86.79245,
    88.67925,
    88.67925,
    88.67925,
    92.45283,
    94.33962,
    100.00000
]

# ============================================================
# 2. CONFIGURACIÓN DE LA GRÁFICA
# ============================================================

x = np.arange(len(modelos))
ancho = 0.25

plt.figure(figsize=(15, 7))

plt.bar(
    x - ancho,
    accuracy,
    width=ancho,
    label="Accuracy",
    color="#1f77b4"
)

plt.bar(
    x,
    sensibilidad,
    width=ancho,
    label="Sensibilidad",
    color="#ff7f0e"
)

plt.bar(
    x + ancho,
    especificidad,
    width=ancho,
    label="Especificidad",
    color="#2ca02c"
)

# ============================================================
# 3. FORMATO DE EJES Y TÍTULOS
# ============================================================

plt.xticks(x, modelos, rotation=35, ha="right")
plt.ylim(85, 101)

plt.ylabel("Porcentaje (%)")
plt.xlabel("Modelo")
plt.title("Comparación de accuracy, sensibilidad y especificidad por modelo")

plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)

# ============================================================
# 4. ETIQUETAS DE VALORES SOBRE LAS BARRAS
# ============================================================

for i in range(len(modelos)):

    plt.text(
        x[i] - ancho,
        accuracy[i] + 0.25,
        f"{accuracy[i]:.1f}",
        ha="center",
        va="bottom",
        fontsize=8
    )

    plt.text(
        x[i],
        sensibilidad[i] + 0.25,
        f"{sensibilidad[i]:.1f}",
        ha="center",
        va="bottom",
        fontsize=8
    )

    plt.text(
        x[i] + ancho,
        especificidad[i] + 0.25,
        f"{especificidad[i]:.1f}",
        ha="center",
        va="bottom",
        fontsize=8
    )

plt.tight_layout()

# ============================================================
# 5. GUARDAR Y MOSTRAR
# ============================================================

plt.savefig(
    "comparacion_metricas_modelos.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()