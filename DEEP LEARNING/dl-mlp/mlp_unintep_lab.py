import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt

# 1. Reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

print(f"Versión de TensorFlow en uso: {tf.__version__}")

# 2. Datos (MNIST), normalizados a [0, 1]
mnist = tf.keras.datasets.mnist
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()

X_train = X_train_raw / 255.0
X_test = X_test_raw / 255.0

print(f"Forma de datos de entrenamiento: {X_train.shape}")
print(f"Forma de datos de prueba: {X_test.shape}")


# 3. Arquitectura MLP
def construir_mlp():
    model = models.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(name="Capa_Entrada_Flatten"),
        layers.Dense(128, activation="relu", name="Capa_Oculta_1"),
        layers.Dense(64, activation="relu", name="Capa_Oculta_2"),
        layers.Dense(10, activation="softmax", name="Capa_Salida_Softmax"),
    ], name="MLP_UNINTEP_RA2")
    return model


modelo_mlp = construir_mlp()
modelo_mlp.summary()

# 4. Compilación
TASA_DE_APRENDIZAJE = 0.001
modelo_mlp.compile(
    optimizer=optimizers.Adam(learning_rate=TASA_DE_APRENDIZAJE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# 5. Entrenamiento
TAMANO_LOTE = 64
EPOCAS = 10

print("\nIniciando entrenamiento del modelo...")
historial = modelo_mlp.fit(
    X_train, y_train_raw,
    epochs=EPOCAS,
    batch_size=TAMANO_LOTE,
    validation_split=0.2,
    verbose=1,
)

# 6. Evaluación en prueba
loss_test, accuracy_test = modelo_mlp.evaluate(X_test, y_test_raw, verbose=0)
print("\n==========================================")
print(f"Pérdida en Test (Loss): {loss_test:.4f}")
print(f"Precisión en Test (Accuracy): {accuracy_test * 100:.2f}%")
print("==========================================")

# 7. Gráfica de pérdida (útil para el reporte)
plt.plot(historial.history["loss"], label="Entrenamiento")
plt.plot(historial.history["val_loss"], label="Validación")
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.title("Modelo base")
plt.legend()
plt.grid(True)
plt.savefig("perdida_base.png")
plt.close()
