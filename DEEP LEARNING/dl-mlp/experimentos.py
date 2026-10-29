import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers

(X_train_raw, y_train), (X_test_raw, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train_raw / 255.0
X_test = X_test_raw / 255.0

EPOCAS = 10
LOTE = 64


def construir(neuronas, lr):
    capas = [layers.Input(shape=(28, 28)), layers.Flatten()]
    for n in neuronas:
        capas.append(layers.Dense(n, activation="relu"))
    capas.append(layers.Dense(10, activation="softmax"))
    modelo = models.Sequential(capas)
    modelo.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


# nombre: (neuronas por capa oculta, learning rate)
EXPERIMENTOS = {
    "Base (128-64, lr=0.001)": ([128, 64], 0.001),
    "Exp1 Profundo (128-64-128-128)": ([128, 64, 128, 128], 0.001),
    "Exp2 Pequeño (16-8)": ([16, 8], 0.001),
    "Exp3a lr=0.5": ([128, 64], 0.5),
    "Exp3b lr=0.00001": ([128, 64], 0.00001),
}

resultados = {}
for nombre, (neuronas, lr) in EXPERIMENTOS.items():
    tf.random.set_seed(42)
    np.random.seed(42)
    print(f"\n=== {nombre} ===")
    modelo = construir(neuronas, lr)
    t0 = time.time()
    h = modelo.fit(X_train, y_train, epochs=EPOCAS, batch_size=LOTE,
                   validation_split=0.2, verbose=0)
    duracion = time.time() - t0
    loss_t, acc_t = modelo.evaluate(X_test, y_test, verbose=0)
    resultados[nombre] = h.history
    print(f"Tiempo: {duracion:.1f}s | loss train final: {h.history['loss'][-1]:.4f} "
          f"| loss val final: {h.history['val_loss'][-1]:.4f} "
          f"| accuracy test: {acc_t * 100:.2f}%")

# Gráfica comparativa de pérdida de validación
plt.figure(figsize=(10, 6))
for nombre, hist in resultados.items():
    plt.plot(hist["val_loss"], label=nombre)
plt.yscale("log")
plt.xlabel("Época")
plt.ylabel("Pérdida de validación (escala log)")
plt.title("Comparación de experimentos")
plt.legend()
plt.grid(True)
plt.savefig("comparacion_perdida.png")
plt.close()
print("\nGráfica guardada: comparacion_perdida.png")
