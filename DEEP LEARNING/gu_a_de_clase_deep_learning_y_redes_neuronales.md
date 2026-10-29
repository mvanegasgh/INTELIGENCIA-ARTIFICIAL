# UNINTEP - UNIDAD TECNOLÓGICA DEL VALLE DEL CAUCA
## Sede Roldanillo | Programa de Ingeniería y Tecnologías de la Información
### Asignatura: Inteligencia Artificial
---

* **Docente:** Profesor Titular de Inteligencia Artificial
* **Unidad Temática:** Fundamentos de Deep Learning y Arquitectura de Redes Neuronales
* **Resultado de Aprendizaje (RA):** **RA2** - Diseñar e implementar modelos de redes neuronales artificiales profundas para la resolución de problemas complejas mediante algoritmos de aprendizaje supervisado.
* **Objetivo de Enseñanza:** Deducir las diferencias estructurales entre Machine Learning y Deep Learning, programando una red neuronal artificial básica.
* **Ecosistema Tecnológico:** Python 3.x, TensorFlow / Keras, VS Code, Swarp.

---

## 1. Contextualización y Estructura Pedagógica

Para lograr las competencias del **RA2**, la sesión de trabajo se articula en tres momentos clave estructurados bajo la metodología de aula invertida y aprendizaje activo:

| Momento | Dedicación | Actividad Principal | Entregable / Producto |
| :--- | :---: | :--- | :--- |
| **Pre-clase** | 4h (Autónoma) | Lectura de Chollet (2021), Capítulos 1 y 2. Análisis del flujo conceptual de entrenamiento. | Mapa conceptual individual del flujo Forward-Loss-Backprop. |
| **Durante la clase** | 4h (Presencial) | Exposición interactiva de fundamentos matemáticos + Laboratorio práctico guiado en VS Code/Swarp. | Código ejecutable del modelo MLP en TensorFlow/Keras. |
| **Post-clase** | 4h (Autónoma) | Ajuste e investigación de hiperparámetros ($N^\circ$ capas, neuronas, $\eta$) y análisis del impacto en la función de pérdida ($\mathcal{L}$). | Reporte técnico de desempeño y comportamiento del error. |

---

## 2. Deep Learning vs. Machine Learning Tradicional

### 2.1 Paradigma del Aprendizaje: *Feature Engineering* vs. *Representation Learning*

En el **Machine Learning Tradicional** (algoritmos como Regresión Logística, Arboles de Decisión, SVM o Random Forest), la extracción de características (*Feature Engineering*) es una tarea predominantemente manual y dependiente del dominio. El ingeniero de datos debe transformar las entradas brutas $\mathbf{x}$ en un conjunto de atributos sintéticos $\phi(\mathbf{x})$ optimizados antes de alimentar al clasificador.

En contraste, el **Deep Learning** (Redes Neuronales Profundas) sustituye el pipeline manual por el **Aprendizaje Conjunto de Representaciones** (*Representation Learning*). El modelo aprende directamente una jerarquía de transformaciones geométricas no lineales compuestas:

$$H^{(1)} = f^{(1)}(X \cdot W^{(1)} + b^{(1)})$$
$$H^{(2)} = f^{(2)}(H^{(1)} \cdot W^{(2)} + b^{(2)})$$
$$\hat{Y} = f^{(L)}(H^{(L-1)} \cdot W^{(L)} + b^{(L)})$$

```
MACHINE LEARNING TRADICIONAL:
[ Datos Brutos ] ---> [ Extracción Manual de Atributos ] ---> [ Algoritmo ML ] ---> [ Resultado ]

DEEP LEARNING:
[ Datos Brutos ] ---> [ Capa 1: Borde ] ---> [ Capa 2: Formas ] ---> [ Capa 3: Objetos ] ---> [ Resultado ]
                      └───────────────────────────────┬──────────────────────────────┘
                                          Feature Learning Automático
```

### 2.2 La Metáfora del *Manifold* ("Desarrugar la Bola de Papel")
Siguiendo a François Chollet (2021), los datos de entrada de alta dimensión residen en una variedad diferenciable compleja o *manifold* plegado dentro del espacio vectorial. El objetivo de una red neuronal profunda es aplicar una secuencia de transformaciones geométricas continuas y diferenciables que "desarruguen" dicha bola de papel, convirtiendo fronteras de separación entrelazadas en superficies linealmente separables.

### 2.3 Escalabilidad respecto a Datos y Cómputo
A diferencia de los modelos tradicionales cuyo rendimiento alcanza una meseta (*plateau*) al incrementar el volumen de datos, las redes profundas muestran un escalamiento sostenido en función de tres factores:
1. **Volumen masivo de datos** ($\text{Big Data}$).
2. **Capacidad de cómputo paralelo** (GPU / TPU).
3. **Optimizaciones algorítmicas** (Inicialización de pesos, funciones de activación no saturantes).

---

## 3. Del Perceptrón Simple al Perceptrón Multicapa (MLP)

### 3.1 El Perceptrón de Rosenblatt (1958) y la TLU (*Threshold Logic Unit*)
El Perceptrón es la unidad básica de cómputo inspirada en la neurona biológica. Recibe un vector de entradas $\mathbf{x} = [x_1, x_2, \dots, x_n]^T \in \mathbb{R}^n$, aplica un vector de pesos $\mathbf{w} = [w_1, w_2, \dots, w_n]^T$ y un sesgo (*bias*) $b$, calculando una combinación afín sucedida por una función de escalón de Heaviside:

$$z = \mathbf{w}^T \mathbf{x} + b = \sum_{i=1}^{n} w_i x_i + b$$

$$\hat{y} = \text{step}(z) = \begin{cases} 1 & \text{si } z \ge 0 \\ 0 & \text{si } z < 0 \end{cases}$$

```
   x₁ ───(w₁)───┐
   x₂ ───(w₂)───┼──> [ Σ = wᵀx + b ] ──> [ f(z) Escalón ] ──> ŷ
   xₙ ───(wₙ)───┘
                  ↑
                 (b)
```

### 3.2 La Frontera de Decisión Lineal y la Limitación del XOR
Un Perceptrón individual define un hiperplano de separación lineal en $\mathbb{R}^n$ dado por el conjunto de puntos donde $\mathbf{w}^T \mathbf{x} + b = 0$.

#### El Colapso Histórico: Problema del XOR (Minsky & Papert, 1969)
Las funciones lógicas `AND` y `OR` son linealmente separables. Sin embargo, la función `XOR` (OR Exclusivo) no admite una única línea recta capaz de separar las clases positivos ($1$) de negativos ($0$):

| $x_1$ | $x_2$ | $x_1 \oplus x_2$ |
| :---: | :---: | :---: |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**Demostración de Inviabilidad:**
Para clasificar XOR con un Perceptrón, se requeriría un sistema incoherente de desigualdades:
1. Para $(0,0) \to 0$: $b < 0$
2. Para $(0,1) \to 1$: $w_2 + b \ge 0$
3. Para $(1,0) \to 1$: $w_1 + b \ge 0$
4. Para $(1,1) \to 0$: $w_1 + w_2 + b < 0$

Sumando (2) y (3): $w_1 + w_2 + 2b \ge 0$. Dado que $b < 0$, implica que $w_1 + w_2 + b > w_1 + w_2 + 2b \ge 0$, lo que contradice directamente la condición (4).

### 3.3 Solución: Arquitectura Multilayer Perceptron (MLP)
Al apilar múltiples TLU en capas intermedias (*capas ocultas*), la red transforma el espacio original de entrada en un espacio latente donde el problema se vuelve linealmente separable.

```
Capas:   Entrada (Input)         Oculta (Hidden)           Salida (Output)
            (x₁) ───────────────> (h₁) ─────────┐
                                    ╳           ├───> (ŷ)
            (x₂) ───────────────> (h₂) ─────────┘
```

---

## 4. Anatomía Matemática y Funciones de Activación

Si una red compusiera múltiples capas lineales sin una función de activación no lineal $\sigma(\cdot)$, el modelo colapsaría equivaliendo a una simple transformación afín:

$$W^{(2)} \cdot (W^{(1)} X + b^{(1)}) + b^{(2)} = (W^{(2)} W^{(1)}) X + (W^{(2)} b^{(1)} + b^{(2)}) = W' X + b'$$

Por ende, las **funciones de activación no lineales** son estructuralmente indispensables para dotar a la red de la capacidad de aproximador universal de funciones (Teorema de universalidad de Molar/Hornik).

### 4.1 Sigmoide (Logística)
$$f(z) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

* **Rango:** $(0, 1)$
* **Derivada analítica:**
  $$\frac{d\sigma(z)}{dz} = \sigma(z)(1 - \sigma(z))$$
* **Uso:** Capa de salida para clasificación binaria.
* **Inconveniente:** Sufre de **Desvanecimiento del Gradiente** (*Vanishing Gradient*) para $|z| \gg 0$, donde $f'(z) \to 0$.

### 4.2 Rectified Linear Unit (ReLU)
$$f(z) = \max(0, z)$$

* **Rango:** $[0, \infty)$
* **Derivada analítica:**
  $$f'(z) = \begin{cases} 1 & \text{si } z > 0 \\ 0 & \text{si } z < 0 \end{cases}$$
* **Uso:** Estándar por defecto para capas ocultas.
* **Ventaja:** Cómputo altamente eficiente y gradiente no saturante para valores positivos.
* **Inconveniente:** Problema de neurona muerta (*Dying ReLU*) si el gradiente negativo apaga permanentemente la neurona.

### 4.3 Softmax
Utilizada exclusivamente en la capa de salida para problemas de clasificación multiclase con $K$ categorías mutuamente excluyentes:

$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} \quad \text{para } i = 1, \dots, K$$

* **Propiedad:** Garantiza que $\sum_{i=1}^{K} \text{Softmax}(z_i) = 1$, generando una distribución de probabilidad explícita.

---

## 5. Flujo de Entrenamiento: Forward Pass, Backpropagation y Optimizadores

El proceso de entrenamiento opera en un ciclo de optimización continuo compuesto por 4 pasos ejecutados iterativamente sobre lotes de datos (*mini-batches*).

```
                 ┌─────────────────────────────────────────┐
                 │              Forward Pass               │
                 │   X ──> [Capas] ──> Ŷ ──> Pérdida (L)  │
                 └────────────────────┬────────────────────┘
                                      │
                                      ▼
                 ┌─────────────────────────────────────────┐
                 │           Backward Pass (Autodiff)      │
                 │   ∂L/∂W ◄── [Regla de la Cadena] ◄── L  │
                 └────────────────────┬────────────────────┘
                                      │
                                      ▼
                 ┌─────────────────────────────────────────┐
                 │         Actualización de Pesos          │
                 │        W_nuevo = W_viejo - η · ∂L/∂W    │
                 └─────────────────────────────────────────┘
```

### 5.1 Propagación hacia Adelante (*Forward Pass*)
Dado un lote $X \in \mathbb{R}^{m \times n}$:
1. $Z^{(1)} = X W^{(1)} + b^{(1)}$
2. $A^{(1)} = \text{ReLU}(Z^{(1)})$
3. $Z^{(2)} = A^{(1)} W^{(2)} + b^{(2)}$
4. $\hat{Y} = \text{Softmax}(Z^{(2)})$

### 5.2 Función de Pérdida (*Loss Function*)
Para clasificación multiclase, la medida de divergencia estándar es la **Entropía Cruzada Categórica** (*Categorical Cross-Entropy*):

$$\mathcal{L}(Y, \hat{Y}) = -\frac{1}{m} \sum_{k=1}^{m} \sum_{c=1}^{K} y_{k,c} \log(\hat{y}_{k,c})$$

### 5.3 Retropropagación (*Backpropagation*) mediante la Regla de la Cadena
El objetivo central de la retropropagación es evaluar las derivadas parciales de $\mathcal{L}$ con respecto a cada matriz de pesos $W^{(l)}$ y sesgo $b^{(l)}$.

Considerando un grafo de computación simplificado $z = w \cdot x + b \to a = \sigma(z) \to \mathcal{L}(a, y)$:

Por Regla de la Cadena:
$$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

Donde:
* $\frac{\partial z}{\partial w} = x$
* $\frac{\partial a}{\partial z} = \sigma'(z)$
* $\frac{\partial \mathcal{L}}{\partial a} = \text{Gradiente de la pérdida}$

En forma matricial para una capa $l$:
$$\delta^{(l)} = \frac{\partial \mathcal{L}}{\partial Z^{(l)}} = \left( \delta^{(l+1)} (W^{(l+1)})^T \right) \odot f'^{(l)}(Z^{(l)})$$
$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = (A^{(l-1)})^T \delta^{(l)}$$
$$\frac{\partial \mathcal{L}}{\partial b^{(l)}} = \sum_{i=1}^{m} \delta_{i,:}^{(l)}$$

*(Donde $\odot$ denota el producto de Hadamard u operación elemento a elemento).*

### 5.4 Algoritmos de Optimización
Los optimizadores ajustan los parámetros del modelo en dirección opuesta al gradiente.

#### Descenso de Gradiente Estocástico por Mini-Lotes (Mini-Batch SGD)
$$W_{t+1} = W_t - \eta \cdot \nabla_W \mathcal{L}(W_t)$$
*(Donde $\eta$ representa la Tasa de Aprendizaje / Learning Rate).*

#### Adam (Adaptive Moment Estimation)
Combina las ventajas de Momentum (promedios móviles del gradiente) y RMSprop (promedios móviles de los gradientes al cuadrado):

1. Estimación del primer momento (Inercia):
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
2. Estimación del segundo momento (Escala adaptativa):
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
3. Corrección de sesgo:
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
4. Regla de actualización:
   $$W_{t+1} = W_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
*(Valores típicos por defecto: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$).*

---

## 6. Laboratorio Práctico: Programación en TensorFlow / Keras

En este laboratorio implementaremos en Python un clasificador Multilayer Perceptron (MLP) utilizando la API `tf.keras.Sequential` en nuestro entorno local (VS Code / Swarp).

### Código Fuente de Clase (`mlp_unintep_lab.py`)

```python
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuración de parámetros y reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

print(f"Versión de TensorFlow en uso: {tf.__version__}")

# 2. Carga y preparación del conjunto de datos (MNIST)
mnist = tf.keras.datasets.mnist
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()

# Normalización de los tensores de entrada al rango [0.0, 1.0]
X_train = X_train_raw / 255.0
X_test = X_test_raw / 255.0

print(f"Forma de datos de entrenamiento: {X_train.shape}")
print(f"Forma de datos de prueba: {X_test.shape}")

# 3. Construcción de la Arquitectura Multilayer Perceptron (MLP)
def construir_mlp():
    model = models.Sequential([
        # Capa de aplanamiento: convierte matrices (28, 28) en vectores de 784 entradas
        layers.Flatten(input_shape=(28, 28), name="Capa_Entrada_Flatten"),
        
        # Primera Capa Oculta Densa con activación ReLU
        layers.Dense(128, activation='relu', name="Capa_Oculta_1"),
        
        # Segunda Capa Oculta Densa
        layers.Dense(64, activation='relu', name="Capa_Oculta_2"),
        
        # Capa de Salida Multiclase con activación Softmax (10 dígitos)
        layers.Dense(10, activation='softmax', name="Capa_Salida_Softmax")
    ], name="MLP_UNINTEP_RA2")
    
    return model

modelo_mlp = construir_mlp()

# Muestra el resumen estructural de parámetros entrenables
modelo_mlp.summary()

# 4. Compilación del Modelo
TASA_DE_APRENDIZAJE = 0.001

optimizador_adam = optimizers.Adam(learning_rate=TASA_DE_APRENDIZAJE)

modelo_mlp.compile(
    optimizer=optimizador_adam,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Proceso de Entrenamiento (Forward - Loss - Backpropagation)
TAMANO_LOTE = 64
EPOCAS = 10

print("\nIniciando entrenamiento del modelo...")
historial = modelo_mlp.fit(
    X_train, y_train_raw,
    epochs=EPOCAS,
    batch_size=TAMANO_LOTE,
    validation_split=0.2,
    verbose=1
)

# 6. Evaluación en datos nunca antes vistos (Prueba)
loss_test, accuracy_test = modelo_mlp.evaluate(X_test, y_test_raw, verbose=0)
print(f"\n==========================================")
print(f"Pérdida en Test (Loss): {loss_test:.4f}")
print(f"Precisión en Test (Accuracy): {accuracy_test * 100:.2f}%")
print(f"==========================================")
```

---

## 7. Guía de Trabajo Autónomo (Post-clase)

### 7.1 Experimentos Obligatorios de Modificación de Hiperparámetros
Para consolidar la evidencia del **RA2**, cada estudiante deberá realizar tres experimentos independientes sobre el código base, modificando un hiperparámetro a la vez y registrando el impacto directo sobre la función de pérdida ($\mathcal{L}$).

```
           [ Modelo Base ] ───> Modificar Hiperparámetro
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
      Experimento 1              Experimento 2              Experimento 3
   Profundidad de Capas       Densidad Neuronal         Tasa de Aprendizaje
 (Añadir 2 capas ocultas)   (Reducir a 16 neuronas)   (η = 0.5 vs. η = 0.00001)
```

1. **Experimento 1 (Efecto de la Profundidad):**
   * Añadir 2 capas ocultas adicionales de 128 neuronas.
   * *Pregunta de análisis:* ¿Aumentó el tiempo de entrenamiento? ¿Se evidencia sobreajuste (*Overfitting*) al comparar la pérdida de entrenamiento con la de validación?
2. **Experimento 2 (Efecto de la Capacidad Neuronal):**
   * Modificar las capas ocultas a únicamente 16 neuronas en la primera capa y 8 en la segunda.
   * *Pregunta de análisis:* ¿Ocurre subajuste (*Underfitting*) por falta de capacidad del modelo para representar el manifold de datos?
3. **Experimento 3 (Efecto del Learning Rate $\eta$):**
   * Modificar la tasa de aprendizaje del optimizador Adam a dos extremos:
     * Tasa explosiva: $\eta = 0.5$
     * Tasa infinitamente lenta: $\eta = 0.00001$
   * *Pregunta de análisis:* ¿Por qué una tasa muy alta hace que la pérdida oscile o diverja, mientras que una muy baja estanca la convergencia?

### 7.2 Rúbrica de Evaluación del RA2

| Criterio | Insuficiente (1.0 - 2.9) | Aceptable (3.0 - 3.9) | Sobresaliente (4.0 - 5.0) |
| :--- | :--- | :--- | :--- |
| **Diferenciación ML vs DL** | No logra argumentar conceptualmente el rol del *Feature Engineering* frente al *Representation Learning*. | Describe conceptualmente la diferencia pero omite fundamentos sobre transformaciones no lineales. | Argumenta con precisión matemática y conceptual las ventajas del aprendizaje jerárquico de representaciones. |
| **Implementación en Keras** | El código presentado contiene errores de sintaxis o no compila. | Construye el modelo secuencial pero omite normalización o muestra configuraciones incoherentes de salida. | Programa de forma fluida el modelo MLP, normaliza adecuadamente los tensores y configura el flujo completo de entrenamiento. |
| **Análisis de Hiperparámetros** | Entrega únicamente capturas de pantalla sin análisis ni gráficas de error. | Analiza los experimentos pero confunde conceptos de Overfitting, Underfitting o divergent gradiente. | Presenta un informe técnico con gráficas comparativas de pérdida ($\mathcal{L}$ vs épocas) y concluye rigurosamente el efecto de $\eta$, capas y neuronas. |

---
*UNINTEP Roldanillo - Construyendo futuro a través de la Inteligencia Artificial.*