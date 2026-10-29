# Guía Práctica de Clase: Regresión Lineal con Python y `uv`

**Asignatura:** Inteligencia Artificial  
**Tema:** Regresión Lineal Supervisada — Implementación en Entorno Local  
**Gestor de Proyecto:** `uv` (Astral)  
**Dataset:** `student-mat.csv` (Rendimiento Académico)  

---

## 1. Configuración del Entorno de Trabajo con `uv`

Para realizar este ejercicio sin depender de entornos Jupyter (como Google Colab) y trabajar de manera profesional con proyectos reproducibles en Python, utilizaremos **`uv`**, la herramienta rápida de gestión de paquetes y entornos virtuales.

### Pasos iniciales en la terminal de la clase:

```bash
# 1. Crear la carpeta del proyecto e inicializarlo
uv init ejecucion_regresion_estudiantes
cd ejecucion_regresion_estudiantes

# 2. Agregar las dependencias necesarias
uv add pandas numpy matplotlib seaborn scikit-learn

# 3. Ubicar el dataset 'student-mat.csv' dentro de la carpeta del proyecto
# Asegurarse de que el archivo 'student-mat.csv' esté en la raíz del proyecto.
```

Estructura final del proyecto:
```
ejecucion_regresion_estudiantes/
├── .venv/
├── pyproject.toml
├── uv.lock
├── student-mat.csv
└── main.py
```

---

## 2. Código Completo del Proyecto (`main.py`)

A continuación se presenta el archivo `main.py` estructurado secuencialmente con salidas legibles por consola y almacenamiento automático de gráficos.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    print("=" * 60)
    print(" PASO 1: Carga e Inspección Inicial de Datos ")
    print("=" * 60)
    
    # Cargar el dataset especificando el separador ';'
    try:
        data = pd.read_csv('student-mat.csv', sep=';')
        print("Dataset cargado exitosamente.")
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'student-mat.csv'. Verifique la ubicación.")
        return

    print(f"\nDimensiones iniciales del dataset: {data.shape[0]} filas, {data.shape[1]} columnas.\n")
    print("--- Primeras 5 filas ---")
    print(data.head())
    
    print("\n--- Tipos de Datos por Columna ---")
    print(data.dtypes)

    print("\n" + "=" * 60)
    print(" PASO 2: Transformación de Variables Categóricas ")
    print("=" * 60)
    
    # Mapear la variable categórica famsize de ('LE3', 'GT3') a (0, 1)
    mapa_tamanho_familia = {'LE3': 0, 'GT3': 1}
    data_modificada = data.copy()
    data_modificada['famsize'] = data_modificada['famsize'].map(mapa_tamanho_familia)
    
    print("Mapeo realizado en columna 'famsize' ('LE3': 0, 'GT3': 1).")
    print(data_modificada[['famsize']].head())

    print("\n" + "=" * 60)
    print(" PASO 3: Análisis Estadístico y Limpieza de Datos ")
    print("=" * 60)
    
    print("--- Resumen Estadístico de Variables Numéricas ---")
    print(data_modificada.describe())
    
    print("\n--- Verificación de Valores Nulos ---")
    nulos = data_modificada.isnull().sum()
    print(nulos[nulos > 0] if nulos.sum() > 0 else "No existen valores nulos en el dataset.")

    # Detección y filtrado de datos atípicos (outliers) mediante IQR
    Q1 = data_modificada.quantile(0.25, numeric_only=True)
    Q3 = data_modificada.quantile(0.75, numeric_only=True)
    IQR = Q3 - Q1
    
    # Filtrar el DataFrame excluyendo registros con atípicos en variables numéricas
    num_cols = data_modificada.select_dtypes(include='number').columns
    condicion_atipicos = ((data_modificada[num_cols] < (Q1 - 1.5 * IQR)) | 
                          (data_modificada[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    
    data_sin_atipicos = data_modificada[~condicion_atipicos].copy()
    print(f"\nDimensiones tras filtrar atípicos (IQR): {data_sin_atipicos.shape[0]} filas.")

    print("\n" + "=" * 60)
    print(" PASO 4: Matriz de Correlación ")
    print("=" * 60)
    
    matriz_correlacion = data_sin_atipicos.corr(numeric_only=True)
    print("Top 5 correlaciones con la nota final (G3):")
    print(matriz_correlacion['G3'].sort_values(ascending=False).head(6))

    # Guardar heatmap como imagen
    plt.figure(figsize=(12, 10))
    sns.heatmap(matriz_correlacion, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("Matriz de Correlación - Rendimiento de Estudiantes")
    plt.tight_layout()
    plt.savefig('matriz_correlacion.png')
    plt.close()
    print("\nGráfico 'matriz_correlacion.png' guardado con éxito.")

    print("\n" + "=" * 60)
    print(" PASO 5: Construcción y Entrenamiento del Modelo de Regresión Lineal ")
    print("=" * 60)
    
    # Para predecir la nota final G3, utilizaremos las notas previas (G1, G2) y tiempo de estudio
    caracteristicas = ['G1', 'G2', 'studytime', 'failures', 'absences']
    variable_objetivo = 'G3'

    X = data_sin_atipicos[caracteristicas]
    y = data_sin_atipicos[variable_objetivo]

    # División en conjuntos de Entrenamiento (80%) y Prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instanciar e integrar el modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    print("Modelo entrenado exitosamente.")
    print(f"Intercepto ($w_0$): {modelo.intercept_:.4f}")
    print("Coeficientes ($w_i$):")
    for col, coef in zip(caracteristicas, modelo.coef_):
        print(f"  - {col}: {coef:.4f}")

    print("\n" + "=" * 60)
    print(" PASO 6: Evaluación del Modelo ")
    print("=" * 60)
    
    y_pred = modelo.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Error Cuadrático Medio (MSE): {mse:.4f}")
    print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}")
    print(f"Coeficiente de Determinación ($R^2$): {r2:.4f}")

    # Gráfica de Valores Reales vs Predichos
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='indigo', alpha=0.7, edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Notas Reales (G3)")
    plt.ylabel("Notas Predichas (G3)")
    plt.title("Predicción de Regresión Lineal: Reales vs Predichos")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('predicciones_vs_reales.png')
    plt.close()
    print("\nGráfico 'predicciones_vs_reales.png' guardado con éxito.")
    print("\nProceso finalizado.")

if __name__ == "__main__":
    main()
```

---

## 3. Explicación Detallada del Código Paso a Paso (Guion de Clase)

Durante la sesión, el docente irá escribiendo y explicando cada bloque del script con los estudiantes en sus editores de código (VS Code, Cursor, etc.) ejecutando el programa mediante:

```bash
uv run main.py
```

---

### Bloque 1: Carga e Inspección de Datos

```python
data = pd.read_csv('student-mat.csv', sep=';')
```
* **¿Por qué `sep=';'`?**  
  A diferencia de los archivos CSV estándar separados por comas, el dataset de rendimiento escolar de la UCI utiliza punto y coma (`;`). Si omitimos este parámetro, Pandas cargará todo el registro en una sola columna.

* **Inspección de Tipos de Datos (`dtypes`):**  
  Explique a los alumnos que los modelos matemáticos en `scikit-learn` únicamente aceptan valores numéricos (`int64`, `float64`). Las columnas etiquetadas como `object` corresponden a cadenas de texto o variables categóricas que deben ser transformadas.

---

### Bloque 2: Mapeo y Transformación de Variables Categóricas

```python
mapa_tamanho_familia = {'LE3': 0, 'GT3': 1}
data_modificada['famsize'] = data_modificada['famsize'].map(mapa_tamanho_familia)
```
* **Propósito Pedagógico:**  
  La columna `famsize` representa el tamaño de la familia del estudiante:
  * `LE3`: Less or Equal to 3 (Menor o igual a 3 miembros) $\rightarrow 0$.
  * `GT3`: Greater Than 3 (Mayor a 3 miembros) $\rightarrow 1$.
* **Método `.map()`:**  
  Convierte directamente los valores de texto en una representación binaria numérica utilizable por el modelo de regresión.

---

### Bloque 3: Tratamiento de Valores Atípicos (Outliers) mediante Rango Intercuartílico (IQR)

```python
Q1 = data_modificada.quantile(0.25, numeric_only=True)
Q3 = data_modificada.quantile(0.75, numeric_only=True)
IQR = Q3 - Q1

condicion_atipicos = ((data_modificada[num_cols] < (Q1 - 1.5 * IQR)) | 
                      (data_modificada[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)

data_sin_atipicos = data_modificada[~condicion_atipicos].copy()
```

* **Explicación Matemática:**  
  Definimos el Rango Intercuartílico como:
  
  $$IQR = Q_3 - Q_1$$
  
  Cualquier observación fuera del intervalo $[Q_1 - 1.5 \times IQR, Q_3 + 1.5 \times IQR]$ se considera un valor atípico.
* **Impacto en la Regresión Lineal:**  
  Como la Regresión Lineal minimiza el Error Cuadrático Medio ($MSE$), los valores atípicos se elevan al cuadrado, ejerciendo una palanca (fuerza desproporcionada) que distorsiona la pendiente e intercepto del hiperplano de ajuste.

---

### Bloque 4: Matriz de Correlación

```python
matriz_correlacion = data_sin_atipicos.corr(numeric_only=True)
```
* **Selección de Características:**  
  La matriz de correlación de Pearson mide el grado de asociación lineal entre pares de variables continuas entre $-1$ y $+1$.
* **Visualización en Modo Script:**  
  En un script ejecutable sin entorno gráfico interactivo de Jupyter, utilizamos `plt.savefig('matriz_correlacion.png')` seguido de `plt.close()` para liberar memoria y guardar la figura en disco.

---

### Bloque 5: Ajuste del Modelo ($Y = \mathbf{w}^T \mathbf{X} + w_0$)

```python
caracteristicas = ['G1', 'G2', 'studytime', 'failures', 'absences']
variable_objetivo = 'G3'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)
```

* **Mapeo de Variables:**
  * **$X$ (Matriz de Atributos):** Incluye notas del primer período (`G1`), segundo período (`G2`), tiempo de estudio (`studytime`), reprobaciones previas (`failures`) y ausencias (`absences`).
  * **$y$ (Variable Objetivo):** Calificación final del año académico (`G3`).
* **División Entrenamiento/Prueba:**  
  Separamos el $80\%$ de las observaciones para calibrar los pesos $\mathbf{w}$ y reservamos el $20\%$ restante para evaluar el nivel de generalización del modelo frente a datos desconocidos.

---

### Bloque 6: Evaluación Cuantitativa del Modelo

```python
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
```

1. **Error Cuadrático Medio ($MSE$):**
   
   $$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$
   

2. **Raíz del Error Cuadrático Medio ($RMSE$):**
   
   $$RMSE = \sqrt{MSE}$$
   
   Expresa la desviación promedio de las predicciones en las mismas unidades de la nota (`0-20` puntos).

3. **Coeficiente de Determinación ($R^2$):**
   
   $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
   
   Indica qué porcentaje de la variabilidad total de la nota final `G3` es explicada por las variables seleccionadas.

---

## 4. Guía de Ejecución para los Estudiantes

1. Asegurarse de tener el entorno sincronizado con `uv`:
   ```bash
   uv sync
   ```
2. Ejecutar el script desde la consola:
   ```bash
   uv run main.py
   ```
3. Verificar los resultados impresos en la consola y las imágenes generadas (`matriz_correlacion.png` y `predicciones_vs_reales.png`).