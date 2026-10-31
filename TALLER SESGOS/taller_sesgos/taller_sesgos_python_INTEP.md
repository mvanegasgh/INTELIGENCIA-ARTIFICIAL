# Taller: Demostración Guiada de Sesgos en Python
### Laboratorio de Scripts de Limpieza de Datos — Curso de Inteligencia Artificial, INTEP

**Modalidad:** guiada + trabajo autónomo
**Duración estimada:** 2 sesiones de 2 horas (4 horas totales)
**Metodología:** socio-constructivista — el docente modela el proceso completo con un dataset de referencia; cada estudiante replica y extiende el análisis sobre la fuente CSV asignada.

---

## 1. Objetivos de aprendizaje

Al finalizar el taller, el estudiante será capaz de:

1. Identificar variables sensibles y proxies de sesgo en un dataset tabular real.
2. Aplicar un flujo de limpieza de datos (missing values, tipos, outliers) sin "ocultar" el sesgo durante el proceso.
3. Calcular métricas simples de disparidad entre grupos (paridad estadística, tasa de positivos por grupo).
4. Visualizar el sesgo detectado con gráficos comparativos.
5. Redactar una reflexión ética/técnica sobre el origen del sesgo y posibles mitigaciones.

---

## 2. Marco conceptual breve (10 min, exposición docente)

Antes de tocar código, se aclaran 3 ideas con el grupo:

- **Sesgo de datos vs. sesgo de modelo:** un dataset puede sobre-representar o sub-representar un grupo (bias de muestreo), o reflejar desigualdades históricas ya existentes (bias histórico), incluso antes de entrenar cualquier modelo.
- **Variable sensible vs. proxy:** género, raza, edad son variables sensibles obvias. Pero variables como código postal, tipo de ocupación o nivel educativo pueden actuar como *proxies* que reintroducen el sesgo aunque se elimine la variable sensible original.
- **Limpiar no es neutralizar:** eliminar filas con "missing values" de un grupo específico, o normalizar de forma distinta por subgrupo, puede *aumentar* el sesgo aunque el script "funcione bien".

---

## 3. Preparación del entorno

```bash
pip install pandas numpy matplotlib seaborn
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
sns.set_style("whitegrid")
```

---

## 4. Fuente de datos propuesta (demostración guiada del docente)

Para el modelado en vivo se propone el dataset **Adult Census Income** (también llamado "Adult" o "Census Income"), del repositorio UCI Machine Learning. Es un clásico de la literatura de *fairness* porque predice si una persona gana más de 50K USD/año a partir de variables demográficas y laborales, y contiene disparidades documentadas por género y raza.

- **Descarga directa (CSV):**
  `https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data`
- **Diccionario de columnas:** age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, income (`<=50K` / `>50K`).
- **Variables sensibles presentes:** `sex`, `race`, `native-country`, `age`.
- **Por qué sirve para este taller:** tiene valores faltantes codificados como `"?"` (no `NaN` estándar, así que obliga a limpiar de verdad), variables categóricas con muchos niveles, y una variable objetivo binaria fácil de usar para medir disparidad de tasas positivas por grupo.

```python
columnas = [
    "age","workclass","fnlwgt","education","education_num",
    "marital_status","occupation","relationship","race","sex",
    "capital_gain","capital_loss","hours_per_week","native_country","income"
]

df = pd.read_csv(
    "adult.data",
    names=columnas,
    na_values=" ?",
    skipinitialspace=True
)
df.head()
```

---

## 5. Fuentes CSV propuestas para el trabajo individual de cada estudiante

Cada estudiante (o pareja) recibe **una fuente distinta** a la usada en la demostración, para evitar resultados idénticos y fomentar comparación en plenaria. Todas cumplen tres criterios: son públicas, tienen al menos una variable sensible clara, y requieren limpieza real (missing values, tipos mixtos o inconsistencias).

| # | Dataset | Variable(s) sensible(s) | Variable objetivo | Fuente |
|---|---------|--------------------------|--------------------|--------|
| 1 | German Credit Data | edad, estado civil/sexo | riesgo crediticio (bueno/malo) | UCI: `statlog+german+credit+data` |
| 2 | COMPAS Recidivism (ProPublica) | raza, sexo | reincidencia (`two_year_recid`) | GitHub `propublica/compas-analysis` |
| 3 | Titanic (Kaggle) | sexo, clase de pasajero | supervivencia | Kaggle `titanic` / `datasciencedojo/titanic` |
| 4 | Heart Disease (UCI Cleveland) | sexo, edad | diagnóstico de enfermedad cardiaca | UCI `heart+disease` |
| 5 | Bank Marketing | edad, estado civil | suscripción a depósito a plazo | UCI `bank+marketing` |
| 6 | Student Performance (Portugal) | sexo, contexto socioeconómico | calificación final (`G3`) | UCI `student+performance` |

> Nota docente: si el grupo es grande, se pueden repetir datasets en subgrupos de 2–3 estudiantes; lo importante es que en la puesta en común se comparen distintos dominios (crédito, justicia, salud, educación) para discutir cómo el sesgo se manifiesta distinto según el contexto.

---

## 6. Flujo guiado paso a paso

### Paso 1 — Exploración inicial (EDA)

```python
df.shape
df.info()
df.isnull().sum()
df["sex"].value_counts()
df["race"].value_counts()
```

**Pregunta guía para el estudiante:** ¿algún grupo demográfico está claramente sub-representado en el número de registros?

### Paso 2 — Limpieza consciente del sesgo

```python
# Ver el impacto de eliminar filas con nulos, ANTES de hacerlo
antes = df["sex"].value_counts(normalize=True)

df_limpio = df.dropna()

despues = df_limpio["sex"].value_counts(normalize=True)

comparacion = pd.DataFrame({"antes": antes, "despues": despues})
print(comparacion)
```

**Punto clave a resaltar en clase:** si `dropna()` cambia notablemente la proporción de un grupo, la limpieza *introdujo* sesgo adicional. Se discute con el grupo si conviene imputar en vez de eliminar, y por qué.

### Paso 3 — Tasa de resultado positivo por grupo (paridad estadística)

```python
def tasa_positiva(data, columna_grupo, columna_objetivo, valor_positivo):
    return (
        data.groupby(columna_grupo)[columna_objetivo]
        .apply(lambda x: (x == valor_positivo).mean())
    )

tasa_por_sexo = tasa_positiva(df_limpio, "sex", "income", ">50K")
tasa_por_raza = tasa_positiva(df_limpio, "race", "income", ">50K")

print(tasa_por_sexo)
print(tasa_por_raza)
```

**Métrica de disparidad (diferencia de paridad estadística):**

```python
disparidad_sexo = tasa_por_sexo.max() - tasa_por_sexo.min()
print(f"Disparidad por sexo: {disparidad_sexo:.3f}")
```

Se explica que valores por encima de ~0.1–0.2 suelen considerarse una señal de alerta en la literatura de *fairness* (el umbral exacto depende del contexto y debe discutirse, no aplicarse mecánicamente).

### Paso 4 — Visualización comparativa

```python
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

tasa_por_sexo.plot(kind="bar", ax=ax[0], color=["#4C72B0", "#DD8452"])
ax[0].set_title("Tasa de ingreso >50K por sexo")
ax[0].set_ylabel("Proporción")

tasa_por_raza.plot(kind="bar", ax=ax[1], color="#55A868")
ax[1].set_title("Tasa de ingreso >50K por raza")
ax[1].set_ylabel("Proporción")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()
```

### Paso 5 — Búsqueda de proxies

```python
# ¿Alguna variable "neutral" correlaciona fuertemente con la variable sensible?
pd.crosstab(df_limpio["occupation"], df_limpio["sex"], normalize="index")
```

**Pregunta guía:** ¿hay ocupaciones dominadas casi por completo por un solo sexo? Si esa variable se usa en un modelo, ¿reintroduce el sesgo que "sex" ya representaba?

### Paso 6 — Reflexión escrita (entregable individual)

Cada estudiante responde, en máximo una página, con base en **su propia fuente asignada**:

1. ¿Qué variable(s) sensible(s) identificó y qué disparidad encontró (con el número exacto de la métrica calculada)?
2. ¿La limpieza aplicada (nulos, duplicados, outliers) alteró la proporción de algún grupo? ¿Cómo lo verificó?
3. ¿Encontró alguna variable proxy? Argumente con evidencia del crosstab o correlación.
4. Proponga una mitigación concreta (no genérica) para su dataset específico.

---

## 7. Rúbrica sugerida

| Criterio | Insuficiente | Aceptable | Sobresaliente |
|---|---|---|---|
| Limpieza documentada | Aplica `dropna()`/`fillna()` sin verificar impacto | Verifica el impacto pero no lo cuantifica | Cuantifica el cambio de proporción antes/después por grupo |
| Métrica de disparidad | No calcula ninguna métrica | Calcula la tasa por grupo | Calcula tasa por grupo y la diferencia de paridad, interpretándola en contexto |
| Detección de proxies | No lo intenta | Identifica una variable candidata sin evidencia | Sustenta el proxy con crosstab o correlación numérica |
| Reflexión | Descriptiva, sin propuesta | Propone mitigación genérica | Propone mitigación específica y justificada para el dominio del dataset |

---

## 8. Cierre en plenaria (20 min)

Se pide a un representante de cada dataset que comparta en una diapositiva o el pizarrón: la disparidad encontrada, el proxy (si lo hubo) y su propuesta de mitigación. El docente cierra comparando cómo el mismo tipo de sesgo (representación desigual) se manifiesta distinto en crédito, justicia, salud y educación — reforzando que la limpieza de datos es una decisión con implicaciones éticas, no solo técnicas.
