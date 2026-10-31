# pip install pandas numpy matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
sns.set_style("whitegrid")

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

# Paso 1
df.shape
df.info()
df.isnull().sum()
df["sex"].value_counts()
df["race"].value_counts()

# Paso 2
# Ver el impacto de eliminar filas con nulos, ANTES de hacerlo
antes = df["sex"].value_counts(normalize=True)

df_limpio = df.dropna()

despues = df_limpio["sex"].value_counts(normalize=True)

comparacion = pd.DataFrame({"antes": antes, "despues": despues})
print(comparacion)

# Paso 3
def tasa_positiva(data, columna_grupo, columna_objetivo, valor_positivo):
    return (
        data.groupby(columna_grupo)[columna_objetivo]
        .apply(lambda x: (x == valor_positivo).mean())
    )

tasa_por_sexo = tasa_positiva(df_limpio, "sex", "income", ">50K")
tasa_por_raza = tasa_positiva(df_limpio, "race", "income", ">50K")

print(tasa_por_sexo)
print(tasa_por_raza)

disparidad_sexo = tasa_por_sexo.max() - tasa_por_sexo.min()
print(f"Disparidad por sexo: {disparidad_sexo:.3f}")

# Paso 4
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

# Paso 5
# ¿Alguna variable "neutral" correlaciona fuertemente con la variable sensible?
pd.crosstab(df_limpio["occupation"], df_limpio["sex"], normalize="index")
