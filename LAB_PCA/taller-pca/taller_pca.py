import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()
x = cancer.data
y = cancer.target
feature_names = cancer.feature_names
target_names = cancer.target_names

# --- Exploración (agregado didáctico) ---
print("Forma de x:", x.shape)
print("Clases:", target_names)
print("Primeras 5 variables:", feature_names[:5])

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


print("Medias después de escalar:", x_scaled.mean(axis=0).round(2)[:5])
print("Desv. estándar después de escalar:", x_scaled.std(axis=0).round(2)[:5])


pca = PCA()
pca.fit(x_scaled)

varianza_explicada = pca.explained_variance_ratio_
varianza_acumulada = varianza_explicada.cumsum()

plt.plot(range(1, 31), varianza_acumulada, marker="o")
plt.axhline(y=0.85, color="r", linestyle="--")
plt.xlabel("Número de componentes")
plt.ylabel("Varianza acumulada")
plt.grid(True)
plt.show()


pca = PCA(n_components=6)
x_pca = pca.fit_transform(x_scaled)

# --- Comprobación ---
print("Forma original:", x_scaled.shape)
print("Forma reducida:", x_pca.shape)


valores_propios = pca.explained_variance_
vectores_propios = pca.components_

print(valores_propios)
print(vectores_propios)


pesos_pc1 = pd.Series(vectores_propios[0], index=feature_names)
print(pesos_pc1.abs().sort_values(ascending=False).head(3))


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score


x_train, x_test, y_train, y_test = train_test_split(
    x_pca, y, test_size=0.3, random_state=42
)

tree = DecisionTreeClassifier(max_depth=2, random_state=42)
tree.fit(x_train, y_train)

pred_tree = tree.predict(x_test)
print("Exactitud:", accuracy_score(y_test, pred_tree))

cm = confusion_matrix(y_test, pred_tree)
disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
disp.plot()
plt.show()


plt.figure(figsize=(10, 6))
plot_tree(
    tree,
    class_names=list(target_names),
    feature_names=[f"PC{i}" for i in range(1, 7)],
    filled=True,
)
plt.show()


