
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import GroupKFold

# Cargar datos
data = pd.read_csv("F:/orthesis_classification/data/features/emg_features_labeled_zscore.csv")
X = data.select_dtypes(include=[np.number])
y = data["label"]

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Entrenar modelo
clf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, min_samples_split=10, random_state=42)
clf.fit(X_train, y_train)
print("✅ Entrenamiento exitoso")

# Predecir
y_pred = clf.predict(X_test)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Graficar matriz de confusión
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion matrix, without normalization")
plt.show()

# Calcular precisión
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ La precisión del modelo es de {accuracy * 100:.2f}%")

print("\n📊 Reporte de clasificación:")
print(classification_report(y_test, y_pred))

print("Distribución de clases:")
print(y.value_counts(normalize=True))

train_pred = clf.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, y_pred)

print(f"🎯 Precisión en entrenamiento: {train_acc*100:.2f}%")
print(f"🧪 Precisión en prueba: {test_acc*100:.2f}%")
