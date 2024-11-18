import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cargar el dataset
lechuzas_data = pd.read_csv("C:/Users/pedro/Desktop/IANAIVE/BD/lechuza (2).csv")

# Seleccionar características y variable objetivo
X = lechuzas_data[['Radiacion', 'Temperatura', 'Temperatura panel']].values
y = lechuzas_data['Potencia'].values

# Discretizar la variable objetivo (Potencia) en categorías
discretizer = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')  # Dividir en 3 categorías
y_discretized = discretizer.fit_transform(y.reshape(-1, 1)).flatten()

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_discretized, test_size=0.2, random_state=42)

# Crear e instanciar el modelo Naive Bayes
model = GaussianNB()

# Entrenar el modelo
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))
print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))

# Opcional: Decodificar las categorías para entenderlas mejor
categories = discretizer.bin_edges_[0]
print("\nCategorías de Potencia:")
for i in range(len(categories) - 1):
    print(f"Categoría {i}: {categories[i]:.2f} - {categories[i+1]:.2f}")
