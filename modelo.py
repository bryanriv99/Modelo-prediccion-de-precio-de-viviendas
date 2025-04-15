import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Cargar el archivo CSV
df = pd.read_csv('casas.csv')

# Mostrar informacion sobre la cantidad de datos
print("Informacion del dataset:")
print(f"Numero total de registros: {df.shape[0]}")
print(f"Numero de caracteristicas: {df.shape[1]}")
print("\nColumnas del dataset:")
print(df.columns.tolist())

# Ver las primeras filas
print("\nDatos originales:")
print(df.head())

# Eliminar columna 'No' (identificador que no aporta al modelo)
df = df.drop(columns=["No"])

# Separar variables de entrada (X) y variable objetivo (y)
X = df.drop(columns=["Y house price of unit area"])  # Variables predictoras
y = df["Y house price of unit area"]                 # Variable a predecir

# Dividir los datos en conjunto de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mostrar tamano de los conjuntos de entrenamiento y prueba
print(f"\nDivision de datos:")
print(f"Datos de entrenamiento: {X_train.shape[0]} registros ({X_train.shape[0]/df.shape[0]*100:.1f}%)")
print(f"Datos de prueba: {X_test.shape[0]} registros ({X_test.shape[0]/df.shape[0]*100:.1f}%)")

# Crear el modelo de regresion lineal y entrenarlo
model = LinearRegression()
model.fit(X_train, y_train)

# Hacer predicciones con el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nEvaluacion del modelo:")
print(f"RMSE (Error cuadratico medio): {rmse:.2f}")
print(f"R^2 (Coeficiente de determinacion): {r2:.2f}")

# Predecir el precio de una nueva casa
# Modificar segun los valores de la casa a predecir
nueva_casa = pd.DataFrame([[2013, 20, 250, 8, 24.98, 121.54]], columns=X.columns)
prediccion = model.predict(nueva_casa)
print(f"\nPrediccion para una nueva casa: {prediccion[0]:.2f} unidades")

# Visualizar la importancia de las variables (coeficientes)
coef = model.coef_
features = X.columns

plt.figure(figsize=(10,6))
plt.barh(features, coef)
plt.title("Importancia de las variables")
plt.xlabel("Peso del coeficiente")
plt.grid(True)
plt.show()