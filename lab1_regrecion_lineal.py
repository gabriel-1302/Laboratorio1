import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


#Leyendo el dataset
df = pd.read_csv('data/winequality-red.csv', sep=';')

pd.set_option('display.max_columns', None)
#print(df)

#Normalizacion de los datos
# Select 11 features
feature_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH','sulphates','alcohol']
X = df[feature_cols].values
y = df['quality'].values

def normalize_features(X):
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    normalized_X = (X - means) / stds
    return normalized_X, means, stds

X_norm, means, stds = normalize_features(X)
print("Primeras 5 filas de X_norm:")
print(X_norm[:5, :])
X_norm = np.column_stack((np.ones(X_norm.shape[0]), X_norm))
print("\nPrimeras 5 filas de X_norm con unos:")
print(X_norm[:5, :])

# print("media antes de normalizar:", means)
# print("desviacion antes de normalizar:", stds)
# print("media despues de normalizar:", np.mean(X_norm, axis=0))
# print("desviacion despues de normalizar:", np.std(X_norm, axis=0))

#Normalizacion de los datos
n_features = X_norm.shape[1]

theta = np.zeros(n_features)



#Funcion de costo
def compute_cost(X, y, theta):
    """
    calcular costo para regresion lineal con variable theta
    """
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions - y))
    return cost

#Funcion de desenso de gradiente
iterations = 5000
alpha = 0.01
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        prediction = X.dot(theta)
        theta = theta - (alpha/m) * (X.T.dot(prediction - y))
        cost_history.append(compute_cost(X, y, theta))

    return theta, cost_history
theta_gd, cost_history = gradient_descent(X_norm, y, theta, alpha, iterations)

#funcion para predecir
def predict(features, theta, means, stds):
    # Normalize features
    features_norm = (features - means) / stds
    # Add intercept term
    features_norm = np.column_stack((np.ones(features_norm.shape[0]), features_norm))
    # Make prediction
    return features_norm.dot(theta)


sample_wine = np.array([[7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51,0.84, 9.4]])
predicted_quality = predict(sample_wine, theta_gd, means, stds)

print(f"Predicted quality: {predicted_quality[0]:.2f}")





# Save theta_gd to a CSV file
theta_df = pd.DataFrame(theta_gd, columns=['Theta'])
theta_df.to_csv('variables/theta_output.csv', index=False)

# Save cost_history to a CSV file
cost_history_df = pd.DataFrame(cost_history, columns=['Cost History'])
cost_history_df.to_csv('variables/cost_history_output.csv', index=False)


#dibuja el decenso
plt.figure(figsize=(10, 6))
plt.plot(range(iterations), cost_history)
plt.xlabel('Iteraciones')
plt.ylabel('Costo')
plt.savefig('graficas/desenso/desenso_2d.png')  # Save the plot to a specific path
plt.close()  # Close the plot to free up memory


# Assuming you have X_norm and y
# Define the range of theta0 and theta1 values
theta0_vals = np.linspace(-5, 5, 100)
theta1_vals = np.linspace(-5, 5, 100)

# Create a meshgrid
THETA0, THETA1 = np.meshgrid(theta0_vals, theta1_vals)

# Initialize a matrix to store cost values
Z = np.zeros(THETA0.shape)

# Calculate cost for each combination of theta0 and theta1
for i in range(THETA0.shape[0]):
    for j in range(THETA0.shape[1]):
        theta_temp = np.array([THETA0[i, j], THETA1[i, j]])
        Z[i, j] = compute_cost(X_norm[:, 0:2], y, theta_temp) # here I am using only 2 features, if you use more you have to slice accordingly


# Plot the 3D surface
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(THETA0, THETA1, Z, cmap=cm.viridis, alpha=0.8)
ax.set_xlabel('Theta 0')
ax.set_ylabel('Theta 1')
ax.set_zlabel('Cost')
ax.set_title('Cost Surface')
plt.savefig('graficas/desenso/desenso_3d.png')  # Save the plot to a specific path
plt.close()





# Ruta donde deseas guardar las imágenes
output_dir = "graficas/grafico_por_columna"  # Reemplaza con la ruta deseada

# Crear la carpeta si no existe
os.makedirs(output_dir, exist_ok=True)

# Lista de columnas a graficar contra 'quality'
columns_to_plot = [col for col in df.columns if col != 'quality']

# Generar y guardar cada gráfico
for i, col in enumerate(columns_to_plot, start=1):
    plt.figure(figsize=(8, 4))
    plt.scatter(df[col], df['quality'], alpha=0.5, color='r')
    plt.title(f'Scatter plot of {col} vs Quality')
    plt.xlabel(col)
    plt.ylabel('Quality')

    # Guardar la imagen
    file_path = os.path.join(output_dir, f"grafico{i}.png")
    plt.savefig(file_path)
    plt.close()  # Cerrar la figura para liberar memoria











# Ruta donde deseas guardar las imágenes
output_dir = "graficas/grafico_comparativo_de_columas_de_x"  # Reemplaza con la ruta deseada

# Crear la carpeta si no existe
os.makedirs(output_dir, exist_ok=True)

# Selecciona el índice de la primera columna (índice 0), que es "fixed acidity"
first_feature_index = 0

# Itera sobre el resto de las columnas para crear un gráfico para cada una
for second_feature_index in range(1, X_norm.shape[1]):
    if second_feature_index == X_norm.shape[1]:  # Evitar que se grafique la columna "quality"
        continue
    
    # 1. Fija los valores del resto de características en su valor medio
    other_features_values = means.copy()
    
    # 2. Crea una malla de puntos para las dos características seleccionadas
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_range, y_range)
    
    # 3. Calcula las predicciones del modelo para cada punto de la malla
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            features = other_features_values.copy()
            features[first_feature_index] = X[i, j]
            features[second_feature_index - 1] = Y[i, j]  # Subtract 1 from second_feature_index
            
            # Normaliza las características (excepto la intercepción)
            features_norm = (features - means) / stds
            # Agrega la intercepción
            features_norm = np.insert(features_norm, 0, 1)
            # Calcula la predicción
            Z[i, j] = features_norm.dot(theta_gd)
    
    # 4. Crea el gráfico 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Grafica la superficie del plano de regresión
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    
    # Grafica los puntos de datos
    ax.scatter(X_norm[:, first_feature_index + 1], X_norm[:, second_feature_index], y, color='r', marker='o', s=20)
    
    # Asigna etiquetas a los ejes
    ax.set_xlabel(df.columns[first_feature_index])
    ax.set_ylabel(df.columns[second_feature_index])
    ax.set_zlabel('Calidad')
    ax.set_title(f'Regresión de calidad con {df.columns[first_feature_index]} vs {df.columns[second_feature_index]}')
    
    # Guardar la imagen
    file_path = os.path.join(output_dir, f"grafico3D_{second_feature_index}.png")
    plt.savefig(file_path)
    plt.close()  # Cerrar la figura para liberar memoria
