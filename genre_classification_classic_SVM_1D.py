import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Función para cargar datos desde CSV
def load_features_from_csv(file_path="audio_features_1D.csv"):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values  # Todas las columnas excepto la última (características)
    y = df.iloc[:, -1].values   # Última columna (etiquetas)
    return X, y

# Cargar los datos desde el archivo CSV
X, y = load_features_from_csv()

# Imprimir dimensiones de los datos
print(f"Dimensiones de X (Características): {X.shape}")
print(f"Dimensiones de y (Etiquetas): {y.shape}")

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División de datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Optimización de hiperparámetros con GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

svm = SVC(probability=True, random_state=42)
grid = GridSearchCV(svm, param_grid, refit=True, verbose=2, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# Mejor modelo encontrado
svm_best = grid.best_estimator_
print("\nMejores parámetros de SVM:", grid.best_params_)

# Entrenar el modelo final en el conjunto de entrenamiento
svm_best.fit(X_train, y_train)

# Evaluación final del modelo en el conjunto de prueba
y_pred_best = svm_best.predict(X_test)

print("\nResultados finales del modelo SVM sin aprendizaje activo:")
print(classification_report(y_test, y_pred_best))
print('Accuracy final:', accuracy_score(y_test, y_pred_best))
