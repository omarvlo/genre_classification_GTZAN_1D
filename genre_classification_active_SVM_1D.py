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

# División de datos en conjunto inicial y pool
X_initial, X_pool, y_initial, y_pool = train_test_split(
    X_scaled, y, train_size=0.2, random_state=42, stratify=y
)

# Función de aprendizaje activo optimizada
def active_learning(X_pool, y_pool, X_train, y_train, svm, iterations=10, batch_size=50):
    for i in range(iterations):
        # Entrenar el modelo en cada iteración
        svm.fit(X_train, y_train)

        # Predecir probabilidades de las muestras del pool
        y_prob = svm.predict_proba(X_pool)

        # Calcular incertidumbre (menor confianza en predicción)
        uncertainty = 1 - np.max(y_prob, axis=1)

        # Seleccionar muestras más inciertas de cada clase (balanceado)
        uncertain_samples = []
        for cls in np.unique(y_train):
            cls_indices = np.where(y_pool == cls)[0]
            cls_uncertainty = uncertainty[cls_indices]

            # Seleccionar solo una fracción de muestras inciertas por clase
            num_samples_per_class = max(1, batch_size // len(np.unique(y_train)))
            cls_uncertain_samples = cls_indices[np.argsort(cls_uncertainty)[-num_samples_per_class:]]

            uncertain_samples.extend(cls_uncertain_samples)

        uncertain_samples = np.array(uncertain_samples)

        # Agregar las muestras seleccionadas al conjunto de entrenamiento
        X_train = np.vstack((X_train, X_pool[uncertain_samples]))
        y_train = np.hstack((y_train, y_pool[uncertain_samples]))

        # Eliminar las muestras seleccionadas del pool
        X_pool = np.delete(X_pool, uncertain_samples, axis=0)
        y_pool = np.delete(y_pool, uncertain_samples, axis=0)

        # Evaluar en cada iteración para monitorear el aprendizaje
        y_train_pred = svm.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        print(f"Iteración {i + 1}: Accuracy en entrenamiento = {train_acc:.4f}")

        # Verificar balance de clases
        unique, counts = np.unique(y_train, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        print(f"Distribución de clases después de la iteración {i + 1}: {class_distribution}")

    return svm

# Optimización de hiperparámetros con GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

svm = SVC(probability=True, random_state=42)
grid = GridSearchCV(svm, param_grid, refit=True, verbose=2, cv=5, scoring='accuracy')
grid.fit(X_initial, y_initial)

# Usar el mejor modelo encontrado
svm_best = grid.best_estimator_
print("\nMejores parámetros de SVM:", grid.best_params_)

# Aplicar aprendizaje activo optimizado
svm_active_best = active_learning(X_pool, y_pool, X_initial, y_initial, svm_best, iterations=10, batch_size=50)

# Evaluación final del modelo optimizado
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42, stratify=y)
y_pred_best = svm_active_best.predict(X_test)

print("\nResultados finales del modelo SVM con aprendizaje activo (Optimizado):")
print(classification_report(y_test, y_pred_best))
print('Accuracy final:', accuracy_score(y_test, y_pred_best))

