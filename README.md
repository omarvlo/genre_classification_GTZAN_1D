# 🎵 Clasificación de Géneros Musicales con SVM Activo y Tonnetz 🎶

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18q8wi934kTNgaxbL-d3UNAWY79lbKryh?usp=sharing)

Este repositorio implementa **clasificación de géneros musicales** usando **Support Vector Machines (SVM)**, comparando un enfoque **clásico** contra una estrategia con **Aprendizaje Activo**.  
Las características de audio se extraen utilizando **MFCC, Chroma, características espectrales y Tonnetz** a partir del **dataset GTZAN**.

> Este proyecto está basado en el artículo:  
> *Enhancing Music Genre Classification Using Tonnetz and Active Learning (2024)*

---

## Pruébalo al instante en Google Colab (sin instalar nada)

Puedes ejecutar **todo el pipeline** directamente en Google Colab sin instalar dependencias en tu máquina.

👉 **Abrir el notebook aquí:**  
https://colab.research.google.com/drive/18q8wi934kTNgaxbL-d3UNAWY79lbKryh?usp=sharing

El notebook automáticamente:

- Descarga el **dataset GTZAN**
- Instala todas las librerías necesarias
- Extrae las características de audio
- Entrena el **SVM Clásico**
- Entrena el **SVM con Aprendizaje Activo**
- Muestra las métricas finales de evaluación

Ideal para **reproducir los resultados rápidamente**.

---

## 📂 Contenido del Repositorio

| Archivo | Descripción |
|---------|-------------|
| `genre_classification_features_1D.py` | Extrae características de audio (MFCC, Chroma, Tonnetz, espectrales) |
| `genre_classification_active_SVM_1D.py` | Clasificación SVM con **Aprendizaje Activo** |
| `genre_classification_classic_SVM_1D.py` | Clasificación SVM tradicional |

---

##  Instalación y Uso Local

### 1️⃣ Instalar dependencias

Asegúrate de tener Python 3.9+ instalado.

```bash
pip install numpy pandas scikit-learn librosa tqdm requests
```

2️⃣ Extraer características de audio

```bash
python genre_classification_features_1D.py
```

Esto genera el archivo:
```bash
audio_features_1D.csv
```
con las características extraídas del dataset GTZAN.

3️⃣ Entrenar los modelos
🔹 SVM con Aprendizaje Activo

```bash
python genre_classification_active_SVM_1D.py
```

Este modelo selecciona iterativamente las muestras más inciertas para mejorar el entrenamiento.

🔹 SVM Clásico

```bash
python genre_classification_classic_SVM_1D.py
```

Entrena un SVM tradicional sin aprendizaje activo.

📊 Resultados Esperados

Ambos scripts generan reportes de clasificación y métricas de accuracy.

Ejemplo de salida del SVM con Aprendizaje Activo:
```bash
Resultados finales del modelo SVM con aprendizaje activo (Optimizado):

              precision    recall  f1-score   support
       blues       0.79      0.94      0.86        33
   classical       0.92      1.00      0.96        33
     country       0.79      0.79      0.79        33
       disco       0.83      0.73      0.77        33
      hiphop       0.78      0.85      0.81        33
        jazz       0.97      0.94      0.95        33
       metal       0.78      0.97      0.86        33
         pop       0.88      0.85      0.86        33
      reggae       0.86      0.76      0.81        33
        rock       0.78      0.55      0.64        33

Accuracy final: 0.8363
```
🎼 Características de Audio Utilizadas

MFCC (Mel Frequency Cepstral Coefficients)
Chroma Features
Spectral Contrast
Zero Crossing Rate
Tonnetz (Red Armónica)

Estas características capturan tanto el timbre como la estructura armónica de la música.

## 📜 Referencias y Recursos

📄 Artículo de referencia
Enhancing Music Genre Classification Using Tonnetz and Active Learning (2024)
https://www.rcs.cic.ipn.mx/2024_153_11/Enhancing%20Music%20Genre%20Classification%20Using%20Tonnetz%20and%20Active%20Learning.pdf

📁 GTZAN Dataset:
https://huggingface.co/datasets/marsyas/gtzan

📚 Librerías utilizadas

librosa → Extracción de características de audio
scikit-learn → Modelos de Machine Learning
numpy → Operaciones numéricas
pandas → Manejo de datos
tqdm → Barras de progreso

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas!
Puedes hacer fork del repositorio y enviar un pull request.

Contacto: Para dudas o sugerencias, abre un issue en GitHub.

¡Gracias por visitar este repositorio! 🎶🔥
