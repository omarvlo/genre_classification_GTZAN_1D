# 🎵 Music Genre Classification with active SVM and Tonnetz 🎶

This repository contains scripts for **music genre classification** using **Support Vector Machines (SVM)**, with and without **active learning**. It uses the **GTZAN dataset** and extracts audio features with `librosa`.

## 📂 Repository Contents
- `genre_classification_features_1D.py` → **Extracts audio features** (MFCC, Chroma, Tonnetz, etc.).
- `genre_classification_active_SVM_1D.py` → **SVM classification with Active Learning**.
- `genre_classification_classic_SVM_1D.py` → **SVM classification (without Active Learning)**.

---

## 📥 Installation and Usage

### 1️⃣ **Install Dependencies**
Ensure you have Python installed along with the required dependencies:

```bash
pip install numpy pandas scikit-learn librosa tqdm requests
```

---

### 2️⃣ **Extract Audio Features**
Run the following script to extract features from the audio files:

```bash
python genre_classification_features_1D.py
```
This will generate the `audio_features_1D.csv` file containing the extracted features from the **GTZAN dataset**.

---

### 3️⃣ **Train SVM Models**
#### **🔹 SVM with Active Learning**
```bash
python genre_classification_active_SVM_1D.py
```
This script implements an **SVM model with active learning**, iteratively selecting uncertain samples to improve training.

#### **🔹 Classic SVM (without Active Learning)**
```bash
python genre_classification_classic_SVM_1D.py
```
This script trains a **traditional SVM** using the extracted features without active learning.

---

## 📊 Expected Results
The trained models will generate **classification reports** and **accuracy metrics**, helping to evaluate the best approach for music genre classification.

Example expected output:
```
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

    accuracy                           0.84       330
   macro avg       0.84      0.84      0.83       330
weighted avg       0.84      0.84      0.83       330

Accuracy final: 0.8363636363636363
```

---

## 📜 References and Resources
- 📄 Enhancing Music Genre Classification Using Tonnetz and Active Learning (2024) (https://marsyas.info/download/data_sets/)
- 📁 **GTZAN Dataset**: [GTZAN Dataset](https://marsyas.info/download/data_sets/)
- 📚 **Libraries Used**:
  - [`librosa`](https://librosa.org/) → Audio feature extraction.
  - [`scikit-learn`](https://scikit-learn.org/) → Machine Learning models.
  - [`numpy`](https://numpy.org/) → Numerical operations.
  - [`pandas`](https://pandas.pydata.org/) → Data handling.
  - [`tqdm`](https://tqdm.github.io/) → Progress bars.

---

## 📌 Contributing
If you'd like to improve the code or add new functionalities, feel free to fork the repository and submit a pull request.

---

## 📌 Contact
For questions or suggestions, feel free to reach out on GitHub.

🚀 **Thank you for visiting this repository!** 🎶🔥
