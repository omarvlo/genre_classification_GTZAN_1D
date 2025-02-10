# -*- coding: utf-8 -*-
"""Audio Feature Extraction Script with MFCC, Spectral, Chroma, and Tonnetz Features"""

import os
import glob
import requests
import numpy as np
import pandas as pd
import librosa
import tarfile
from tqdm import tqdm
import warnings

# Constants
DATASET_PATH = "gtzan_dataset/genres"
TAR_GZ_URL = "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz"
TAR_GZ_PATH = os.path.join("gtzan_dataset", "genres.tar.gz")

# Function to download and extract the dataset if not already present
def download_and_extract_dataset():
    if not os.path.exists(DATASET_PATH) or len(os.listdir(DATASET_PATH)) == 0:
        os.makedirs("gtzan_dataset", exist_ok=True)

        print(f"Dataset not found at {DATASET_PATH}. Downloading...")
        response = requests.get(TAR_GZ_URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(TAR_GZ_PATH, "wb") as file, tqdm(
            desc="Downloading genres.tar.gz",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                progress_bar.update(len(chunk))

        print("Extracting the dataset...")
        with tarfile.open(TAR_GZ_PATH, "r:gz") as tar:
            tar.extractall(path="gtzan_dataset")

        os.remove(TAR_GZ_PATH)
        print(f"Dataset downloaded and extracted in {DATASET_PATH}")

# Function to extract MFCCs and additional features from an audio file
def extract_features(file_path, n_mfcc=20):
    signal, sr = librosa.load(file_path, sr=None)

    # MFCC Features
    mfcc_features = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = mfcc_features.mean(axis=1)
    mfcc_var = mfcc_features.var(axis=1)

    # Start with MFCC base features
    features = np.concatenate([mfcc_mean, mfcc_var])
    
    # Spectral and Energy Features
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr).mean()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr).mean()
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=signal).mean()
    rmse = librosa.feature.rms(y=signal).mean()

    spectral_energy_features = np.array([
        spectral_centroid, spectral_bandwidth, spectral_rolloff,
        zero_crossing_rate, rmse
    ])

    features = np.concatenate([features, spectral_energy_features])
    
    # Chroma and Tonnetz Features
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_var = chroma.var(axis=1)

    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(signal), sr=sr)
    tonnetz_mean = tonnetz.mean(axis=1)
    tonnetz_var = tonnetz.var(axis=1)

    chroma_tonnetz_features = np.concatenate([chroma_mean, chroma_var, tonnetz_mean, tonnetz_var])
    
    # # Tempo Feature (Commented out)
    # onset_env = librosa.onset.onset_strength(y=signal, sr=sr)
    # tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    # features = np.concatenate([features, [tempo]])
    
    features = np.concatenate([features, chroma_tonnetz_features])

    return features

# Function to process the entire dataset and save features
def process_dataset(dataset_path=DATASET_PATH, output_file="audio_features_1D.csv"):
    features = []
    labels = []

    # Verificar que el dataset contiene los 10 géneros
    genre_folders = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]
    
    if len(genre_folders) != 10:
        raise ValueError(f"Error: Se encontraron {len(genre_folders)} géneros en lugar de 10. Verifica la extracción del dataset.")

    print(f"Se encontraron {len(genre_folders)} géneros. Procesando...")

    for genre_folder in tqdm(genre_folders, desc="Processing Genres"):
        genre_path = os.path.join(dataset_path, genre_folder)
        file_paths = glob.glob(os.path.join(genre_path, "*.wav"))

        # Verificar que haya 100 archivos en cada género
        if len(file_paths) != 100:
            warnings.warn(f"Advertencia: Se encontraron {len(file_paths)} archivos en {genre_folder} en lugar de 100.")

        for file_path in file_paths:
            try:
                feature_vector = extract_features(file_path)
                features.append(feature_vector)
                labels.append(genre_folder)
            except Exception as e:
                warnings.warn(f"No se pudo procesar el archivo {file_path}: {e}")
                continue  # Skip this file

    # Convert to DataFrame and save as CSV
    feature_df = pd.DataFrame(features)
    feature_df["label"] = labels
    feature_df.to_csv(output_file, index=False)
    print(f"Características guardadas en {output_file}")

if __name__ == "__main__":
    # Descargar y extraer el dataset si es necesario
    download_and_extract_dataset()

    # Procesar el conjunto de datos y extraer características
    process_dataset(dataset_path=DATASET_PATH, output_file="audio_features_1D.csv")

