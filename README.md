# -UrbanSound8K---Environmental-Sound-Classification-using-Machine-Learning
The UrbanSound8K dataset is a collection of 8,732 labeled audio clips (each ≤ 4 seconds long), grouped into 10 classes of common urban sounds such as air conditioner, car horn, children playing, dog bark, drilling, engine idling, gunshot, jackhammer, siren, and street music. Each clip is a .wav file, and metadata is provided in a CSV file .
# 🔊🎵 UrbanSound8K - Environmental Sound Classification using Machine Learning

## 📌 Project Overview

This project aims to classify **urban sound clips** into predefined categories using **machine learning techniques**. We use the **UrbanSound8K dataset**, which includes 8,732 labeled sound excerpts (<=4s) of urban sounds from 10 classes such as air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, and street_music.

By extracting meaningful audio features from WAV files, such as **MFCCs, chroma, and zero-crossing rate**, and training traditional ML classifiers like **Random Forests**, **SVM**, or **KNN**, we aim to build an effective environmental sound recognition system.

---

## 🎯 Objectives

- Load and process the UrbanSound8K audio files.
- Extract meaningful **audio features** for classification.
- Apply **machine learning algorithms** for multi-class classification.
- Evaluate model accuracy and performance.
- Visualize classification metrics and feature distributions.

---

## 🗂️ Dataset: UrbanSound8K

- 📁 Format: WAV audio files + metadata CSV
- 🔟 Classes:
  - air_conditioner
  - car_horn
  - children_playing
  - dog_bark
  - drilling
  - engine_idling
  - gun_shot
  - jackhammer
  - siren
  - street_music
- 🔗 [UrbanSound8K Dataset](https://urbansounddataset.weebly.com/urbansound8k.html)


import librosa
import os
import pandas as pd

# Load metadata
metadata = pd.read_csv("UrbanSound8K/metadata/UrbanSound8K.csv")

# Example to load a WAV file
file_path = "UrbanSound8K/audio/fold1/101415-3-0-2.wav"
audio, sr = librosa.load(file_path, sr=None)

# Extract MFCC features
mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
