# Real-Time American Sign Language (ASL) Recognition System

**Undergraduate Thesis Project**
**Author**: Tresa Joby
**Institution**: [Your University Name]
**Academic Year**: 2024-2025

---

## 📋 Project Overview

This project implements a real-time American Sign Language (ASL) recognition system using computer vision and deep learning techniques. The system recognizes both **static gestures** (individual letters/signs) and **dynamic gestures** (signs involving motion) to facilitate communication accessibility.

### Key Features
- Real-time hand landmark detection using MediaPipe
- Static gesture recognition using Multi-Layer Perceptron (MLP)
- Dynamic gesture recognition using Bidirectional LSTM (BiLSTM)
- Live webcam inference with OpenCV interface
- Privacy-preserving design (no raw video storage)

---

## 🎯 Academic Objectives

This project addresses:
1. **Problem Understanding**: Bridging communication gaps for deaf and hard-of-hearing communities
2. **Technical Implementation**: Computer vision + deep learning pipeline
3. **ML Methodology**: Dataset collection, feature engineering, temporal modeling
4. **Evaluation**: Quantitative metrics and performance analysis
5. **Ethics**: Privacy, fairness, and bias considerations

---

## 🏗️ System Architecture

```
Input (Webcam) → MediaPipe Hand Detection → Feature Extraction →
Classification (MLP/BiLSTM) → Prediction Output
```

### Technology Stack
- **Language**: Python 3.8+
- **Computer Vision**: OpenCV, MediaPipe
- **Deep Learning**: TensorFlow/Keras
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn

---

## 📂 Project Structure

```
Sign_Language_Recognition/
│
├── data/                          # Dataset storage
│   ├── raw/                       # Raw collected data
│   │   ├── static/               # Static gesture data
│   │   └── dynamic/              # Dynamic gesture sequences
│   ├── processed/                # Preprocessed features
│   │   ├── static_features.npy
│   │   └── dynamic_sequences.npy
│   └── labels/                   # Label mappings
│
├── src/                          # Source code modules
│   ├── data_collection/          # Data collection scripts
│   │   ├── collect_static.py
│   │   └── collect_dynamic.py
│   ├── preprocessing/            # Feature extraction
│   │   ├── hand_detector.py     # MediaPipe wrapper
│   │   └── feature_extractor.py # Landmark processing
│   ├── models/                   # Model architectures
│   │   ├── static_mlp.py        # MLP for static gestures
│   │   └── dynamic_lstm.py      # BiLSTM for sequences
│   ├── training/                 # Training scripts
│   │   ├── train_static.py
│   │   └── train_dynamic.py
│   ├── inference/                # Real-time inference
│   │   └── realtime_recognizer.py
│   └── utils/                    # Utility functions
│       ├── config.py            # Configuration management
│       ├── visualization.py     # Plotting utilities
│       └── metrics.py           # Evaluation metrics
│
├── models/                       # Saved trained models
│   ├── static_model.h5
│   └── dynamic_model.h5
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
│
├── tests/                        # Unit tests
│   ├── test_preprocessing.py
│   └── test_models.py
│
├── docs/                         # Documentation
│   ├── thesis_notes.md
│   ├── architecture.md
│   └── references.bib
│
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8 or higher
Webcam (for data collection and inference)
```

### Installation
```bash
# Clone the repository
git clone https://github.com/tresajoby/Sign_Language_Recognition.git
cd Sign_Language_Recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start
```bash
# Collect static gesture data
python src/data_collection/collect_static.py

# Train static gesture model
python src/training/train_static.py

# Run real-time inference
python src/inference/realtime_recognizer.py
```

---

## 📊 Dataset

### Static Gestures
- **Classes**: A-Z (26 letters), 0-9 (10 digits), common words
- **Samples per class**: 100-500 images
- **Features**: 21 hand landmarks (x, y, z) = 63 dimensions

### Dynamic Gestures
- **Classes**: "Hello", "Thank You", "Please", etc.
- **Sequence length**: 30 frames per gesture
- **Features**: 63 dimensions × 30 timesteps

---

## 🧪 Model Performance

| Model | Task | Accuracy | F1-Score | FPS |
|-------|------|----------|----------|-----|
| MLP | Static Gestures | TBD | TBD | TBD |
| BiLSTM | Dynamic Gestures | TBD | TBD | TBD |

---

## 🔒 Privacy & Ethics

- **No Video Storage**: Only hand landmarks are extracted and stored
- **Data Anonymization**: No personally identifiable information collected
- **Bias Mitigation**: Diverse hand sizes, skin tones, and lighting conditions
- **Consent**: All participants provide informed consent

---

## 📚 References

1. Google MediaPipe: https://mediapipe.dev/
2. TensorFlow/Keras Documentation: https://www.tensorflow.org/
3. ASL Research: [Add academic references]

---

## 📝 License

This project is developed for academic purposes as part of an undergraduate thesis.

---

## 📧 Contact

**Tresa Joby**
Email: [Your Email]
GitHub: [@tresajoby](https://github.com/tresajoby)

---
