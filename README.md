# Real-Time American Sign Language (ASL) Recognition System

**Author**: Tresa Joby
**Email**: tresajoby18@gmail.com
**GitHub**: [@tresajoby](https://github.com/tresajoby)

---

## 📋 Project Overview

A real-time American Sign Language recognition system using computer vision and deep learning. The system recognises both **static gestures** (hand poses for letters and digits) and **dynamic gestures** (motion-based signs) directly from a standard webcam — no specialist hardware required.

### Key Features
- Real-time hand landmark detection using MediaPipe (~37 FPS on CPU)
- Static gesture recognition using a Multi-Layer Perceptron (MLP)
- Dynamic gesture recognition using a Bidirectional LSTM (BiLSTM)
- Automatic mode switching based on wrist motion detection
- Privacy-preserving design — only hand landmarks stored, no raw video

---

## 🏗️ System Architecture

```
Webcam Frame
    → MediaPipe Hands (21 landmarks × 3 coords)
    → Wrist-relative normalisation (63-dim feature vector)
    → Motion detection
         ├─ Still  → Static MLP  → Letter / Digit
         └─ Moving → BiLSTM      → Word / Motion letter
    → Prediction + confidence displayed on screen
```

### Technology Stack
- **Language**: Python 3.11
- **Computer Vision**: OpenCV 4.x, MediaPipe 0.10.14
- **Deep Learning**: TensorFlow 2.21, Keras 3
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn, Streamlit

---

## 📂 Project Structure

```
Sign_Language_Recognition/
│
├── data/
│   ├── raw/
│   │   ├── static/          # Per-class landmark .npy files
│   │   └── dynamic/         # Per-class sequence .npy files
│   ├── processed/           # Train/val/test split arrays
│   └── labels/              # Label mapping JSON files
│
├── src/
│   ├── data_collection/     # Webcam collection scripts
│   ├── preprocessing/       # HandDetector + FeatureExtractor
│   ├── models/              # MLP and BiLSTM architectures
│   ├── training/            # Train + evaluate scripts
│   ├── inference/           # Real-time inference engine
│   └── utils/               # Config, metrics, visualisation
│
├── models/
│   ├── static_model.h5      # Trained MLP (268 KB)
│   └── dynamic_model.h5     # Trained BiLSTM (1.37 MB)
│
├── docs/plots/              # Confusion matrices, accuracy plots
├── app.py                   # Streamlit evaluation dashboard
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11
- Webcam (for data collection and live inference)

### Installation
```bash
git clone https://github.com/tresajoby/Sign_Language_Recognition.git
cd Sign_Language_Recognition

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install tensorflow==2.21.0 mediapipe==0.10.14 opencv-python numpy pandas streamlit matplotlib seaborn scikit-learn
```

### Run Real-Time Inference
```bash
python -m src.inference.run
```

Controls: `SPACE` capture · `R` reset buffer · `S` screenshot · `Q` quit

### Run Evaluation Dashboard (local)
```bash
streamlit run app.py
```

### Retrain Models
```bash
python -m src.training.train_static
python -m src.training.train_dynamic
```

### Collect New Data
```bash
python -m src.data_collection.collect_static
python -m src.data_collection.collect_dynamic
```

---

## 📊 Dataset

### Static Gestures
| Property | Value |
|---|---|
| Classes | A–Y (excl. J, Z) + digits 0–9 = **34 classes** |
| Samples per class | 90–100 landmark arrays |
| Total samples | **3,300** |
| Train / Val / Test | 2,309 / 496 / 495 |
| Feature dimensions | 21 landmarks × 3 coords = **63** |

### Dynamic Gestures
| Property | Value |
|---|---|
| Classes | J, Z, hello, thanks, please, sorry, yes, no, help, stop, more, finish = **12 classes** |
| Sequences per class | 100 |
| Total sequences | **1,200** |
| Train / Val / Test | 840 / 180 / 180 |
| Sequence shape | 30 frames × 63 features |

---

## 🧪 Model Performance

| Model | Task | Accuracy | Macro F1 | Inference FPS |
|---|---|---|---|---|
| MLP | Static gestures (34 classes) | **97.37%** | **97.29%** | 10.6 |
| BiLSTM | Dynamic gestures (12 classes) | **97.22%** | **97.22%** | 10.3 |
| MediaPipe | Hand detection | — | — | **37.7** |

### Top Misclassifications
- **Static**: U↔R (similar finger extension), 0↔O (identical round shape)
- **Dynamic**: J→finish (overlapping curved wrist motion)

---

## 🔒 Privacy & Ethics

- **No video storage** — only 63-dimensional landmark vectors are saved
- **No PII collected** — landmarks carry no identity information
- **Single-signer dataset** — results may vary across different users; multi-signer extension is a noted future improvement

---

## 📚 References

1. Lugaresi, C. et al. (2019). MediaPipe: A Framework for Perceiving and Processing Reality. *Google Research*.
2. Abadi, M. et al. (2015). TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems. *OSDI*.
3. Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
4. Schuster, M. & Paliwal, K. K. (1997). Bidirectional Recurrent Neural Networks. *IEEE Transactions on Signal Processing*, 45(11), 2673–2681.

---

## 📝 License

This project is developed for academic purposes.

---
