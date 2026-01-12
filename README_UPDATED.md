# Real-Time American Sign Language (ASL) Recognition System

**Undergraduate Thesis Project**
**Author**: Tresa Joby
**Repository**: https://github.com/tresajoby/Sign_Language_Recognition

---

## 📋 Project Overview

This project implements a real-time American Sign Language (ASL) recognition system using computer vision and deep learning techniques. The system recognizes both **static gestures** (individual letters/signs) and **dynamic gestures** (signs involving motion) to facilitate communication accessibility.

### Key Features
- ✅ Real-time hand landmark detection using MediaPipe
- ✅ Static gesture recognition using Multi-Layer Perceptron (MLP)
- ✅ Dynamic gesture recognition using Bidirectional LSTM (BiLSTM)
- ✅ Live webcam inference with OpenCV interface
- ✅ Privacy-preserving design (no raw video storage)
- ✅ Interactive data collection tools

---

## 🎯 System Architecture

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
├── data/                          # Dataset storage
│   ├── raw/                       # Raw collected data
│   │   ├── static/               # Static gesture data
│   │   └── dynamic/              # Dynamic gesture sequences
│   ├── processed/                # Preprocessed features
│   └── labels/                   # Label mappings
├── src/                          # Source code modules
│   ├── data_collection/          # Data collection scripts
│   │   ├── collect_static.py    # Static gesture collection
│   │   └── collect_dynamic.py   # Dynamic gesture collection
│   ├── preprocessing/            # Feature extraction
│   │   ├── hand_detector.py     # MediaPipe wrapper
│   │   └── feature_extractor.py # Landmark processing
│   ├── models/                   # Model architectures (Step 3)
│   ├── training/                 # Training scripts (Step 3)
│   ├── inference/                # Real-time inference (Step 4)
│   └── utils/                    # Utility functions
│       └── config.py            # Configuration management
├── models/                       # Saved trained models
├── notebooks/                    # Jupyter notebooks
├── tests/                        # Unit tests
├── docs/                         # Documentation
│   ├── STEP1_SUMMARY.md        # Architecture phase
│   ├── STEP2_SUMMARY.md        # Data collection phase
│   └── thesis_notes.md         # Thesis writing guide
├── legacy_Main.py               # Original implementation
├── legacy_Function.py           # Original functions
├── requirements.txt             # Python dependencies
├── test_system.py               # System test script
└── README.md                    # This file
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Webcam (for data collection and inference)
- 8GB RAM (minimum), 16GB recommended

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/tresajoby/Sign_Language_Recognition.git
cd Sign_Language_Recognition
```

2. **Create virtual environment:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Test the installation:**
```bash
python test_system.py
```

Expected output:
```
✅ Configuration module working
✅ Hand detector initialized
✅ Feature extractor initialized
✅ All core components working!
```

---

## 🎮 Usage

### Step 1: Test Hand Detection

```bash
python src/preprocessing/hand_detector.py
```

- Webcam window will open showing your hand with landmarks
- Green dots mark the 21 hand landmarks
- Yellow box shows hand bounding box
- Press **'q'** to quit

### Step 2: Collect Your Dataset

#### Static Gestures (A-Z, 0-9)

```bash
python src/data_collection/collect_static.py
```

**Instructions**:
1. Select gesture classes to collect (all, letters only, numbers only, or custom)
2. Enter samples per gesture (default: 300)
3. For each gesture:
   - Position your hand to show the gesture
   - Press **SPACE** to capture frame
   - Progress bar shows completion
4. Press **ENTER** when done with current gesture
5. Press **ESC** to quit

**Controls**:
- `SPACE` - Capture current frame
- `ENTER` - Move to next gesture
- `R` - Reset progress for current gesture
- `ESC` - Quit collection

**Output**:
- Raw landmarks: `data/raw/static/{gesture}/landmarks_*.npy`
- Processed data: `data/processed/static_features.npy` (N, 63)
- Labels: `data/processed/static_labels.npy`

#### Dynamic Gestures (hello, thanks, please, etc.)

```bash
python src/data_collection/collect_dynamic.py
```

**Instructions**:
1. Press **SPACE** to start recording a sequence
2. 3-second countdown begins
3. Perform the gesture
4. System auto-records 30 frames (~1 second)
5. Sequence saved if >70% frames have hand detected

**Controls**:
- `SPACE` - Start recording sequence
- `ENTER` - Move to next gesture
- `R` - Reset progress
- `ESC` - Quit

**Output**:
- Raw sequences: `data/raw/dynamic/{gesture}/sequence_*.npy` (30, 21, 3)
- Processed data: `data/processed/dynamic_sequences.npy` (N, 30, 63)
- Labels: `data/processed/dynamic_labels.npy`

---

## 📊 Dataset

### Static Gestures
- **Classes**: 36 (A-Z letters, 0-9 digits)
- **Samples per class**: 300 (default)
- **Total samples**: 10,800
- **Features**: 63 dimensions (21 landmarks × 3 coords)

### Dynamic Gestures
- **Classes**: 10 (hello, thanks, please, sorry, yes, no, help, stop, more, finish)
- **Sequences per class**: 100 (default)
- **Total sequences**: 1,000
- **Sequence length**: 30 frames
- **Features per frame**: 63 dimensions

---

## 🔧 Configuration

All system parameters are in [src/utils/config.py](src/utils/config.py):

```python
# MediaPipe
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5
MAX_NUM_HANDS = 1

# Data Collection
STATIC_SAMPLES_PER_CLASS = 300
DYNAMIC_SEQUENCES_PER_CLASS = 100
DYNAMIC_SEQUENCE_LENGTH = 30

# Features
FEATURE_DIM = 63  # 21 landmarks × 3 coords
NORMALIZATION_METHOD = 'wrist'  # wrist-relative normalization
```

---

## 🎓 Academic Contributions

### System Architecture
- Modular pipeline with clear separation of concerns
- Privacy-by-design (only landmarks stored, no raw video)
- Justified technology choices with academic references

### Feature Engineering
- Wrist-relative normalization for scale and position invariance
- 3D landmark coordinates for robust gesture representation
- Feature dimension: 63 (suitable for real-time processing)

### Data Collection
- Systematic protocol for balanced dataset
- Progress tracking and resume capability
- Automatic validation and interpolation for dynamic gestures

---

## 📈 Project Status

### ✅ Completed
- [x] Step 1: System Architecture & Project Structure
- [x] Step 2: Data Collection & Preprocessing
  - [x] MediaPipe hand detector wrapper
  - [x] Feature extraction and normalization
  - [x] Static gesture data collection interface
  - [x] Dynamic gesture data collection interface

### 🔄 In Progress
- [ ] Step 3: Model Development
  - [ ] Static MLP model implementation
  - [ ] Dynamic BiLSTM model implementation
  - [ ] Training pipelines
  - [ ] Model evaluation

### 📋 Planned
- [ ] Step 4: Real-Time Inference System
- [ ] Step 5: Evaluation & Results Analysis
- [ ] Step 6: Thesis Documentation

---

## 🔒 Privacy & Ethics

- **No Video Storage**: Only hand landmarks are extracted and stored
- **Data Anonymization**: No personally identifiable information collected
- **Bias Mitigation**: Diverse hand sizes, skin tones, and lighting conditions
- **Consent**: All participants provide informed consent

---

## 📚 References

1. Zhang, F., et al. (2020). "MediaPipe Hands: On-device Real-time Hand Tracking." arXiv:2006.10214
2. Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory." Neural computation.
3. Google MediaPipe: https://mediapipe.dev/
4. TensorFlow/Keras: https://www.tensorflow.org/

---

## 🐛 Troubleshooting

### Webcam not detected
```bash
# Check available cameras
python -c "import cv2; print('Camera 0:', cv2.VideoCapture(0).isOpened())"

# Try different camera ID in config.py
CAMERA_ID = 1  # or 2
```

### MediaPipe installation issues
```bash
pip uninstall mediapipe
pip install mediapipe==0.10.8
```

### Import errors
```bash
# Make sure you're in the project root directory
cd "C:\Users\Adven\OneDrive\Documents\My files\Sign_Language_Recognition"

# Verify PYTHONPATH
python -c "import sys; print('\\n'.join(sys.path))"
```

---

## 💡 Tips for Data Collection

1. **Lighting**: Collect in well-lit areas with consistent lighting
2. **Background**: Use plain backgrounds for better detection
3. **Variation**: Vary hand position and orientation slightly between samples
4. **Consistency**: Keep gestures consistent with ASL standards
5. **Sessions**: Collect in multiple sessions to avoid fatigue
6. **Speed**: For dynamic gestures, perform at natural speed (~1 second per gesture)

---

## 📞 Contact

**Tresa Joby**
GitHub: [@tresajoby](https://github.com/tresajoby)
Repository: https://github.com/tresajoby/Sign_Language_Recognition

---

## 🙏 Acknowledgments

- Thesis Supervisor: [Supervisor Name]
- [Your University Name], Department of [Your Department]
- Google MediaPipe and TensorFlow teams for open-source tools

---

## 📝 License

This project is developed for academic purposes as part of an undergraduate thesis.

---

*Last Updated: 2026-01-12*
*Current Phase: Step 2 - Data Collection & Preprocessing (Complete)*
