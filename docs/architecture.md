# System Architecture Documentation

## 1. Overview

The Real-Time ASL Recognition System follows a **modular pipeline architecture** designed for:
- **Separation of Concerns**: Each module has a single, well-defined responsibility
- **Testability**: Individual components can be tested in isolation
- **Maintainability**: Easy to update or replace specific modules
- **Scalability**: Simple to add new gesture classes or models

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT LAYER                              │
│  - Webcam/Video Stream                                       │
│  - Frame Capture (OpenCV)                                    │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│              PREPROCESSING & FEATURE EXTRACTION              │
│  - Hand Detection (MediaPipe)                                │
│  - Landmark Extraction (21 points × 3D coords)               │
│  - Normalization (wrist-relative)                            │
│  - Feature Vector Construction                               │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
                  ┌────┴────┐
                  │ Router  │ (Static vs Dynamic)
                  └────┬────┘
                       ↓
          ┌────────────┴────────────┐
          ↓                         ↓
┌─────────────────┐        ┌─────────────────┐
│  STATIC MODEL   │        │ DYNAMIC MODEL   │
│  - MLP          │        │  - BiLSTM/GRU   │
│  - Single Frame │        │  - Sequence     │
│  - 63 features  │        │  - 30 frames    │
└────────┬────────┘        └────────┬────────┘
         │                          │
         └────────────┬─────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│                     OUTPUT LAYER                             │
│  - Prediction Class                                          │
│  - Confidence Score                                          │
│  - Visual Feedback (landmarks, bounding box, text)           │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Module Breakdown

### 3.1 Input Layer (`src/data_collection/`)

**Purpose**: Capture visual input and manage data collection

**Components**:
- `collect_static.py`: Captures individual frames for static gestures
- `collect_dynamic.py`: Captures temporal sequences for dynamic gestures

**Technical Decisions**:
- OpenCV for cross-platform webcam access
- Fixed resolution (640×480) for consistency
- Frame buffering for dynamic gestures

**Inputs**:
- Webcam feed (live)
- Video files (batch processing)

**Outputs**:
- Raw frames for immediate processing
- Saved landmark data for training

---

### 3.2 Preprocessing & Feature Extraction (`src/preprocessing/`)

**Purpose**: Convert raw images to structured feature vectors

**Components**:
- `hand_detector.py`: MediaPipe wrapper for hand detection
- `feature_extractor.py`: Landmark normalization and feature engineering

**Key Algorithms**:

1. **Hand Detection** (MediaPipe):
   - Uses CNN-based palm detection
   - 21 landmark regression per hand
   - 3D coordinates (x, y, z)

2. **Normalization** (Wrist-Relative):
   ```
   For each landmark (x, y, z):
       x_norm = (x - wrist_x) / hand_width
       y_norm = (y - wrist_y) / hand_height
       z_norm = z - wrist_z
   ```

   **Justification**:
   - Scale invariance (works with different hand sizes)
   - Position invariance (hand can be anywhere in frame)
   - Preserves hand shape information

**Inputs**:
- RGB frames (H × W × 3)

**Outputs**:
- Feature vector: 63-dimensional (21 landmarks × 3 coords)
- Sequence of features for dynamic gestures: (30, 63)

---

### 3.3 Classification Models (`src/models/`)

#### 3.3.1 Static Gesture Model - MLP

**Architecture**:
```
Input (63) → Dense(128, ReLU) → Dropout(0.3) →
Dense(64, ReLU) → Dropout(0.3) →
Dense(32, ReLU) → Dropout(0.3) →
Dense(36, Softmax)
```

**Justification**:
- **MLP Suitability**: Static gestures have no temporal dependency; spatial hand configuration is sufficient
- **Dropout**: Prevents overfitting on small datasets
- **Layer Sizes**: Gradual reduction from 128 → 64 → 32 creates hierarchical feature learning

**Parameters**: ~10K (lightweight, fast inference)

---

#### 3.3.2 Dynamic Gesture Model - BiLSTM

**Architecture**:
```
Input (30, 63) → BiLSTM(64, return_sequences=True) → Dropout(0.3) →
BiLSTM(32) → Dropout(0.3) →
Dense(10, Softmax)
```

**Justification**:
- **LSTM**: Captures temporal dependencies in gesture sequences
- **Bidirectional**: Uses both past and future context (important for recognizing motion patterns)
- **Sequence Modeling**: Processes 30-frame windows to capture full gesture trajectory

**Alternative Considered**: GRU (fewer parameters, faster training)

**Parameters**: ~50K

---

### 3.4 Training Pipeline (`src/training/`)

**Components**:
- `train_static.py`: Static model training with data augmentation
- `train_dynamic.py`: Sequence model training

**Training Strategy**:
1. **Data Split**: 70% train / 15% validation / 15% test
2. **Data Augmentation** (for robustness):
   - Random rotation (±15°)
   - Random scaling (0.9-1.1×)
   - Gaussian noise (σ=0.01)
3. **Regularization**:
   - Dropout layers
   - Early stopping (patience=10)
4. **Optimization**:
   - Adam optimizer (lr=0.001)
   - Categorical cross-entropy loss

---

### 3.5 Real-Time Inference (`src/inference/`)

**Component**: `realtime_recognizer.py`

**Pipeline**:
1. Capture frame from webcam
2. Detect hand landmarks
3. Extract features
4. Route to appropriate model (static/dynamic)
5. Display prediction + confidence
6. Render visual feedback

**Performance Optimization**:
- Frame skipping if FPS drops
- Model inference on CPU (sufficient for real-time)
- Efficient NumPy operations

**Target Performance**:
- ≥15 FPS (minimum for smooth interaction)
- <100ms inference latency

---

## 4. Data Flow

### 4.1 Training Phase

```
Raw Data → Feature Extraction → Train/Val/Test Split →
Model Training → Evaluation → Model Saving
```

### 4.2 Inference Phase

```
Webcam Frame → Hand Detection → Feature Extraction →
Model Prediction → Display Output
```

---

## 5. Technology Justification

| Technology | Purpose | Justification |
|------------|---------|---------------|
| **Python** | Language | Rich ML/CV ecosystem, rapid prototyping |
| **OpenCV** | Video I/O | Industry standard, cross-platform |
| **MediaPipe** | Hand Detection | State-of-the-art accuracy, real-time performance |
| **TensorFlow/Keras** | Deep Learning | High-level API, extensive documentation |
| **NumPy** | Array Operations | Fast numerical computation |

---

## 6. Design Principles Applied

### 6.1 Modularity
Each module can be developed and tested independently:
- Data collection doesn't depend on models
- Models can be swapped without changing preprocessing
- Inference pipeline is model-agnostic

### 6.2 Configuration Management
All hyperparameters centralized in `config.py`:
- Easy to document for thesis
- Supports reproducibility
- Simplifies experimentation

### 6.3 Privacy by Design
- No raw video storage
- Only landmark coordinates saved
- Anonymized data collection

### 6.4 Extensibility
Easy to add:
- New gesture classes (update config)
- New models (add to `models/`)
- New features (extend `feature_extractor.py`)

---

## 7. Limitations & Future Improvements

### Current Limitations
1. Single-hand detection only (can extend to 2 hands)
2. Controlled lighting required (can add augmentation)
3. Limited gesture vocabulary (10-36 classes)
4. No sentence-level recognition (word-by-word only)

### Proposed Improvements
1. **Multi-hand Support**: Detect both hands simultaneously
2. **Attention Mechanism**: Add attention layers to BiLSTM for better sequence modeling
3. **Transfer Learning**: Use pre-trained models for feature extraction
4. **Edge Deployment**: Optimize for mobile/embedded devices
5. **User Adaptation**: Fine-tune models for individual users

---

## 8. Thesis Integration

This architecture directly addresses thesis requirements:

| Requirement | Architecture Component |
|-------------|------------------------|
| **Problem Understanding** | Clear motivation for each design choice |
| **System Design** | Modular pipeline with justified technology |
| **Technical Implementation** | Correct use of ML/CV libraries |
| **ML Methodology** | Explicit training pipeline and evaluation |
| **Ethics** | Privacy-preserving data handling |
| **Documentation** | Comprehensive architecture documentation |

---

## References

1. MediaPipe Hands: https://arxiv.org/abs/2006.10214
2. Zhang, F., et al. (2020). "MediaPipe: A Framework for Building Perception Pipelines"
3. Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory"
