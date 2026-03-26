# ASL Recognition System - Project Status

**Student**: Tresa Joby
**Project**: Real-Time American Sign Language Recognition System
**Date**: 2026-01-12
**Current Phase**: Step 1 - System Architecture & Project Structure

---

## 🎯 Project Overview

Developing a real-time ASL recognition system that recognizes:
- **Static gestures**: A-Z letters, 0-9 digits (36 classes)
- **Dynamic gestures**: Words like "hello", "thanks", "please" (10 classes)

Using:
- **MediaPipe** for hand landmark detection
- **MLP** for static gesture classification
- **BiLSTM** for dynamic gesture sequence recognition
- **OpenCV** for real-time video processing

---

## ✅ Step 1: COMPLETE - System Architecture & Project Structure

### What Was Built

#### 1. Professional Directory Structure
```
Sign_Language_Recognition/
├── data/                      # Dataset storage
│   ├── raw/static/           # Static gesture data
│   ├── raw/dynamic/          # Dynamic gesture sequences
│   ├── processed/            # Preprocessed features
│   └── labels/               # Label mappings
├── src/                       # Source code
│   ├── data_collection/      # Data capture
│   ├── preprocessing/        # Feature extraction
│   ├── models/               # Model architectures
│   ├── training/             # Training scripts
│   ├── inference/            # Real-time recognition
│   └── utils/                # Configuration
├── models/                    # Saved trained models
├── notebooks/                 # Jupyter notebooks
├── tests/                     # Unit tests
└── docs/                      # Documentation
    ├── architecture.md       # System design
    ├── thesis_notes.md       # Writing guide
    ├── references.bib        # Bibliography
    └── plots/                # Evaluation figures
```

#### 2. System Architecture

```
┌─────────────────┐
│  Webcam Input   │
└────────┬────────┘
         ↓
┌─────────────────┐
│   MediaPipe     │ ← Hand Detection
│ Landmark Extract│   (21 points × 3D)
└────────┬────────┘
         ↓
    ┌────┴────┐
    │ Router  │ (Static vs Dynamic)
    └────┬────┘
         ↓
    ┌────┴────┐
    ↓         ↓
┌───────┐  ┌──────────┐
│  MLP  │  │  BiLSTM  │
│Static │  │ Dynamic  │
└───┬───┘  └────┬─────┘
    │           │
    └─────┬─────┘
          ↓
   ┌──────────────┐
   │  Prediction  │
   └──────────────┘
```

#### 3. Key Configuration Parameters

**MediaPipe**:
- Detection confidence: 0.7
- Tracking confidence: 0.5
- Max hands: 1

**Data Collection**:
- Static: 300 samples × 36 classes = 10,800 samples
- Dynamic: 100 sequences × 10 classes × 30 frames = 30,000 frames

**Features**:
- Dimension: 63 (21 landmarks × 3 coords)
- Normalization: Wrist-relative

**Static Model (MLP)**:
- Architecture: 63 → 128 → 64 → 32 → 36
- Dropout: 0.3
- Activation: ReLU → Softmax

**Dynamic Model (BiLSTM)**:
- Architecture: (30, 63) → BiLSTM(64) → BiLSTM(32) → 10
- Sequence length: 30 frames
- Dropout: 0.3

**Training**:
- Split: 70% train / 15% val / 15% test
- Batch size: 32
- Epochs: 50
- Optimizer: Adam (lr=0.001)
- Early stopping: patience=10

#### 4. Documentation Created

| Document | Purpose | Status |
|----------|---------|--------|
| README.md | Project overview | ✅ Complete |
| SETUP_GUIDE.md | Environment setup | ✅ Complete |
| docs/architecture.md | System design | ✅ Complete |
| docs/thesis_notes.md | Writing guide | ✅ Complete |
| docs/references.bib | Bibliography | ✅ Complete |
| docs/STEP1_SUMMARY.md | Phase summary | ✅ Complete |
| requirements.txt | Dependencies | ✅ Complete |
| src/utils/config.py | Configuration | ✅ Complete |

---

## 📊 Thesis Alignment

| Requirement | Status | Evidence |
|-------------|--------|----------|
| **A. Problem Understanding** | ✅ | README motivation section |
| **B. System Design** | ✅ | Modular architecture documented |
| **C. Technical Implementation** | 🔄 | Structure ready, code pending |
| **D. ML Methodology** | ✅ | Models & training strategy defined |
| **E. Evaluation** | 📋 | Metrics planned |
| **F. Ethics & Privacy** | ✅ | No video storage policy |
| **G. Documentation** | ✅ | Comprehensive docs |

Legend: ✅ Complete | 🔄 In Progress | 📋 Planned | ❌ Not Started

---

## 🎓 Academic Justifications

### Why MediaPipe?
- State-of-the-art accuracy (Google Research)
- Real-time performance (>30 FPS on CPU)
- 3D landmark coordinates
- Well-documented and maintained

### Why MLP for Static Gestures?
- No temporal dependency in static poses
- Landmark coordinates already encode spatial relationships
- Lightweight and fast inference
- Proven effective for similar tasks

### Why BiLSTM for Dynamic Gestures?
- Captures temporal dependencies in motion
- Bidirectional context improves accuracy
- Handles variable-length sequences
- Standard for sequence modeling

### Why Wrist-Relative Normalization?
- **Scale invariance**: Works with different hand sizes
- **Position invariance**: Hand location doesn't matter
- **Preserves shape**: Maintains hand geometry
- **Standard practice**: Used in gesture recognition literature

---

## 📅 Project Roadmap

### ✅ Phase 1: Architecture & Structure (Week 1)
- [x] Project structure
- [x] Configuration system
- [x] Documentation
- [x] Thesis planning

### 📋 Phase 2: Data Collection & Preprocessing (Weeks 2-3)
- [ ] MediaPipe hand detector implementation
- [ ] Static gesture data collection interface
- [ ] Dynamic gesture data collection interface
- [ ] Feature extraction and normalization
- [ ] Dataset management utilities

### 📋 Phase 3: Model Development (Weeks 4-5)
- [ ] Static MLP model implementation
- [ ] Dynamic BiLSTM model implementation
- [ ] Training pipeline
- [ ] Model evaluation utilities

### 📋 Phase 4: Real-Time Inference (Week 6)
- [ ] Real-time recognition interface
- [ ] Webcam integration
- [ ] Visualization and feedback
- [ ] Performance optimization

### 📋 Phase 5: Evaluation & Analysis (Week 7)
- [ ] Train final models
- [ ] Generate evaluation metrics
- [ ] Create confusion matrices
- [ ] Runtime performance testing
- [ ] Robustness analysis

### 📋 Phase 6: Documentation & Thesis (Weeks 8-12)
- [ ] Complete methodology chapter
- [ ] Results and discussion
- [ ] Generate all figures and tables
- [ ] Final thesis draft
- [ ] Defense preparation

---

## 🔍 Next Steps

**WAIT FOR APPROVAL** before proceeding to Step 2.

When ready, say:
> **"Proceed to the next step."**

**Step 2 will cover**:
1. MediaPipe hand detection wrapper
2. Feature extraction implementation
3. Data collection interfaces (static & dynamic)
4. Dataset management utilities
5. Preprocessing pipeline

**Estimated time**: 1-2 weeks
**Deliverables**: Working data collection system + preprocessed dataset

---

## 💡 Questions to Consider

Before moving forward, discuss with supervisor:

1. **Dataset Size**: Is 300 samples per class sufficient for your timeline?
2. **Participants**: How many people can help with data collection?
3. **Hardware**: Webcam available? Specifications?
4. **Compute**: Will you train on CPU or GPU?
5. **Thesis Format**: University template requirements?
6. **Timeline**: Submission deadline?

---

## 📚 Key Files to Review

For your thesis writing:

1. **Architecture**: [docs/architecture.md](docs/architecture.md)
   - System design diagrams
   - Technical justifications
   - Module descriptions

2. **Thesis Guide**: [docs/thesis_notes.md](docs/thesis_notes.md)
   - Chapter structure
   - Writing examples
   - Academic style guide
   - Defense questions

3. **Configuration**: [src/utils/config.py](src/utils/config.py)
   - All hyperparameters
   - Reproducibility settings
   - System parameters

4. **Bibliography**: [docs/references.bib](docs/references.bib)
   - Key papers to cite
   - MediaPipe, LSTM, gesture recognition

---

## ✨ Strengths of Current Design

1. **Modular**: Each component independent and testable
2. **Documented**: Comprehensive architecture documentation
3. **Justified**: Every design choice has technical reasoning
4. **Reproducible**: Centralized configuration with random seeds
5. **Privacy-Preserving**: No raw video storage
6. **Scalable**: Easy to add new gesture classes
7. **Professional**: Follows software engineering best practices

---

## ⚠️ Limitations to Acknowledge

For thesis discussion:

1. **Single Hand**: Currently limited to one hand detection
2. **Controlled Environment**: Requires adequate lighting
3. **Limited Vocabulary**: 46 total classes (not full ASL)
4. **No Grammar**: Word-by-word, not sentence-level
5. **Static Background**: May struggle with cluttered backgrounds

These are acceptable for an undergraduate thesis and provide good "Future Work" discussion.

---

## 🎯 Success Criteria

The project will be successful if it:

- ✅ Achieves >85% accuracy on static gestures
- ✅ Achieves >75% accuracy on dynamic gestures
- ✅ Runs at ≥15 FPS for real-time interaction
- ✅ Demonstrates clear methodology and evaluation
- ✅ Addresses ethical considerations
- ✅ Provides comprehensive documentation

---

**Current Status**: Step 1 Complete ✅
**Ready for**: Step 2 - Data Collection & Preprocessing
**Awaiting**: Supervisor approval to proceed

---

*Last Updated: 2026-01-12*
