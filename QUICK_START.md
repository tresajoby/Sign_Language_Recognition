# Quick Start Guide

## Step-by-Step Setup & Testing

### 1. Install Dependencies

First, you need to install all required Python packages:

```bash
# Navigate to project directory
cd "C:\Users\Adven\OneDrive\Documents\My files\Sign_Language_Recognition"

# Install all dependencies
pip install numpy==1.24.3 pandas==2.0.3 opencv-python==4.8.1.78 mediapipe==0.10.8 tensorflow==2.15.0 matplotlib==3.7.2 seaborn==0.12.2 scikit-learn==1.3.0 tqdm==4.66.1

# Or use requirements file
pip install -r requirements.txt
```

### 2. Test the System

After installation, run:

```bash
python test_system.py
```

Expected output:
```
======================================================================
ASL RECOGNITION SYSTEM - COMPONENT TEST
======================================================================

[TEST 1] Testing Configuration...
✅ Configuration module working
   - Static classes: 36
   - Dynamic classes: 10
   - Feature dimension: 63

[TEST 2] Testing Hand Detector...
✅ Hand detector initialized
   - Max hands: 1
   - Detection confidence: 0.7
   - Detector callable: ✅

[TEST 3] Testing Feature Extractor...
✅ Feature extractor initialized
   - Normalization: wrist
   - Feature dimension: 63
   - Feature extraction: ✅
   - Output shape: (63,)

[TEST 4] Checking Data Collection Scripts...
✅ Static collection script importable
✅ Dynamic collection script importable

[TEST 5] Checking Directory Structure...
   ✅ Data: ...
   ✅ Raw data: ...
   ✅ Processed data: ...
   ✅ Models: ...
   ✅ Labels: ...

======================================================================
TEST SUMMARY
======================================================================

✅ All core components working!
```

### 3. Test Hand Detection with Webcam

```bash
python src/preprocessing/hand_detector.py
```

This will:
- Open a webcam window
- Show real-time hand detection
- Display 21 hand landmarks
- Show bounding box around your hand
- Press **'q'** to quit

### 4. Collect Your Dataset

#### Option A: Collect Static Gestures (Letters & Numbers)

```bash
python src/data_collection/collect_static.py
```

Follow the prompts:
1. Choose what to collect (all, letters, numbers, or custom)
2. Set samples per gesture (default 300)
3. For each gesture:
   - Show the gesture to the camera
   - Press **SPACE** to capture
   - Progress bar shows completion

#### Option B: Collect Dynamic Gestures (Words/Phrases)

```bash
python src/data_collection/collect_dynamic.py
```

For each gesture:
1. Press **SPACE** to start
2. Wait for 3-second countdown
3. Perform the gesture naturally
4. System captures 30 frames automatically

### 5. Check Collected Data

After collection, verify:

```bash
# Check static data
ls "data/processed/"
# Should see: static_features.npy, static_labels.npy

# Check dynamic data (if collected)
ls "data/processed/"
# Should see: dynamic_sequences.npy, dynamic_labels.npy
```

---

## Quick Test (If No Webcam Available)

If you don't have a webcam or want to test without it:

```bash
# Test only the feature extraction (no webcam needed)
python src/preprocessing/feature_extractor.py
```

This will test the feature extraction with synthetic data.

---

## Common Issues & Solutions

### Issue: "No module named 'numpy'"
**Solution**:
```bash
pip install numpy
```

### Issue: "Cannot open camera"
**Solutions**:
- Make sure webcam is connected
- Close other applications using the camera (Zoom, Teams, etc.)
- Try different camera ID in config.py: `CAMERA_ID = 1`

### Issue: "TensorFlow not found"
**Solution**:
```bash
pip install tensorflow==2.15.0
```

### Issue: Hand not detected
**Solutions**:
- Ensure good lighting
- Keep hand within camera view
- Use plain background
- Check MediaPipe confidence settings in config.py

---

## What Each File Does

| File | Purpose |
|------|---------|
| `test_system.py` | Tests all components without webcam |
| `src/preprocessing/hand_detector.py` | Tests hand detection with webcam |
| `src/preprocessing/feature_extractor.py` | Tests feature extraction |
| `src/data_collection/collect_static.py` | Collects static gesture data |
| `src/data_collection/collect_dynamic.py` | Collects dynamic gesture data |
| `src/utils/config.py` | All configuration parameters |

---

## Next Steps After Data Collection

Once you've collected your dataset:

1. ✅ Data collected and saved in `data/processed/`
2. ⏭️ Proceed to **Step 3: Model Training**
   - Implement MLP for static gestures
   - Implement BiLSTM for dynamic gestures
   - Train models on your collected data
3. ⏭️ **Step 4: Real-time Inference**
   - Build live recognition system
   - Test with webcam

---

## Project Progress

- ✅ **Step 1**: System Architecture (Complete)
- ✅ **Step 2**: Data Collection & Preprocessing (Complete)
- ⏭️ **Step 3**: Model Training (Next)
- ⏭️ **Step 4**: Real-time Inference
- ⏭️ **Step 5**: Evaluation & Results
- ⏭️ **Step 6**: Thesis Writing

---

*Ready to proceed with Step 3 after data collection!*
