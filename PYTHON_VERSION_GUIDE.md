# Python Version Compatibility Guide

## Issue: MediaPipe Compatibility

You're currently using **Python 3.13**, but MediaPipe's hand tracking functionality (the legacy `mp.solutions.hands` API) is only available in versions **0.10.8 and earlier**, which don't support Python 3.13.

The newer MediaPipe versions (0.10.30+) have a completely different API that requires downloading model files and more complex setup.

## Solution Options

### Option 1: Use Python 3.10 or 3.11 (RECOMMENDED)

This is the **easiest and most reliable solution** for this project.

#### Steps:

1. **Install Python 3.10 or 3.11**:
   - Download from: https://www.python.org/downloads/
   - Choose Python 3.10.x or 3.11.x (latest stable)
   - During installation, check "Add Python to PATH"

2. **Create a virtual environment**:
   ```bash
   cd "C:\Users\Adven\OneDrive\Documents\My files\Sign_Language_Recognition"

   # Using Python 3.10
   py -3.10 -m venv venv

   # Or using Python 3.11
   py -3.11 -m venv venv

   # Activate
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install numpy==1.24.3 opencv-python==4.8.1.78 mediapipe==0.10.8 pandas matplotlib scikit-learn
   ```

4. **Test the system**:
   ```bash
   python test_system_simple.py
   ```

### Option 2: Update to New MediaPipe API (Advanced)

This requires significant code changes and is **NOT recommended for a thesis project** due to complexity and time constraints.

The new API (0.10.30+) requires:
- Downloading hand landmark model files
- Different initialization code
- Different result processing
- Less documentation available

###Option 3: Use OpenCV DNN with Custom Hand Detection (Not Recommended)

This would require training your own hand detection model, which is beyond the scope of this project.

## Recommended Action

**Install Python 3.11** (most stable for this project):

1. Download Python 3.11.x from https://www.python.org/downloads/
2. Install (don't uninstall Python 3.13, just install 3.11 alongside)
3. Create virtual environment with Python 3.11
4. Install mediapipe==0.10.8

## Why Python 3.10/3.11?

- ✅ Fully compatible with MediaPipe 0.10.8
- ✅ Stable with all required packages
- ✅ Extensive documentation available
- ✅ Used in most ASL recognition research
- ✅ TensorFlow/Keras full support

## After Installing Python 3.10/3.11

Run these commands:

```bash
# Navigate to project
cd "C:\Users\Adven\OneDrive\Documents\My files\Sign_Language_Recognition"

# Create virtual environment with Python 3.11
py -3.11 -m venv venv

# Activate
venv\Scripts\activate

# Verify Python version
python --version
# Should show: Python 3.11.x

# Install dependencies
pip install --upgrade pip
pip install numpy==1.24.3
pip install opencv-python==4.8.1.78
pip install mediapipe==0.10.8
pip install tensorflow==2.15.0
pip install pandas matplotlib seaborn scikit-learn

# Test
python test_system_simple.py
```

## For Your Thesis

In your thesis methodology chapter, mention:

> "The system was implemented using Python 3.11 with MediaPipe 0.10.8 for hand landmark detection, chosen for its stability, real-time performance, and extensive research validation."

---

## Quick Check: Which Python Versions Do You Have?

```bash
# Check all installed Python versions
py --list

# Example output:
#  -V:3.13 *        Python 3.13.9
#  -V:3.11          Python 3.11.5
#  -V:3.10          Python 3.10.11
```

The asterisk (*) shows your default version.

---

**Bottom Line**: Install Python 3.11, create a virtual environment, install mediapipe==0.10.8, and everything will work smoothly!
