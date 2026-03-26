# ASL Recognition System - Setup Guide

This guide will help you set up your development environment for the ASL Recognition System thesis project.

---

## 🖥️ System Requirements

### Hardware Requirements
- **Processor**: Intel i5 or equivalent (minimum)
- **RAM**: 8GB (minimum), 16GB recommended
- **GPU**: Optional (NVIDIA GPU with CUDA for faster training)
- **Webcam**: Required for data collection and real-time inference
- **Storage**: 5GB free space

### Software Requirements
- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: 3.8 or higher
- **Git**: For version control

---

## 📋 Step-by-Step Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/tresajoby/Sign_Language_Recognition.git
cd Sign_Language_Recognition
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 3: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# Install the project in development mode (optional but recommended)
pip install -e .
```

### Step 4: Verify Installation

Run the configuration script to create directories and verify setup:

```bash
python -c "from src.utils.config import create_directories, print_configuration; create_directories(); print_configuration()"
```

You should see a detailed configuration printout without errors.

### Step 5: Test Webcam Access

Test if OpenCV can access your webcam:

```bash
python -c "import cv2; cap = cv2.VideoCapture(0); ret, frame = cap.read(); print('Webcam OK' if ret else 'Webcam FAILED'); cap.release()"
```

### Step 6: Test MediaPipe

Verify MediaPipe is working:

```bash
python -c "import mediapipe as mp; print('MediaPipe version:', mp.__version__)"
```

---

## 🧪 Running Tests

After completing the implementation, you can run tests:

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest --cov=src tests/
```

---

## 📊 Project Workflow

### 1. Data Collection Phase

```bash
# Collect static gesture data
python src/data_collection/collect_static.py

# Collect dynamic gesture sequences
python src/data_collection/collect_dynamic.py
```

### 2. Training Phase

```bash
# Train static gesture model
python src/training/train_static.py

# Train dynamic gesture model
python src/training/train_dynamic.py
```

### 3. Inference Phase

```bash
# Run real-time recognition
python src/inference/realtime_recognizer.py
```

### 4. Evaluation

```bash
# Evaluate models (after implementation)
python src/training/evaluate_static.py
python src/training/evaluate_dynamic.py
```

---

## 🐛 Troubleshooting

### Issue: "No module named 'cv2'"
**Solution**:
```bash
pip install opencv-python
```

### Issue: "Failed to load MediaPipe"
**Solution**:
```bash
pip uninstall mediapipe
pip install mediapipe==0.10.8
```

### Issue: "TensorFlow not found"
**Solution**:
```bash
pip install tensorflow==2.15.0
```

### Issue: Webcam not detected
**Solutions**:
1. Check if webcam is connected
2. Close other applications using the webcam
3. Try different camera ID (0, 1, 2) in config.py
4. On Linux, check permissions: `sudo usermod -a -G video $USER`

### Issue: "Permission denied" on macOS
**Solution**: Grant camera access to Terminal/IDE in System Preferences → Security & Privacy → Camera

---

## 🔧 Development Environment Setup

### Recommended IDE
- **VS Code** (recommended)
  - Extensions: Python, Pylance, Jupyter
- **PyCharm** (Community or Professional)
- **Jupyter Notebook** (for experimentation)

### VS Code Setup

1. Install Python extension
2. Select virtual environment:
   - `Ctrl+Shift+P` → "Python: Select Interpreter"
   - Choose `./venv/bin/python`

3. Install recommended extensions:
   ```json
   {
     "recommendations": [
       "ms-python.python",
       "ms-python.vscode-pylance",
       "ms-toolsai.jupyter",
       "visualstudioexptteam.vscodeintellicode"
     ]
   }
   ```

---

## 📁 Verifying Directory Structure

After setup, your directory structure should look like this:

```
Sign_Language_Recognition/
├── data/
│   ├── raw/
│   │   ├── static/
│   │   └── dynamic/
│   ├── processed/
│   └── labels/
├── src/
│   ├── data_collection/
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   ├── inference/
│   └── utils/
│       ├── config.py
│       └── __init__.py
├── models/
├── notebooks/
├── tests/
├── docs/
│   ├── architecture.md
│   ├── thesis_notes.md
│   ├── references.bib
│   └── plots/
├── requirements.txt
├── README.md
├── .gitignore
└── setup.py
```

---

## 🎯 Next Steps

1. ✅ Complete setup (you're here!)
2. ⏭️ Proceed to Step 2: Data Collection & Preprocessing (when ready)
3. Build static gesture model
4. Build dynamic gesture model
5. Implement real-time inference
6. Evaluate and document results

---

## 📚 Useful Resources

### Documentation
- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

### Community
- [TensorFlow Forum](https://discuss.tensorflow.org/)
- [OpenCV Forum](https://forum.opencv.org/)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/mediapipe)

---

## 💡 Tips for Thesis Development

1. **Version Control**: Commit frequently with descriptive messages
2. **Documentation**: Document as you code, not after
3. **Experimentation**: Keep a lab notebook (use notebooks/ folder)
4. **Reproducibility**: Always set random seeds
5. **Backups**: Back up your data and models regularly

---

## 📧 Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Search GitHub issues
3. Ask your thesis supervisor
4. Post on relevant forums with detailed error messages

---

## ✅ Setup Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed
- [ ] Webcam accessible
- [ ] MediaPipe working
- [ ] TensorFlow working
- [ ] Directory structure created
- [ ] Configuration script runs successfully
- [ ] Git repository initialized
- [ ] Ready to start data collection!

---

**🎉 Congratulations! Your development environment is ready.**

Proceed to the next step when instructed by your supervisor.
