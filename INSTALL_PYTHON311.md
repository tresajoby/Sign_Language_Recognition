# Installing Python 3.11 - Step-by-Step Guide

## Why Python 3.11?

Python 3.13 (your current version) is too new for MediaPipe 0.10.8 (the version with the hand detection API we need). Python 3.11 is the sweet spot: stable, well-supported, and compatible with all our packages.

---

## Step 1: Download Python 3.11

### Option A: Direct Download (Recommended)

1. **Go to**: https://www.python.org/downloads/release/python-3119/

2. **Scroll down** to "Files" section at the bottom

3. **Download** the appropriate installer for your system:
   - **For Windows 64-bit**: `Windows installer (64-bit)`
   - File name: `python-3.11.9-amd64.exe`
   - Size: ~27 MB

4. **Save** the file to your Downloads folder

### Option B: From Main Downloads Page

1. **Go to**: https://www.python.org/downloads/
2. **Look for**: Python 3.11.x in the releases list
3. **Click**: Download button for Python 3.11.9
4. **Choose**: Windows installer (64-bit)

---

## Step 2: Install Python 3.11

### Installation Steps

1. **Locate** the downloaded file:
   - Usually in: `C:\Users\Adven\Downloads\python-3.11.9-amd64.exe`

2. **Double-click** the installer to run it

3. **IMPORTANT**: On the first screen, CHECK both boxes:
   ```
   ☑ Install launcher for all users (recommended)
   ☑ Add python.exe to PATH
   ```
   **This is CRITICAL!**

4. **Click**: "Customize installation" (not "Install Now")

5. **Optional Features screen** - Check all:
   ```
   ☑ Documentation
   ☑ pip
   ☑ tcl/tk and IDLE
   ☑ Python test suite
   ☑ py launcher
   ☑ for all users (requires admin privileges)
   ```

6. **Click**: Next

7. **Advanced Options screen** - Check these:
   ```
   ☑ Install for all users
   ☑ Associate files with Python (requires the py launcher)
   ☑ Create shortcuts for installed applications
   ☑ Add Python to environment variables
   ☑ Precompile standard library
   ```

8. **Customize install location** (optional):
   - Default: `C:\Program Files\Python311\`
   - You can keep this default

9. **Click**: Install

10. **Wait** for installation to complete (~2-3 minutes)

11. **Click**: Close when finished

---

## Step 3: Verify Installation

1. **Open a NEW Command Prompt** (or PowerShell):
   - Press `Win + R`
   - Type `cmd`
   - Press Enter

2. **Check Python 3.11 is installed**:
   ```cmd
   py -3.11 --version
   ```

   **Expected output**:
   ```
   Python 3.11.9
   ```

3. **Check pip is available**:
   ```cmd
   py -3.11 -m pip --version
   ```

   **Expected output**:
   ```
   pip 24.x.x from ... (python 3.11)
   ```

4. **List all Python versions**:
   ```cmd
   py --list
   ```

   **Expected output**:
   ```
   -V:3.13 *        Python 3.13.9
   -V:3.11          Python 3.11.9
   ```

   (The asterisk shows your default version)

---

## Step 4: Create Virtual Environment for Your Project

Now let's set up Python 3.11 specifically for your ASL project:

1. **Open Command Prompt**

2. **Navigate to your project**:
   ```cmd
   cd "C:\Users\Adven\OneDrive\Documents\My files\Sign_Language_Recognition"
   ```

3. **Create virtual environment with Python 3.11**:
   ```cmd
   py -3.11 -m venv venv
   ```

   This creates a folder called `venv` with Python 3.11

4. **Activate the virtual environment**:
   ```cmd
   venv\Scripts\activate
   ```

   **You should see**: `(venv)` at the start of your command line

5. **Verify you're using Python 3.11**:
   ```cmd
   python --version
   ```

   **Should show**: `Python 3.11.9`

---

## Step 5: Install Project Dependencies

Now install all packages for the ASL project:

```cmd
# Make sure venv is activated (you see "(venv)" in prompt)

# Upgrade pip first
python -m pip install --upgrade pip

# Install core packages
pip install numpy==1.24.3
pip install opencv-python==4.8.1.78
pip install mediapipe==0.10.8

# Install TensorFlow
pip install tensorflow==2.15.0

# Install other dependencies
pip install pandas matplotlib seaborn scikit-learn

# Verify MediaPipe installation
python -c "import mediapipe as mp; print('MediaPipe version:', mp.__version__)"
```

**Expected output**:
```
MediaPipe version: 0.10.8
```

---

## Step 6: Test Your Setup

Run the system test:

```cmd
python test_system_simple.py
```

**Expected output**:
```
======================================================================
ASL RECOGNITION SYSTEM - COMPONENT TEST
======================================================================

[TEST 1] Testing Configuration...
[OK] Configuration module working
     - Static classes: 36
     - Dynamic classes: 10
     - Feature dimension: 63

[TEST 2] Testing Hand Detector...
[OK] Hand detector initialized
     - Max hands: 1
     - Detection confidence: 0.7
     - Detector callable: [OK]

[TEST 3] Testing Feature Extractor...
[OK] Feature extractor initialized
...

[SUCCESS] All core components working!
```

---

## Troubleshooting

### Problem 1: "py: command not found"

**Solution**: Restart Command Prompt after installation

### Problem 2: Python 3.11 not showing in `py --list`

**Solution**:
1. Reinstall Python 3.11
2. Make sure to check "Add to PATH" during installation
3. Restart computer

### Problem 3: "Import error: mediapipe"

**Solution**:
```cmd
# Make sure venv is activated
venv\Scripts\activate

# Reinstall mediapipe
pip uninstall mediapipe
pip install mediapipe==0.10.8
```

### Problem 4: Virtual environment activation fails

**Solution**:
```cmd
# Try this alternative activation method
venv\Scripts\activate.bat

# Or use PowerShell
venv\Scripts\Activate.ps1
```

If PowerShell gives error about execution policy:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Quick Reference Commands

```cmd
# Check Python version
py -3.11 --version

# Create virtual environment
py -3.11 -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Deactivate virtual environment
deactivate

# Install package
pip install package_name

# List installed packages
pip list

# Upgrade pip
python -m pip install --upgrade pip
```

---

## What's Next?

After successful installation:

1. ✅ Python 3.11 installed
2. ✅ Virtual environment created
3. ✅ All packages installed
4. ✅ System test passed

**You can now**:
- Collect your dataset
- Train your models
- Run real-time recognition

---

## Important Notes

- **Always activate the virtual environment** before working on the project:
  ```cmd
  cd "C:\Users\Adven\OneDrive\Documents\My files\Sign_Language_Recognition"
  venv\Scripts\activate
  ```

- **You'll see `(venv)`** in your command prompt when activated

- **To deactivate**: Just type `deactivate`

- **Python 3.13 is still there**: Your other Python projects won't be affected

---

## Visual Confirmation Checklist

After installation, you should be able to run these without errors:

```cmd
☐ py -3.11 --version                → Shows "Python 3.11.9"
☐ py --list                         → Shows both 3.13 and 3.11
☐ cd to project folder              → Navigate successfully
☐ py -3.11 -m venv venv            → Creates venv folder
☐ venv\Scripts\activate            → Shows "(venv)" in prompt
☐ python --version                 → Shows "Python 3.11.9"
☐ pip install mediapipe==0.10.8    → Installs successfully
☐ python test_system_simple.py    → All tests pass
```

---

**Once all checks pass, you're ready to start collecting data and training your models!** 🚀
