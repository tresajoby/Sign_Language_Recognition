# Quick Fix - How to Run Your Code

## The Problem
You're getting "No module named 'mediapipe'" because the virtual environment isn't activated.

## The Solution (3 Simple Steps)

### Step 1: Open Command Prompt
- Press `Win + R`
- Type `cmd`
- Press Enter

### Step 2: Navigate and Activate
```cmd
cd "C:\Users\Adven\OneDrive\Documents\My files\Sign_Language_Recognition"
venv\Scripts\activate
```

**IMPORTANT:** After running `activate`, your prompt should look like:
```
(venv) C:\Users\Adven\OneDrive\Documents\My files\Sign_Language_Recognition>
```

Notice the **(venv)** at the start - this means it worked!

### Step 3: Run Your Test
```cmd
python test_system_simple.py
```

This should now work! You'll see all tests pass.

---

## Why This Works

- ❌ **Without (venv)**: Uses system Python → No MediaPipe
- ✅ **With (venv)**: Uses project Python → MediaPipe 0.10.8 installed

---

## Every Time You Work on This Project

**Always start with these two commands:**
```cmd
cd "C:\Users\Adven\OneDrive\Documents\My files\Sign_Language_Recognition"
venv\Scripts\activate
```

Look for **(venv)** in your prompt before running any `python` command!

---

## Alternative: Use the Batch File

Double-click this file in your project folder:
```
start_project.bat
```

It will automatically activate the venv for you!

---

## Troubleshooting

**Q: I don't see (venv) after running activate**

Try:
```cmd
venv\Scripts\activate.bat
```

**Q: PowerShell says "scripts disabled"**

Use Command Prompt (cmd) instead - it's simpler!

**Q: Still getting "No module named 'mediapipe'"**

Make sure you see **(venv)** in your prompt first. If you do and still get the error:
```cmd
pip list
```

Look for mediapipe in the list. If it's there but still not working, restart Command Prompt and try again.

---

**Your Python 3.11.9 and MediaPipe 0.10.8 are perfect - you just need to activate the venv!**
