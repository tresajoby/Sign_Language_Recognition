# How to Activate Virtual Environment - Quick Guide

## The Problem

When you run Python commands, they're using the system Python (without MediaPipe), not the virtual environment Python (with MediaPipe 0.10.8).

## The Solution: Always Activate First!

---

## Method 1: Command Prompt (cmd) - RECOMMENDED ✅

### Step 1: Open Command Prompt

Press `Win + R`, type `cmd`, press Enter

### Step 2: Navigate to Project

```cmd
cd "C:\Users\Adven\OneDrive\Documents\My files\Sign_Language_Recognition"
```

### Step 3: Activate Virtual Environment

```cmd
venv\Scripts\activate.bat
```

or simply:

```cmd
venv\Scripts\activate
```

### Step 4: Verify Activation

You should see `(venv)` at the start of your prompt:

```
(venv) C:\Users\Adven\OneDrive\Documents\My files\Sign_Language_Recognition>
```

### Step 5: Now Run Your Commands

```cmd
python test_system_simple.py
```

---

## Method 2: PowerShell (Requires Setup)

### Step 1: Open PowerShell as Administrator

Press `Win + X`, select "Windows PowerShell (Admin)"

### Step 2: Enable Script Execution (ONE TIME ONLY)

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Type `Y` and press Enter

### Step 3: Close Administrator PowerShell

Close it and open regular PowerShell

### Step 4: Navigate to Project

```powershell
cd "C:\Users\Adven\OneDrive\Documents\My files\Sign_Language_Recognition"
```

### Step 5: Activate

```powershell
venv\Scripts\Activate.ps1
```

or:

```powershell
venv\Scripts\activate
```

---

## Method 3: Double-Click Batch File (EASIEST) ⭐

### Just Double-Click This File:

```
start_project.bat
```

Located in your project folder. It will:
- ✅ Automatically activate the virtual environment
- ✅ Show you're in (venv)
- ✅ Display quick commands

---

## Method 4: VS Code Terminal (If Using VS Code)

### Step 1: Open Project in VS Code

File → Open Folder → Select your project folder

### Step 2: Open Terminal in VS Code

Terminal → New Terminal (or Ctrl + `)

### Step 3: Select Command Prompt

If it opens PowerShell, click the dropdown (top right of terminal) and select "Command Prompt"

### Step 4: Activate

```cmd
venv\Scripts\activate
```

---

## How to Know If Virtual Environment is Active

### ✅ ACTIVE (Correct)
```
(venv) C:\Users\Adven\OneDrive\Documents\My files\Sign_Language_Recognition>
```
You'll see **(venv)** at the start!

### ❌ NOT ACTIVE (Wrong)
```
C:\Users\Adven\OneDrive\Documents\My files\Sign_Language_Recognition>
```
No (venv) - packages won't be found!

---

## Quick Test After Activation

Once you see `(venv)` in your prompt, test:

```cmd
python --version
```
Should show: `Python 3.11.9`

```cmd
python -c "import mediapipe as mp; print(mp.__version__)"
```
Should show: `MediaPipe version: 0.10.8`

```cmd
python test_system_simple.py
```
Should run all tests successfully!

---

## Common Mistakes

### ❌ Mistake 1: Forgetting to Activate
```cmd
# Wrong - no activation
cd project
python test_system_simple.py  # ERROR: No module named 'mediapipe'
```

### ✅ Correct Way
```cmd
# Correct - activate first
cd project
venv\Scripts\activate
python test_system_simple.py  # Works!
```

### ❌ Mistake 2: Wrong Directory
```cmd
# Wrong - not in project folder
C:\Users\Adven> venv\Scripts\activate  # ERROR: venv not found
```

### ✅ Correct Way
```cmd
# Correct - navigate first
cd "C:\Users\Adven\OneDrive\Documents\My files\Sign_Language_Recognition"
venv\Scripts\activate  # Works!
```

---

## Daily Workflow

**Every time** you work on your project:

1. Open Command Prompt (Win + R → `cmd`)
2. Navigate: `cd "C:\Users\Adven\OneDrive\Documents\My files\Sign_Language_Recognition"`
3. Activate: `venv\Scripts\activate`
4. Verify: Look for `(venv)` in prompt
5. Work: Run your Python commands
6. Deactivate when done: `deactivate`

---

## Troubleshooting

### Problem: "venv\Scripts\activate is not recognized"

**Solution**: Make sure you're in the project directory first:
```cmd
cd "C:\Users\Adven\OneDrive\Documents\My files\Sign_Language_Recognition"
```

### Problem: "Cannot be loaded because running scripts is disabled" (PowerShell)

**Solution**: Use Command Prompt instead, or fix PowerShell execution policy (see Method 2 above)

### Problem: Still getting "No module named 'mediapipe'"

**Solution**:
1. Verify you see `(venv)` in your prompt
2. If not, activation didn't work - try again
3. If yes, reinstall MediaPipe:
```cmd
pip install mediapipe==0.10.8
```

### Problem: "python is not recognized"

**Solution**: Make sure Python 3.11 is installed and added to PATH. Or use:
```cmd
py -3.11 -m pip install mediapipe==0.10.8
```

---

## Quick Reference Card

| Task | Command |
|------|---------|
| Navigate to project | `cd "C:\Users\Adven\OneDrive\Documents\My files\Sign_Language_Recognition"` |
| Activate (cmd) | `venv\Scripts\activate` |
| Activate (PowerShell) | `venv\Scripts\Activate.ps1` |
| Check if active | Look for `(venv)` in prompt |
| Test Python version | `python --version` |
| Test MediaPipe | `python -c "import mediapipe as mp; print(mp.__version__)"` |
| Run system test | `python test_system_simple.py` |
| Deactivate | `deactivate` |

---

## Remember

**The Golden Rule**: Always see `(venv)` before running Python commands!

```
(venv) C:\...> python script.py  ✅ Will work
C:\...> python script.py         ❌ Won't find packages
```

---

## Already Have a Terminal Open?

If you're in the middle of work and realize you forgot to activate:

```cmd
# Don't close the terminal!
# Just activate now:
venv\Scripts\activate

# Now you're good to go
python test_system_simple.py
```

---

**Need help? Just double-click `start_project.bat` - it does everything for you!**
