# Complete Data Collection Guide for ASL Recognition Thesis

## Overview
You're collecting data for **36 static gesture classes** (A-Z letters + 0-9 digits) for your thesis project.

---

## Recommended Dataset Size

### For Thesis-Quality Results:
- **Target per gesture**: 100-300 samples
- **Minimum acceptable**: 50 samples per gesture
- **Quick testing**: 20 samples per gesture

### Why More Samples Matter:
- ✅ Better model accuracy (important for thesis evaluation)
- ✅ More robust to variations (hand position, lighting, angles)
- ✅ Better train/validation/test split (70/15/15)
- ✅ Stronger academic justification

---

## Data Collection Strategy

### Phase 1: Quick Validation (Day 1)
**Goal**: Verify the system works end-to-end

**Collect**:
- 20 samples for digits 0-9 (10 classes)
- 20 samples for letters A-J (10 classes)
- **Total**: 400 samples

**Time**: ~30-45 minutes

**Purpose**: Train a quick model to verify everything works before investing more time.

---

### Phase 2: Complete Dataset (Days 2-5)
**Goal**: Collect thesis-quality dataset

**Collect**:
- 100+ samples for ALL digits (0-9)
- 100+ samples for ALL letters (A-Z)
- **Total**: 3,600+ samples

**Time**: ~3-4 hours (split across multiple sessions)

**Purpose**: Final thesis model with high accuracy.

---

## How to Collect Data

### Step 1: Activate Environment
```cmd
cd "C:\Users\Adven\OneDrive\Documents\My files\Sign_Language_Recognition"
venv\Scripts\activate
```

**Verify**: You should see `(venv)` in your prompt.

---

### Step 2: Start Collection Script
```cmd
python src/data_collection/collect_static.py
```

---

### Step 3: Collection Interface

The interface will show:
```
===== ASL STATIC GESTURE DATA COLLECTION =====

Instructions:
1. Press the key (0-9, A-Z) for the gesture you want to collect
2. Show the gesture to the camera and press SPACE to capture
3. The system will capture when a hand is detected
4. Press Q to quit and process data

Current Progress:
[Shows how many samples collected per gesture]

Ready! Press a gesture key to start...
```

---

### Step 4: Collection Tips

#### For Best Results:
1. **Hand Position**: Center your hand in the frame
2. **Distance**: Keep hand 1-2 feet from camera
3. **Lighting**: Use good lighting (face a window or light source)
4. **Background**: Keep background uncluttered
5. **Variations**: Slightly vary position/angle between samples
   - Move hand left/right a bit
   - Tilt hand slightly
   - Change distance slightly
   - This makes the model more robust!

#### Collection Workflow:
1. Press the gesture key (e.g., press `A` to collect letter A)
2. Show the ASL sign for that gesture
3. Press SPACE to capture
4. Repeat until you have enough samples for that gesture
5. Move to next gesture

#### Taking Breaks:
- You can quit anytime (press Q)
- Progress is automatically saved
- Resume later by running the script again
- It remembers where you left off!

---

## Data Collection Sessions

### Recommended Schedule:

#### Session 1 (30 minutes): Digits
- Collect 20-100 samples each for: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- **Total**: 200-1000 samples

#### Session 2 (45 minutes): Letters A-M
- Collect 20-100 samples each for: A through M
- **Total**: 260-1300 samples

#### Session 3 (45 minutes): Letters N-Z
- Collect 20-100 samples each for: N through Z
- **Total**: 260-1300 samples

#### Session 4 (Optional - 1 hour): Boost Low-Count Classes
- Review progress
- Add more samples to classes with fewer samples
- Aim for balanced dataset

---

## Progress Tracking

### Check Your Progress:
The script shows real-time progress:
```
Current Progress:
A: 45/100  [=============>        ] 45%
B: 78/100  [====================> ] 78%
0: 100/100 [======================] 100%
...
```

### Progress File:
Location: `data/raw/static/collection_progress.json`

View progress anytime:
```cmd
cat data/raw/static/collection_progress.json
```

---

## After Collection Complete

### Automatic Processing:
When you press Q to quit:
1. Script automatically processes all raw landmark files
2. Creates normalized feature vectors (wrist-relative normalization)
3. Saves to: `data/processed/static_features.npy`
4. Saves labels to: `data/processed/static_labels.npy`

### Verify Collection:
```cmd
python -c "import numpy as np; features = np.load('data/processed/static_features.npy'); labels = np.load('data/processed/static_labels.npy'); print(f'Total samples: {len(features)}'); print(f'Feature shape: {features.shape}'); print(f'Classes: {np.unique(labels)}')"
```

Expected output:
```
Total samples: 3600
Feature shape: (3600, 63)
Classes: ['0' '1' '2' ... 'X' 'Y' 'Z']
```

---

## Troubleshooting

### Issue: "No hand detected"
**Solution**:
- Move hand closer to camera
- Improve lighting
- Keep hand flat and visible
- Check camera is working

### Issue: "Wrong gesture detected"
**Solution**:
- This is normal - the script saves whatever you capture
- You're training the model to recognize YOUR gestures
- Be consistent in how you show each gesture

### Issue: "Collection is slow"
**Solution**:
- You can collect fewer samples per gesture (minimum 20)
- Take breaks - progress is saved
- Split collection across multiple days

### Issue: "Script crashed"
**Solution**:
- Your progress is saved
- Just run the script again to resume
- It will pick up where you left off

---

## Quick Start Commands

### Start Fresh Collection:
```cmd
cd "C:\Users\Adven\OneDrive\Documents\My files\Sign_Language_Recognition"
venv\Scripts\activate
python src/data_collection/collect_static.py
```

### Resume Collection:
```cmd
cd "C:\Users\Adven\OneDrive\Documents\My files\Sign_Language_Recognition"
venv\Scripts\activate
python src/data_collection/collect_static.py
```
(Same command - it auto-resumes!)

### Check Progress:
```cmd
cat data/raw/static/collection_progress.json
```

---

## Next Steps After Collection

Once you have collected sufficient data:

1. **Train Static Model** (Step 3)
   ```cmd
   python src/training/train_static.py
   ```

2. **Test Real-Time Recognition** (Step 4)
   ```cmd
   python src/inference/realtime_static.py
   ```

3. **Collect Dynamic Gestures** (for words like "hello", "thanks")
   ```cmd
   python src/data_collection/collect_dynamic.py
   ```

---

## Tips for Thesis Success

### Data Quality > Quantity
- 50 good samples beats 200 sloppy samples
- Be consistent in how you show each gesture
- Use good lighting

### Document Your Process
- Take screenshots of collection progress
- Note any challenges you faced
- This goes in your thesis methodology section!

### Balanced Dataset
- Try to collect similar number of samples for each gesture
- Model performs best with balanced data

### Test Early, Iterate
- Start with 20 samples per gesture
- Train and test
- Then collect more data to improve accuracy

---

**Ready to start collecting? Run the commands above and let me know if you have any questions!**
