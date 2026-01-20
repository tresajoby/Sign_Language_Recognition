# START HERE - Data Collection Steps

## Everything is cleaned! Ready for fresh start.

---

## STEP 1: Open Command Prompt & Activate Environment

### Copy and paste these commands one by one:

```cmd
cd "C:\Users\Adven\OneDrive\Documents\My files\Sign_Language_Recognition"
```

Press Enter, then:

```cmd
venv\Scripts\activate
```

Press Enter.

**IMPORTANT**: You MUST see **(venv)** at the start of your prompt!

It should look like:
```
(venv) C:\Users\Adven\OneDrive\Documents\My files\Sign_Language_Recognition>
```

---

## STEP 2: Start the Collection Script

```cmd
python src/data_collection/collect_static.py
```

A webcam window will open.

---

## STEP 3: Understanding the Interface

You'll see:
```
===== ASL STATIC GESTURE DATA COLLECTION =====

Instructions:
1. Press the key (0-9, A-Z) for the gesture you want to collect
2. Show the gesture to the camera and press SPACE to capture
3. Press Q to quit and process data

Current Progress:
(empty at start)

Ready! Press a gesture key to start...
```

---

## STEP 4: Collect Your First Gesture (Digit "0")

### Follow these steps exactly:

1. **Press the key `0`** (zero) on your keyboard
   - You'll see: "Collecting: 0"

2. **Make the ASL sign for zero** with your hand in front of the camera
   - Keep your hand centered in the view
   - Distance: About 1-2 feet from camera

3. **Press SPACE** to capture
   - You'll see a green flash and "Captured sample 1/300"
   - The file is saved automatically

4. **Slightly move your hand** (a little left, right, closer, or farther)

5. **Press SPACE again** to capture sample 2

6. **Repeat** until you have collected **20 samples** for digit "0"
   - Keep varying your hand position slightly
   - This makes your model robust!

---

## STEP 5: Move to Next Gesture

Once you have 20 samples for "0":

1. **Press the key `1`** on your keyboard
2. **Make the ASL sign for "1"**
3. **Press SPACE** 20 times (with slight variations)
4. Continue with digits: 2, 3, 4, 5, 6, 7, 8, 9

---

## STEP 6: Collect Letters

After completing digits 0-9:

1. **Press the key `A`** on your keyboard
2. **Make the ASL sign for letter "A"**
3. **Press SPACE** 20 times (with slight variations)
4. Continue with: B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y

**IMPORTANT**: Skip **J** and **Z** - these require movement and will be collected separately as dynamic gestures!

---

## Tips for Good Data:

### DO:
✅ **Center your hand** in the camera view
✅ **Use good lighting** (face a window or light source)
✅ **Vary position slightly** between samples:
   - Move hand left/right a bit
   - Move hand closer/farther
   - Tilt hand slightly
✅ **Be consistent** with the gesture shape
✅ **Take breaks** - Press Q to quit anytime, progress is saved!

### DON'T:
❌ Don't collect all samples in exact same position
❌ Don't block the hand with your body
❌ Don't use dim lighting
❌ Don't change the gesture shape itself

---

## Progress Tracking

The interface shows your progress:
```
Current Progress:
0: 20/300  [===>              ] 6.7%
1: 20/300  [===>              ] 6.7%
A: 15/300  [==>               ] 5.0%
...
```

---

## Taking Breaks

You can quit ANYTIME:
1. **Press Q** to quit
2. The script will automatically process your data
3. To resume later, just run the script again:
   ```cmd
   python src/data_collection/collect_static.py
   ```
4. It will remember where you left off!

---

## Recommended Collection Plan

### Session 1 (20 minutes): Digits
- Collect 20 samples each for: **0, 1, 2, 3, 4, 5, 6, 7, 8, 9**
- Total: 200 samples
- Press Q when done

### Session 2 (25 minutes): Letters A-M (Skip J)
- Collect 20 samples each for: **A, B, C, D, E, F, G, H, I, K, L, M**
- Total: 240 samples (12 letters)
- Press Q when done

### Session 3 (25 minutes): Letters N-Y (Skip Z)
- Collect 20 samples each for: **N, O, P, Q, R, S, T, U, V, W, X, Y**
- Total: 240 samples (12 letters)
- Press Q when done

**Grand Total for Static Gestures**: 680 samples (20 per class × 34 classes)

**Note**: Letters **J** and **Z** will be collected later as dynamic gestures using the dynamic collection script!

---

## What Happens When You Press Q?

The script automatically:
1. Processes all raw landmark files
2. Applies wrist-relative normalization
3. Creates feature vectors (63 dimensions each)
4. Saves to: `data/processed/static_features.npy`
5. Saves labels to: `data/processed/static_labels.npy`

You'll see:
```
PROCESSING COLLECTED DATA
Processing gesture: 0
Processing gesture: 1
...
Processing complete!
Saved 720 samples to data/processed/
```

---

## Troubleshooting

### Problem: "No hand detected"
**Solution**:
- Move hand closer to camera
- Improve lighting
- Make sure hand is visible and not blocked

### Problem: "Camera not opening"
**Solution**:
- Close any other apps using the camera
- Try running the script again
- Check if camera is working (test with another app)

### Problem: "Script closed unexpectedly"
**Solution**:
- Your progress IS saved
- Just run the script again
- It will resume automatically

### Problem: "I pressed wrong key"
**Solution**:
- Just press the correct gesture key
- The script will switch to that gesture
- Previous captures are still saved

---

## After Collection Complete

Once you've collected all samples and pressed Q:

### Verify Your Data:
```cmd
python -c "import numpy as np; f = np.load('data/processed/static_features.npy'); l = np.load('data/processed/static_labels.npy'); print(f'Total: {len(f)} samples'); print(f'Shape: {f.shape}'); import collections; print('Per class:', dict(collections.Counter(l)))"
```

Expected output:
```
Total: 720 samples
Shape: (720, 63)
Per class: {'0': 20, '1': 20, '2': 20, ... 'Z': 20}
```

---

## Next Step: Train Your Model

After successful collection:
```cmd
python src/training/train_static.py
```

This will train your recognition model!

---

## START NOW!

### Run these commands:

```cmd
cd "C:\Users\Adven\OneDrive\Documents\My files\Sign_Language_Recognition"
venv\Scripts\activate
python src/data_collection/collect_static.py
```

**Start with digit "0"** - Press `0`, make the sign, press SPACE 20 times!

Good luck! Let me know when you've collected some samples! 🎯
