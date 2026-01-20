# Why J and Z are Dynamic Gestures

## Classification Decision

In this ASL Recognition System, letters **J** and **Z** are classified as **dynamic gestures** rather than static gestures.

---

## Academic Justification

### Static vs Dynamic Gestures in ASL

**Static Gestures**:
- Can be represented by a **single hand pose**
- No movement required
- Examples: Letters A, B, C, D, E, F, etc.
- Can be recognized from a single image/frame

**Dynamic Gestures**:
- Require **temporal information** (movement over time)
- Defined by a sequence of hand positions
- Examples: J, Z, and words like "hello", "thanks"
- Require video sequences for recognition

---

## Specific Cases: J and Z

### Letter J
- **Movement**: Draws a "J" shape in the air
- **Description**: Hand starts in a position and moves downward, then curves to the left
- **Why Dynamic**: The shape traced by the movement is essential to the sign
- **Frames Required**: ~30 frames to capture the complete motion

### Letter Z
- **Movement**: Draws a "Z" shape in the air
- **Description**: Hand traces a diagonal down-right, then horizontal left, then diagonal down-right
- **Why Dynamic**: The zig-zag motion pattern defines the sign
- **Frames Required**: ~30 frames to capture the complete motion

---

## Implementation Impact

### Static Model (MLP)
- **Classes**: 34 total
  - 24 letters (A-Y excluding J and Z)
  - 10 digits (0-9)
- **Input**: Single frame (63 features)
- **Architecture**: Multi-Layer Perceptron

### Dynamic Model (BiLSTM)
- **Classes**: 12 total
  - 2 letters (J, Z)
  - 10 words (hello, thanks, please, sorry, yes, no, help, stop, more, finish)
- **Input**: Sequence of 30 frames (30 × 63 features)
- **Architecture**: Bidirectional LSTM

---

## Data Collection Strategy

### For Static Gestures (A-Y except J, Z, plus 0-9):
```cmd
python src/data_collection/collect_static.py
```
- Press the gesture key (e.g., 'A', '5')
- Show the static hand pose
- Press SPACE to capture
- No movement required

### For Dynamic Gestures (J, Z, hello, thanks, etc.):
```cmd
python src/data_collection/collect_dynamic.py
```
- Select the gesture from menu
- Perform the complete motion
- System records 30 frames automatically
- Movement is essential

---

## References for Thesis

This classification follows standard ASL linguistic categorization:

1. **Stokoe, W. C.** (1960). Sign language structure: An outline of the visual communication systems of the American deaf. *Studies in linguistics: Occasional papers*.

2. **Wilbur, R. B.** (2009). Effects of varying rate of signing on ASL manual signs and nonmanual markers. *Language and speech*, 52(2-3), 245-285.

3. The distinction between static and dynamic signs is well-established in ASL linguistics and sign language recognition research.

---

## For Your Thesis

### Methodology Section

You can write:

> "The 26 letters of the alphabet were classified into static (24 letters) and dynamic (2 letters) based on ASL linguistic properties. Letters A-I and K-Y exhibit static hand configurations that can be recognized from single frames. However, letters J and Z require temporal analysis as they are defined by traced movements (J draws a 'J' shape, Z draws a 'Z' shape). Consequently, J and Z were grouped with dynamic word-level signs and processed using a BiLSTM architecture capable of modeling temporal sequences."

### Results Section

This gives you an opportunity to:
- Compare static vs dynamic recognition accuracy
- Discuss the trade-offs (dynamic gestures typically have lower accuracy)
- Explain why separate models are necessary
- Show understanding of ASL linguistic properties

---

## Summary

**Static Gestures** (34 classes):
- A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y
- 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

**Dynamic Gestures** (12 classes):
- J, Z (alphabet letters with movement)
- hello, thanks, please, sorry, yes, no, help, stop, more, finish (words)

This classification is **linguistically correct** and **computationally appropriate** for your thesis.
