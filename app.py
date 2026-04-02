"""
ASL Recognition System — Streamlit Web App
Evaluation dashboard — no TF/MediaPipe required on cloud.
"""

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title="ASL Recognition System",
    page_icon="🤟",
    layout="wide",
)

st.title("🤟 ASL Sign Language Recognition System")
st.caption("MLP (static gestures) · BiLSTM (dynamic gestures) · MediaPipe hand detection")

tab1, tab2, tab3 = st.tabs(["📊 Evaluation Results", "🖐 Live Demo", "ℹ️ About"])

PLOTS = Path("docs/plots")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Evaluation Results
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Model Evaluation Results")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Static Model Accuracy",  "96.8%",  "+29.8% vs baseline")
    col2.metric("Dynamic Model Accuracy", "97.2%")
    col3.metric("MediaPipe Detection FPS", "37.7 FPS")
    col4.metric("End-to-End Latency",     "~124 ms")

    st.divider()

    # ── Static model ──────────────────────────────────────────────────────────
    st.subheader("Static Gesture Model (MLP) — Letters A–Y + Digits 0–9")

    m_col, n_col = st.columns(2)
    with m_col:
        st.dataframe(pd.DataFrame({
            "Metric":  ["Accuracy", "Weighted Precision", "Weighted Recall",
                        "Weighted F1", "Macro Precision", "Macro Recall", "Macro F1"],
            "Value":   ["96.80%", "95.36%", "96.80%", "95.88%",
                        "80.80%", "83.24%", "81.22%"],
        }), hide_index=True, use_container_width=True)
    with n_col:
        st.info(
            "**Note:** Digits 0–9 have only 10 training samples each vs 100 for "
            "letters, which lowers macro averages. Letters A–Y achieve near-perfect scores."
        )

    for img, caption in [
        ("confusion_matrix_static.png",  "Confusion Matrix — Static Model"),
        ("per_class_accuracy_static.png","Per-Class Accuracy — Static Model"),
        ("error_analysis_static.png",    "Error Analysis — Static Model"),
        ("training_history_static.png",  "Training History — Static MLP"),
    ]:
        p = PLOTS / img
        if p.exists():
            st.image(str(p), caption=caption)

    st.divider()

    # ── Dynamic model ─────────────────────────────────────────────────────────
    st.subheader("Dynamic Gesture Model (BiLSTM) — J, Z + 10 Common Words")

    st.dataframe(pd.DataFrame({
        "Metric": ["Accuracy", "Weighted Precision", "Weighted Recall",
                   "Weighted F1", "Macro F1"],
        "Value":  ["97.22%", "97.35%", "97.22%", "97.22%", "97.22%"],
    }), hide_index=True, use_container_width=True)

    for img, caption in [
        ("confusion_matrix_dynamic.png",  "Confusion Matrix — Dynamic Model"),
        ("per_class_accuracy_dynamic.png","Per-Class Accuracy — Dynamic Model"),
        ("error_analysis_dynamic.png",    "Error Analysis — Dynamic Model"),
        ("training_history_dynamic.png",  "Training History — Dynamic BiLSTM"),
    ]:
        p = PLOTS / img
        if p.exists():
            st.image(str(p), caption=caption)

    st.divider()

    # ── Per-class breakdown ───────────────────────────────────────────────────
    st.subheader("Per-Class Results — Static Model (Letters)")
    letter_data = {
        "Class": ["A","B","C","D","E","F","G","H","I","K","L","M",
                  "N","O","P","Q","R","S","T","U","V","W","X","Y"],
        "Precision": [1.00,0.94,1.00,1.00,1.00,0.83,1.00,1.00,1.00,1.00,
                      1.00,1.00,1.00,1.00,1.00,0.94,0.88,1.00,1.00,1.00,
                      1.00,0.88,1.00,1.00],
        "Recall":    [1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,
                      1.00,1.00,1.00,1.00,0.93,1.00,1.00,1.00,1.00,0.87,
                      1.00,1.00,1.00,1.00],
        "F1":        [1.00,0.97,1.00,1.00,1.00,0.91,1.00,1.00,1.00,1.00,
                      1.00,1.00,1.00,1.00,0.97,0.97,0.94,1.00,1.00,0.93,
                      1.00,0.94,1.00,1.00],
    }
    st.dataframe(pd.DataFrame(letter_data), hide_index=True, use_container_width=True)

    st.subheader("Per-Class Results — Dynamic Model")
    dynamic_data = {
        "Gesture":   ["hello","help","no","please","sorry","thanks","yes","Z",
                      "J","stop","more","finish"],
        "Precision": [1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,0.88,0.93,0.87],
        "Recall":    [1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,0.87,1.00,0.93,0.87],
        "F1":        [1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,0.93,0.94,0.93,0.87],
    }
    st.dataframe(pd.DataFrame(dynamic_data), hide_index=True, use_container_width=True)

    st.divider()

    # ── Benchmark ─────────────────────────────────────────────────────────────
    st.subheader("Inference Speed Benchmark")
    st.dataframe(pd.DataFrame({
        "Component": ["MediaPipe Hand Detection",
                      "Static MLP Inference",
                      "Dynamic BiLSTM Inference"],
        "Mean Latency (ms)": [26.50, 94.02, 97.11],
        "Std Dev (ms)":      [8.28, 39.28, 16.26],
        "Throughput (FPS)":  [37.73, 10.64, 10.30],
    }), hide_index=True, use_container_width=True)

    st.divider()

    # ── Top confusion pairs ───────────────────────────────────────────────────
    st.subheader("Top Misclassifications")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Static Model**")
        st.dataframe(pd.DataFrame({
            "True": ["U","9","6","2","P"],
            "Predicted": ["R","F","W","3","Q"],
            "Count": [2,2,2,2,1],
            "Reason": ["Similar finger shape","Insufficient digit data",
                       "Insufficient digit data","Insufficient digit data",
                       "Near-identical pose"],
        }), hide_index=True, use_container_width=True)
    with c2:
        st.write("**Dynamic Model**")
        st.dataframe(pd.DataFrame({
            "True": ["J","finish","finish","more"],
            "Predicted": ["finish","stop","more","stop"],
            "Count": [2,1,1,1],
            "Reason": ["Curved motion overlap","Similar trajectory",
                       "Similar wrist motion","Similar finger motion"],
        }), hide_index=True, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Live Demo
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Live Gesture Demo")
    st.info(
        "**The real-time webcam demo runs locally only.**  \n"
        "TensorFlow and MediaPipe cannot be installed on Streamlit Cloud's free tier "
        "(Python 3.14, no compatible wheels).  \n\n"
        "**To run the demo on your own machine:**\n"
        "```bash\n"
        "git clone https://github.com/tresajoby/Sign_Language_Recognition\n"
        "cd Sign_Language_Recognition\n"
        "pip install tensorflow mediapipe opencv-python numpy\n"
        "python -m src.inference.run\n"
        "```"
    )

    st.subheader("Demo Screenshot")
    demo_img = PLOTS / "screenshot_demo.png"
    if demo_img.exists():
        st.image(str(demo_img), caption="Real-time ASL recognition running locally")
    else:
        st.markdown("""
        The live system displays:
        - **Static prediction** (letter/digit) with confidence bar
        - **Dynamic prediction** (word/motion letter) with confidence bar
        - **Active mode** indicator (STATIC / DYNAMIC)
        - **FPS counter** and **sequence buffer** progress
        - **Hand landmark overlay** drawn by MediaPipe
        """)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — About
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("About This System")
    st.markdown("""
    ## ASL Recognition System

    A real-time American Sign Language recognition prototype built as part of a
    final-year thesis project.

    ### Models

    | Component | Architecture | Classes | Accuracy |
    |---|---|---|---|
    | Static gestures | MLP (3 hidden layers) | A–Y (excl. J/Z), 0–9 | **96.8%** |
    | Dynamic gestures | BiLSTM | J, Z, hello, thanks, please, sorry, yes, no, help, stop, more, finish | **97.2%** |

    ### Pipeline
    1. **MediaPipe Hands** — detects 21 hand landmarks per frame (~37 FPS)
    2. **Wrist-relative normalisation** — 63-dimensional feature vector
    3. **Static MLP** — classifies single-frame hand poses
    4. **BiLSTM** — classifies 30-frame motion sequences

    ### Dataset
    | Split | Static | Dynamic |
    |---|---|---|
    | Train | 1,750 samples | 840 sequences |
    | Val | 375 samples | 180 sequences |
    | Test | 375 samples | 180 sequences |

    ### Tech Stack
    Python 3.11 · TensorFlow 2.21 · Keras 3 · MediaPipe 0.10.14 · OpenCV 4.x · Streamlit
    """)
