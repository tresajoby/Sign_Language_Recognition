"""
ASL Recognition System — Streamlit Web App
Displays evaluation results and runs static gesture inference on uploaded images.
"""

import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ASL Recognition System",
    page_icon="🤟",
    layout="wide",
)

st.title("🤟 ASL Sign Language Recognition System")
st.caption("MLP (static gestures) · BiLSTM (dynamic gestures) · MediaPipe hand detection")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Evaluation Results", "🖐 Live Demo", "ℹ️ About"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Evaluation Results
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Model Evaluation Results")

    # ── Summary metrics ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Static Model Accuracy", "96.8%", "+29.8% vs baseline")
    col2.metric("Dynamic Model Accuracy", "97.2%")
    col3.metric("MediaPipe Detection FPS", "37.7 FPS")
    col4.metric("End-to-End Latency", "~124 ms")

    st.divider()

    # ── Static model ──
    st.subheader("Static Gesture Model (MLP) — Letters A–Y + Digits 0–9")

    metrics_col, note_col = st.columns([1, 1])
    with metrics_col:
        import pandas as pd
        static_df = pd.DataFrame({
            "Metric": ["Accuracy", "Weighted Precision", "Weighted Recall",
                       "Weighted F1", "Macro Precision", "Macro Recall", "Macro F1"],
            "Value": ["96.80%", "95.36%", "96.80%", "95.88%",
                      "80.80%", "83.24%", "81.22%"],
        })
        st.dataframe(static_df, hide_index=True, use_container_width=True)

    with note_col:
        st.info(
            "**Note on macro vs weighted scores:** Digits 0–9 have only 10 training "
            "samples each (vs 100 for letters), which lowers macro averages. "
            "Letters A–Y achieve near-perfect scores individually."
        )

    plots_dir = Path("docs/plots")
    cm_static = plots_dir / "confusion_matrix_static.png"
    pc_static  = plots_dir / "per_class_accuracy_static.png"
    ea_static  = plots_dir / "error_analysis_static.png"

    if cm_static.exists():
        c1, c2 = st.columns(2)
        with c1:
            st.image(str(cm_static), caption="Confusion Matrix — Static Model")
        with c2:
            st.image(str(pc_static), caption="Per-Class Accuracy — Static Model")
        st.image(str(ea_static), caption="Error Analysis — Static Model")
    else:
        st.warning("Run `python -m src.training.evaluate_models` to generate plots.")

    st.divider()

    # ── Dynamic model ──
    st.subheader("Dynamic Gesture Model (BiLSTM) — J, Z + 10 Words")

    dynamic_df = pd.DataFrame({
        "Metric": ["Accuracy", "Weighted Precision", "Weighted Recall",
                   "Weighted F1", "Macro F1"],
        "Value": ["97.22%", "97.35%", "97.22%", "97.22%", "97.22%"],
    })
    st.dataframe(dynamic_df, hide_index=True, use_container_width=True)

    cm_dynamic = plots_dir / "confusion_matrix_dynamic.png"
    pc_dynamic  = plots_dir / "per_class_accuracy_dynamic.png"
    ea_dynamic  = plots_dir / "error_analysis_dynamic.png"

    if cm_dynamic.exists():
        c1, c2 = st.columns(2)
        with c1:
            st.image(str(cm_dynamic), caption="Confusion Matrix — Dynamic Model")
        with c2:
            st.image(str(pc_dynamic), caption="Per-Class Accuracy — Dynamic Model")
        st.image(str(ea_dynamic), caption="Error Analysis — Dynamic Model")

    st.divider()

    # ── Inference speed ──
    st.subheader("Inference Speed Benchmark")
    bench_df = pd.DataFrame({
        "Component": ["MediaPipe Hand Detection", "Static MLP Inference", "Dynamic BiLSTM Inference"],
        "Mean Latency (ms)": [26.50, 94.02, 97.11],
        "Std Dev (ms)": [8.28, 39.28, 16.26],
        "Throughput (FPS)": [37.73, 10.64, 10.30],
    })
    st.dataframe(bench_df, hide_index=True, use_container_width=True)

    st.divider()

    # ── Training history plots ──
    st.subheader("Training History")
    th_static  = plots_dir / "training_history_static.png"
    th_dynamic = plots_dir / "training_history_dynamic.png"
    if th_static.exists() or th_dynamic.exists():
        c1, c2 = st.columns(2)
        if th_static.exists():
            with c1:
                st.image(str(th_static), caption="Training History — Static MLP")
        if th_dynamic.exists():
            with c2:
                st.image(str(th_dynamic), caption="Training History — Dynamic BiLSTM")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Live Demo (image upload)
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Static Gesture Demo")
    st.write(
        "Upload a photo of your hand or take one with your webcam. "
        "The model will detect the hand and predict the ASL letter/digit."
    )

    @st.cache_resource
    def load_models():
        try:
            import mediapipe as mp
            from src.inference.predictor import Predictor
            predictor = Predictor()
            hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5,
            )
            return predictor, hands
        except Exception as e:
            return None, str(e)

    source = st.radio("Input source", ["Upload image", "Use webcam"], horizontal=True)

    img_array = None

    if source == "Upload image":
        uploaded = st.file_uploader("Upload a hand image", type=["jpg", "jpeg", "png"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            img_array = np.array(img)
    else:
        photo = st.camera_input("Take a photo")
        if photo:
            img = Image.open(photo).convert("RGB")
            img_array = np.array(img)

    if img_array is not None:
        import cv2

        predictor, hands = load_models()

        if hands is None or isinstance(hands, str):
            st.error(f"Could not load models: {hands}")
        else:
            col_img, col_result = st.columns([1, 1])

            with col_img:
                st.image(img_array, caption="Input image", use_container_width=True)

            with col_result:
                results = hands.process(img_array)

                if results.multi_hand_landmarks:
                    hand_lms = results.multi_hand_landmarks[0]
                    hand = [(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]
                    landmark_array = np.array(hand, dtype=np.float32).flatten()

                    prediction = predictor.predict_static(landmark_array)

                    if prediction:
                        label, confidence = prediction
                        st.success(f"### Predicted: **{label}**")
                        st.metric("Confidence", f"{confidence:.1%}")
                        st.progress(confidence)
                    else:
                        st.warning(
                            "Hand detected but confidence below threshold (40%). "
                            "Try a clearer, well-lit photo."
                        )

                    # Show top 5 predictions
                    from src.preprocessing.feature_extractor import FeatureExtractor
                    fe = FeatureExtractor()
                    features = fe.extract(landmark_array).reshape(1, -1)
                    probs = predictor.static_model.predict(features, verbose=0)[0]
                    top5_idx = np.argsort(probs)[::-1][:5]
                    import pandas as pd
                    top5_df = pd.DataFrame({
                        "Gesture": [predictor.static_label_encoder.get(i, str(i)) for i in top5_idx],
                        "Confidence": [f"{probs[i]:.1%}" for i in top5_idx],
                    })
                    st.write("**Top 5 predictions:**")
                    st.dataframe(top5_df, hide_index=True, use_container_width=True)

                else:
                    st.error(
                        "No hand detected in the image. "
                        "Make sure your hand is clearly visible and well-lit."
                    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — About
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("About This System")

    st.markdown("""
    ## ASL Recognition System

    This prototype recognises **American Sign Language (ASL)** gestures in real time
    using a two-model pipeline:

    | Component | Model | Classes | Accuracy |
    |---|---|---|---|
    | Static gestures | MLP (3 hidden layers) | A–Y (no J/Z), 0–9 | 96.8% |
    | Dynamic gestures | BiLSTM | J, Z, hello, thanks, please, sorry, yes, no, help, stop, more, finish | 97.2% |

    ### Pipeline
    1. **MediaPipe Hands** — detects 21 hand landmarks per frame (37.7 FPS)
    2. **Feature extraction** — wrist-relative normalisation → 63-dim vector
    3. **Static MLP** — classifies single-frame poses
    4. **BiLSTM** — classifies 30-frame motion sequences

    ### Dataset
    - Static: **100 samples/class** for letters, 10 samples/class for digits
    - Dynamic: **100 sequences/class** × 30 frames each

    ### Tech Stack
    - Python 3.11 · TensorFlow 2.21 · MediaPipe 0.10.14 · OpenCV · Streamlit
    """)
