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
    col1.metric("Static Model Accuracy",  "97.37%", "+30.37% vs baseline")
    col2.metric("Dynamic Model Accuracy", "97.22%")
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
            "Value":   ["97.37%", "97.69%", "97.37%", "97.34%",
                        "97.71%", "97.27%", "97.29%"],
        }), hide_index=True, use_container_width=True)
    with n_col:
        st.info(
            "**Dataset:** 34 classes (A–Y excl. J/Z, digits 0–9).  \n"
            "Each class has **90–100 samples** (3,300 total).  \n"
            "Train / Val / Test split: **2,309 / 496 / 495** samples."
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

    d_col, d_info = st.columns(2)
    with d_col:
        st.dataframe(pd.DataFrame({
            "Metric": ["Accuracy", "Weighted Precision", "Weighted Recall",
                       "Weighted F1", "Macro F1"],
            "Value":  ["97.22%", "97.35%", "97.22%", "97.22%", "97.22%"],
        }), hide_index=True, use_container_width=True)
    with d_info:
        st.info(
            "**Dataset:** 12 classes (J, Z, hello, thanks, please, sorry, yes, no, help, stop, more, finish).  \n"
            "Each class has **100 sequences** (1,200 total).  \n"
            "Train / Val / Test split: **840 / 180 / 180** sequences."
        )

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
    st.subheader("Per-Class Results — Static Model (Letters A–Y)")
    letter_data = {
        "Class":     ["A","B","C","D","E","F","G","H","I","K","L","M",
                      "N","O","P","Q","R","S","T","U","V","W","X","Y"],
        "Precision": [1.00,1.00,1.00,1.00,1.00,0.88,1.00,1.00,1.00,1.00,
                      1.00,1.00,1.00,0.78,1.00,1.00,0.79,1.00,1.00,1.00,
                      1.00,0.94,1.00,1.00],
        "Recall":    [1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,
                      1.00,1.00,1.00,0.93,1.00,1.00,1.00,1.00,1.00,0.73,
                      1.00,1.00,1.00,1.00],
        "F1":        [1.00,1.00,1.00,1.00,1.00,0.94,1.00,1.00,1.00,1.00,
                      1.00,1.00,1.00,0.85,1.00,1.00,0.88,1.00,1.00,0.85,
                      1.00,0.97,1.00,1.00],
    }
    st.dataframe(pd.DataFrame(letter_data), hide_index=True, use_container_width=True)

    st.subheader("Per-Class Results — Static Model (Digits 0–9)")
    digit_data = {
        "Class":     ["0","1","2","3","4","5","6","7","8","9"],
        "Precision": [0.90,1.00,1.00,1.00,0.93,1.00,1.00,1.00,1.00,1.00],
        "Recall":    [0.69,1.00,0.93,1.00,1.00,1.00,0.93,1.00,1.00,0.86],
        "F1":        [0.78,1.00,0.96,1.00,0.97,1.00,0.96,1.00,1.00,0.92],
    }
    st.dataframe(pd.DataFrame(digit_data), hide_index=True, use_container_width=True)

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
            "True":      ["U", "0", "9", "O", "6"],
            "Predicted": ["R", "O", "F", "0", "W"],
            "Count":     [4,   4,   2,   1,   1],
            "Reason":    ["Similar finger shape",
                          "Ambiguous round shape",
                          "Similar loop shape",
                          "Ambiguous round shape",
                          "Similar finger spread"],
        }), hide_index=True, use_container_width=True)
    with c2:
        st.write("**Dynamic Model**")
        st.dataframe(pd.DataFrame({
            "True":      ["J",     "finish", "finish", "more"],
            "Predicted": ["finish","stop",   "more",   "stop"],
            "Count":     [2,       1,        1,        1],
            "Reason":    ["Curved motion overlap", "Similar trajectory",
                          "Similar wrist motion",  "Similar finger motion"],
        }), hide_index=True, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Live Demo
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Live Gesture Demo")
    st.markdown(
        "Point your webcam at your hand and press **START**. "
        "The system will detect static letters and digits, and automatically "
        "switch to dynamic mode when it detects hand motion."
    )

    MOTION_THRESHOLD   = 0.012
    MOTION_HISTORY     = 6
    STILL_FRAMES_RESET = 20

    try:
        import av
        import cv2 as _cv2
        from collections import deque as _deque
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

        try:
            from src.data_collection.hand_detector import HandDetector as _HD
            from src.preprocessing.feature_extractor import FeatureExtractor as _FE
            from src.inference.predictor import Predictor as _PR
            from src.utils.config import DataCollectionConfig as _DC
            _PIPELINE_OK = True
        except Exception as _e:
            _PIPELINE_OK = False
            st.error(f"Pipeline import failed: {_e}")

        if _PIPELINE_OK:
            class ASLVideoProcessor(VideoProcessorBase):
                def __init__(self):
                    self.detector     = _HD()
                    self.extractor    = _FE()
                    self.predictor    = _PR()
                    self.buffer       = _deque(maxlen=_DC.DYNAMIC_SEQUENCE_LENGTH)
                    self._wrist_hist  = _deque(maxlen=MOTION_HISTORY)
                    self._still_count = 0
                    self.static_result  = None
                    self.dynamic_result = None
                    self.last_mode      = "STATIC"

                def _is_moving(self, wrist_xy):
                    self._wrist_hist.append(wrist_xy)
                    if len(self._wrist_hist) < 2:
                        return False
                    pos = list(self._wrist_hist)
                    disps = [
                        np.hypot(pos[i][0] - pos[i-1][0], pos[i][1] - pos[i-1][1])
                        for i in range(1, len(pos))
                    ]
                    return float(np.mean(disps)) > MOTION_THRESHOLD

                def _bar(self, frame, x, y, conf, color):
                    bw = int(150 * conf)
                    _cv2.rectangle(frame, (x, y), (x + 150, y + 10), (80, 80, 80), -1)
                    _cv2.rectangle(frame, (x, y), (x + bw,  y + 10), color,        -1)

                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    annotated, landmarks = self.detector.detect(img)

                    if landmarks:
                        hand         = landmarks[0]
                        lm_array     = self.detector.get_landmark_array(hand)
                        features     = self.extractor.extract(lm_array)

                        self.static_result = self.predictor.predict_static(lm_array)
                        self.buffer.append(features)

                        wrist_xy   = (hand[0][0], hand[0][1])
                        moving     = self._is_moving(wrist_xy)

                        if moving:
                            self._still_count = 0
                            self.last_mode    = "DYNAMIC"
                            if len(self.buffer) == _DC.DYNAMIC_SEQUENCE_LENGTH:
                                seq = np.array(self.buffer)
                                self.dynamic_result = self.predictor.predict_dynamic(seq)
                        else:
                            self._still_count += 1
                            if self._still_count >= STILL_FRAMES_RESET:
                                self.last_mode      = "STATIC"
                                self.dynamic_result = None
                                self.buffer.clear()
                                self._still_count   = 0

                    # ── draw static row ──────────────────────────────────────
                    if self.last_mode == "STATIC" and self.static_result:
                        label, conf = self.static_result
                        _cv2.putText(annotated, f"Static: {label}  {conf:.0%}",
                                     (10, 55), _cv2.FONT_HERSHEY_SIMPLEX,
                                     0.85, (0, 200, 100), 2)
                        self._bar(annotated, 10, 62, conf, (0, 200, 100))
                    else:
                        _cv2.putText(annotated, "Static: --",
                                     (10, 55), _cv2.FONT_HERSHEY_SIMPLEX,
                                     0.75, (180, 180, 180), 2)

                    # ── draw dynamic row ─────────────────────────────────────
                    if self.dynamic_result:
                        label, conf = self.dynamic_result
                        _cv2.putText(annotated, f"Dynamic: {label}  {conf:.0%}",
                                     (10, 105), _cv2.FONT_HERSHEY_SIMPLEX,
                                     0.85, (200, 100, 0), 2)
                        self._bar(annotated, 10, 112, conf, (200, 100, 0))
                    else:
                        _cv2.putText(annotated, "Dynamic: --",
                                     (10, 105), _cv2.FONT_HERSHEY_SIMPLEX,
                                     0.75, (180, 180, 180), 2)

                    # ── mode indicator ───────────────────────────────────────
                    m_color = (0, 200, 100) if self.last_mode == "STATIC" else (200, 100, 0)
                    _cv2.putText(annotated, f"Mode: {self.last_mode}",
                                 (10, 150), _cv2.FONT_HERSHEY_SIMPLEX,
                                 0.65, m_color, 2)

                    # ── buffer counter ───────────────────────────────────────
                    buf = len(self.buffer)
                    _cv2.putText(annotated, f"Buf: {buf}/{_DC.DYNAMIC_SEQUENCE_LENGTH}",
                                 (10, 175), _cv2.FONT_HERSHEY_SIMPLEX,
                                 0.55, (180, 180, 180), 1)

                    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

            webrtc_streamer(
                key="asl-live",
                video_processor_factory=ASLVideoProcessor,
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                ),
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            st.caption(
                "**Static mode** — hold a letter (A–Y) or digit (0–9) still in front of the camera.  \n"
                "**Dynamic mode** — sign a word (hello, thanks, please, sorry, yes, no, "
                "help, stop, more, finish) or the letters J / Z."
            )

    except ImportError:
        st.warning(
            "The live demo requires **streamlit-webrtc**, **av**, **mediapipe**, and **tensorflow**.  \n"
            "Install them and relaunch:"
        )
        st.code(
            "pip install streamlit-webrtc av mediapipe==0.10.14 tensorflow==2.21.0",
            language="bash"
        )

    st.divider()
    st.subheader("How to run from the GitHub repository")
    st.code(
        "git clone https://github.com/tresajoby/Sign_Language_Recognition\n"
        "cd Sign_Language_Recognition\n"
        "pip install -r requirements.txt\n"
        "streamlit run app.py",
        language="bash"
    )
    st.info(
        "For the **OpenCV standalone** inference window (full-screen, higher FPS) run:  \n"
        "```\npython -m src.inference.run\n```  \n"
        "Controls: **Q** quit · **R** reset buffer · **S** screenshot"
    )

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
    | Static gestures | MLP (3 hidden layers) | A–Y (excl. J/Z), 0–9 | **97.37%** |
    | Dynamic gestures | BiLSTM | J, Z, hello, thanks, please, sorry, yes, no, help, stop, more, finish | **97.22%** |

    ### Pipeline
    1. **MediaPipe Hands** — detects 21 hand landmarks per frame (~37 FPS)
    2. **Wrist-relative normalisation** — 63-dimensional feature vector
    3. **Static MLP** — classifies single-frame hand poses
    4. **BiLSTM** — classifies 30-frame motion sequences

    ### Dataset
    | Split | Static | Dynamic |
    |---|---|---|
    | Train | 2,309 samples | 840 sequences |
    | Val | 496 samples | 180 sequences |
    | Test | 495 samples | 180 sequences |
    | **Total** | **3,300 samples** | **1,200 sequences** |

    Samples per class: **90–100** (static) · **100** (dynamic, all classes equal)

    ### Tech Stack
    Python 3.11 · TensorFlow 2.21 · Keras 3 · MediaPipe 0.10.14 · OpenCV 4.x · Streamlit
    """)
