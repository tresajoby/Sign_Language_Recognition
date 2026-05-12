import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import InferenceConfig, DataCollectionConfig, EvaluationConfig
from src.data_collection.hand_detector import HandDetector
from src.preprocessing.feature_extractor import FeatureExtractor
from src.inference.predictor import Predictor


MOTION_THRESHOLD    = 0.008  # normalised units; lower = more sensitive
MOTION_HISTORY      = 6      # frames to average wrist displacement over
# Fingertip + wrist landmark indices (MediaPipe): thumb, index, middle, ring, pinky tips + wrist
KEY_LANDMARK_IDX    = [0, 4, 8, 12, 16, 20]
STILL_FRAMES_RESET  = 20     # consecutive still frames before switching to static
DETECT_EVERY        = 2      # run MediaPipe every N frames, cache landmarks between
STATIC_INFER_EVERY  = 2      # run static MLP every N frames
DYNAMIC_INFER_EVERY = 3      # run dynamic BiLSTM every N frames


class RealtimeRecognizer:

    def __init__(self):
        self.detector = HandDetector()
        self.extractor = FeatureExtractor()
        self.predictor = Predictor()
        self.buffer = deque(maxlen=DataCollectionConfig.DYNAMIC_SEQUENCE_LENGTH)
        self._wrist_history = deque(maxlen=MOTION_HISTORY)
        self._still_count = 0
        self._frame_count = 0
        self._cached_landmarks = None  # reused between detection frames

        if not self.predictor.static_ready:
            print("Static model not loaded")
        if not self.predictor.dynamic_ready:
            print("Dynamic model not loaded")

    def _is_moving(self, key_points):
        """Return True when average displacement of fingertips + wrist exceeds threshold.

        Tracks 6 key landmarks so finger-only motion (e.g. 'no', 'yes') is caught
        even when the wrist stays mostly still.
        """
        self._wrist_history.append(key_points)
        if len(self._wrist_history) < 2:
            return False
        frames = list(self._wrist_history)
        per_frame_disp = [
            float(np.mean([
                np.hypot(frames[i][k][0] - frames[i-1][k][0],
                         frames[i][k][1] - frames[i-1][k][1])
                for k in range(len(KEY_LANDMARK_IDX))
            ]))
            for i in range(1, len(frames))
        ]
        return float(np.mean(per_frame_disp)) > MOTION_THRESHOLD

    def run(self):
        cap = cv2.VideoCapture(DataCollectionConfig.CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DataCollectionConfig.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DataCollectionConfig.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, DataCollectionConfig.FPS)

        prev_time = time.time()
        static_result  = None
        dynamic_result = None
        last_mode      = "STATIC"

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self._frame_count += 1

            # Run MediaPipe every DETECT_EVERY frames; reuse cached landmarks in between
            if self._frame_count % DETECT_EVERY == 0:
                annotated_frame, landmarks = self.detector.detect(frame)
                self._cached_landmarks = landmarks
            else:
                annotated_frame = frame
                landmarks = self._cached_landmarks

            if landmarks:
                hand = landmarks[0]
                landmark_array = self.detector.get_landmark_array(hand)
                features = self.extractor.extract(landmark_array)

                if self._frame_count % STATIC_INFER_EVERY == 0:
                    static_result = self.predictor.predict_static(landmark_array)

                self.buffer.append(features)

                key_points  = [(hand[i][0], hand[i][1]) for i in KEY_LANDMARK_IDX]
                hand_moving = self._is_moving(key_points)

                if hand_moving:
                    self._still_count = 0
                    last_mode = "DYNAMIC"
                    if (len(self.buffer) == DataCollectionConfig.DYNAMIC_SEQUENCE_LENGTH
                            and self._frame_count % DYNAMIC_INFER_EVERY == 0):
                        sequence = np.array(self.buffer)
                        dynamic_result = self.predictor.predict_dynamic(sequence)
                else:
                    self._still_count += 1
                    if self._still_count >= STILL_FRAMES_RESET:
                        last_mode = "STATIC"
                        dynamic_result = None
                        self.buffer.clear()
                        self._still_count = 0

                if InferenceConfig.DISPLAY_BBOX:
                    bbox = self._get_bbox(frame, hand)
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2),
                                      InferenceConfig.BBOX_COLOR, 2)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            self._draw_info_panel(annotated_frame, fps, last_mode,
                                  static_result, dynamic_result)

            cv2.imshow("ASL Recognition", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.buffer.clear()
                dynamic_result = None
                print("Dynamic buffer reset.")
            elif key == ord('s'):
                save_dir = Path(EvaluationConfig.PLOTS_DIR)
                save_dir.mkdir(parents=True, exist_ok=True)
                filename = save_dir / f"screenshot_{int(time.time())}.png"
                cv2.imwrite(str(filename), annotated_frame)
                print(f"Screenshot saved: {filename}")

        cap.release()
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_info_panel(self, frame, fps, mode, static_result, dynamic_result):
        panel_x, panel_y = 8, 8
        panel_w, panel_h = 260, 175

        # Blend only the panel ROI — cheap because it's a small region, not the full frame
        roi = frame[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w]
        dark = np.zeros_like(roi)
        dark[:] = (30, 30, 30)
        cv2.addWeighted(dark, 0.55, roi, 0.45, 0, roi)

        if InferenceConfig.DISPLAY_FPS:
            cv2.putText(frame, f"FPS: {fps:.1f}", (panel_x + 8, panel_y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        mode_color = (0, 210, 110) if mode == "STATIC" else (60, 140, 255)
        cv2.putText(frame, f"Mode: {mode}", (panel_x + 8, panel_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, mode_color, 2)

        result     = static_result if mode == "STATIC" else dynamic_result
        pred_color = (0, 230, 120) if mode == "STATIC" else (80, 160, 255)

        if result:
            label, confidence = result
            cv2.putText(frame, label, (panel_x + 8, panel_y + 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, pred_color, 3)
            cv2.putText(frame, f"{confidence:.0%}", (panel_x + 8, panel_y + 138),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
            bar_x     = panel_x + 8
            bar_y     = panel_y + 148
            bar_w     = panel_w - 20
            bar_h     = 10
            filled_w  = int(bar_w * confidence)
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + filled_w, bar_y + bar_h), pred_color, -1)
        else:
            cv2.putText(frame, "--", (panel_x + 8, panel_y + 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (100, 100, 100), 3)

        buf_count = len(self.buffer)
        buf_max   = DataCollectionConfig.DYNAMIC_SEQUENCE_LENGTH
        cv2.putText(frame, f"Buf {buf_count}/{buf_max}",
                    (DataCollectionConfig.FRAME_WIDTH - 110, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    def _get_bbox(self, frame, landmarks_raw):
        h, w = frame.shape[:2]
        xs = [lm[0] * w for lm in landmarks_raw]
        ys = [lm[1] * h for lm in landmarks_raw]
        if not xs or not ys:
            return None
        pad = 20
        x1 = max(0, int(min(xs)) - pad)
        y1 = max(0, int(min(ys)) - pad)
        x2 = min(w, int(max(xs)) + pad)
        y2 = min(h, int(max(ys)) + pad)
        return x1, y1, x2, y2
