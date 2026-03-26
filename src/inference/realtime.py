import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import InferenceConfig, DataCollectionConfig, EvaluationConfig
from src.data_collection.hand_detector import HandDetector
from src.preprocessing.feature_extractor import FeatureExtractor
from src.inference.predictor import Predictor


class RealtimeRecognizer:

    def __init__(self):
        self.detector = HandDetector()
        self.extractor = FeatureExtractor()
        self.predictor = Predictor()
        self.buffer = deque(maxlen=DataCollectionConfig.DYNAMIC_SEQUENCE_LENGTH)

        if not self.predictor.static_ready:
            print("Static model not loaded")
        if not self.predictor.dynamic_ready:
            print("Dynamic model not loaded")

    def run(self):
        cap = cv2.VideoCapture(DataCollectionConfig.CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DataCollectionConfig.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DataCollectionConfig.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, DataCollectionConfig.FPS)

        prev_time = time.time()
        static_result = None
        dynamic_result = None
        last_mode = "STATIC"

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, landmarks = self.detector.detect(frame)

            if landmarks:
                hand = landmarks[0]  # first (and only) hand — list of 21 (x,y,z) tuples
                landmark_array = self.detector.get_landmark_array(hand)
                features = self.extractor.extract(landmark_array)

                static_result = self.predictor.predict_static(landmark_array)

                self.buffer.append(features)
                if len(self.buffer) == DataCollectionConfig.DYNAMIC_SEQUENCE_LENGTH:
                    sequence = np.array(self.buffer)
                    dynamic_result = self.predictor.predict_dynamic(sequence)

                if InferenceConfig.DISPLAY_BBOX:
                    bbox = self._get_bbox(frame, hand)
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2),
                                      InferenceConfig.BBOX_COLOR, 2)

            if static_result and dynamic_result:
                last_mode = "STATIC" if static_result[1] >= dynamic_result[1] else "DYNAMIC"
            elif static_result:
                last_mode = "STATIC"
            elif dynamic_result:
                last_mode = "DYNAMIC"

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            if InferenceConfig.DISPLAY_FPS:
                self._draw_fps(annotated_frame, fps)

            if static_result:
                self._draw_prediction(annotated_frame, static_result[0], static_result[1],
                                      (10, 60), (0, 200, 100))
            else:
                cv2.putText(annotated_frame, "Static: --", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            InferenceConfig.TEXT_COLOR, 2)

            if dynamic_result:
                self._draw_prediction(annotated_frame, dynamic_result[0], dynamic_result[1],
                                      (10, 110), (200, 100, 0))
            else:
                cv2.putText(annotated_frame, "Dynamic: --", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            InferenceConfig.TEXT_COLOR, 2)

            mode_color = (0, 200, 100) if last_mode == "STATIC" else (200, 100, 0)
            cv2.putText(annotated_frame, f"Mode: {last_mode}", (10, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, mode_color, 2)

            buf_count = len(self.buffer)
            buf_max = DataCollectionConfig.DYNAMIC_SEQUENCE_LENGTH
            cv2.putText(annotated_frame, f"Buf: {buf_count}/{buf_max}",
                        (DataCollectionConfig.FRAME_WIDTH - 130, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        InferenceConfig.TEXT_COLOR, 1)

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

    def _draw_prediction(self, frame, label, confidence, position, color):
        x, y = position
        text = f"{label}: {confidence:.0%}"
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, InferenceConfig.TEXT_COLOR, 2)

        bar_x = x
        bar_y = y + 6
        bar_w = 150
        bar_h = 10
        filled_w = int(bar_w * confidence)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), color, -1)

    def _draw_fps(self, frame, fps):
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, InferenceConfig.TEXT_COLOR, 2)

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
