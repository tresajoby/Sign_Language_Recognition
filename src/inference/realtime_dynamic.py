"""
Real-Time Dynamic Gesture Recognition

This module provides real-time recognition of dynamic (motion-based) ASL gestures.
It buffers frames into sequences and uses the trained BiLSTM for classification.

Usage:
    python src/inference/realtime_dynamic.py

Controls:
    Q - Quit
    SPACE - Pause/Resume
    R - Start recording sequence
"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path
from collections import deque
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.preprocessing.hand_detector import HandDetector
from src.preprocessing.feature_extractor import FeatureExtractor
from src.utils.config import (
    InferenceConfig,
    DataCollectionConfig,
    DynamicModelConfig,
    MODELS_DIR,
    LABELS_DIR
)


class RealtimeDynamicRecognizer:
    """
    Real-time dynamic gesture recognition system.

    Dynamic gestures require temporal sequences, so this system:
    1. Buffers frames into sequences of length N (default: 30)
    2. Extracts features from each frame
    3. Feeds the complete sequence to BiLSTM
    4. Displays the prediction

    Modes:
    - Continuous: Rolling window, predict every frame
    - Triggered: User presses 'R' to start recording, auto-predict after N frames
    """

    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = InferenceConfig.CONFIDENCE_THRESHOLD,
        sequence_length: int = DynamicModelConfig.SEQUENCE_LENGTH,
        mode: str = 'triggered'  # 'continuous' or 'triggered'
    ):
        """
        Initialize the recognizer.

        Args:
            model_path: Path to trained model
            confidence_threshold: Minimum confidence for predictions
            sequence_length: Number of frames in sequence
            mode: 'continuous' or 'triggered'
        """
        # Initialize components
        self.detector = HandDetector(static_image_mode=False, max_num_hands=1)
        self.extractor = FeatureExtractor()

        # Load model
        if model_path is None:
            model_path = MODELS_DIR / "dynamic_model_final.h5"

        self.model = self._load_model(model_path)

        # Load label mapping
        self.label_mapping = self._load_label_mapping()

        # Configuration
        self.confidence_threshold = confidence_threshold
        self.sequence_length = sequence_length
        self.mode = mode

        # Sequence buffer
        self.frame_buffer = deque(maxlen=sequence_length)

        # Recording state (for triggered mode)
        self.recording = False
        self.frames_recorded = 0

        # Prediction state
        self.current_prediction = None
        self.current_confidence = 0.0
        self.prediction_cooldown = 0

        # FPS tracking
        self.frame_times = deque(maxlen=30)

    def _load_model(self, model_path: Path):
        """Load trained model."""
        from tensorflow import keras

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Please train the model first:\n"
                "  python src/training/train_dynamic.py"
            )

        print(f"[INFO] Loading model from {model_path}")
        model = keras.models.load_model(str(model_path))
        print(f"[INFO] Model loaded successfully")

        return model

    def _load_label_mapping(self) -> dict:
        """Load label mapping."""
        mapping_path = LABELS_DIR / "dynamic_label_mapping.json"

        if mapping_path.exists():
            with open(mapping_path, 'r') as f:
                mapping = {int(k): v for k, v in json.load(f).items()}
        else:
            mapping = {i: gesture for i, gesture in
                      enumerate(DataCollectionConfig.DYNAMIC_CLASSES)}

        return mapping

    def add_frame(self, landmarks: np.ndarray):
        """
        Add frame to buffer.

        Args:
            landmarks: Hand landmarks (21, 3)
        """
        # Extract features
        features = self.extractor.extract_features(landmarks)

        # Add to buffer
        self.frame_buffer.append(features)

        if self.recording:
            self.frames_recorded += 1

    def predict_sequence(self) -> tuple:
        """
        Predict gesture from buffered sequence.

        Returns:
            Tuple of (gesture_name, confidence, success)
        """
        if len(self.frame_buffer) < self.sequence_length:
            return None, 0.0, False

        # Convert buffer to array
        sequence = np.array(list(self.frame_buffer))

        # Reshape for model input (1, seq_length, feature_dim)
        sequence = sequence.reshape(1, self.sequence_length, -1)

        # Get prediction
        probabilities = self.model.predict(sequence, verbose=0)[0]

        # Get top prediction
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]

        # Get gesture name
        gesture_name = self.label_mapping.get(predicted_class, "Unknown")

        return gesture_name, confidence, True

    def calculate_fps(self) -> float:
        """Calculate FPS."""
        if len(self.frame_times) < 2:
            return 0.0

        time_diff = self.frame_times[-1] - self.frame_times[0]
        if time_diff == 0:
            return 0.0

        return (len(self.frame_times) - 1) / time_diff

    def draw_ui(
        self,
        frame: np.ndarray,
        gesture: str = None,
        confidence: float = 0.0,
        fps: float = 0.0,
        hand_detected: bool = False,
        buffer_fill: float = 0.0
    ) -> np.ndarray:
        """Draw UI on frame."""
        h, w = frame.shape[:2]

        # Create overlay
        overlay = frame.copy()

        # Top panel
        cv2.rectangle(overlay, (0, 0), (w, 60), (50, 50, 50), -1)

        # Bottom panel
        cv2.rectangle(overlay, (0, h - 40), (w, h), (50, 50, 50), -1)

        # Blend
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Title
        cv2.putText(
            frame,
            "ASL Dynamic Gesture Recognition",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        # FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (w - 150, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Recording indicator (triggered mode)
        if self.mode == 'triggered':
            if self.recording:
                # Recording indicator
                cv2.circle(frame, (w - 200, 35), 12, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"Recording {self.frames_recorded}/{self.sequence_length}",
                    (w - 400, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

        # Buffer fill indicator (continuous mode)
        if self.mode == 'continuous':
            bar_width = 200
            bar_height = 15
            bar_x = w - 220
            bar_y = 15

            # Background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)

            # Fill
            fill_width = int(bar_width * buffer_fill)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 255, 0), -1)

            # Border
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)

        # Prediction box
        if gesture and confidence >= self.confidence_threshold:
            box_height = 120
            box_width = 350
            box_x = (w - box_width) // 2
            box_y = h // 2 - 150

            # Draw box
            cv2.rectangle(
                frame,
                (box_x, box_y),
                (box_x + box_width, box_y + box_height),
                (0, 255, 0),
                3
            )

            # Gesture name
            cv2.putText(
                frame,
                gesture.upper(),
                (box_x + 20, box_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.8,
                (0, 255, 0),
                3
            )

            # Confidence
            cv2.putText(
                frame,
                f"{confidence*100:.1f}%",
                (box_x + 20, box_y + 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        # Controls
        if self.mode == 'triggered':
            controls = "Q: Quit  |  SPACE: Pause  |  R: Record Gesture"
        else:
            controls = "Q: Quit  |  SPACE: Pause  |  C: Clear Buffer"

        cv2.putText(
            frame,
            controls,
            (20, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        return frame

    def run(self, camera_id: int = 0):
        """Run the recognition system."""
        print("\n" + "="*70)
        print("REAL-TIME DYNAMIC GESTURE RECOGNITION")
        print("="*70)
        print(f"\nMode: {self.mode.upper()}")

        if self.mode == 'triggered':
            print("\nControls:")
            print("  Q     - Quit")
            print("  SPACE - Pause/Resume")
            print("  R     - Start recording gesture")
            print("\nHow to use:")
            print("  1. Press 'R' to start recording")
            print("  2. Perform the gesture")
            print("  3. Prediction shows automatically after 30 frames")
        else:
            print("\nControls:")
            print("  Q     - Quit")
            print("  SPACE - Pause/Resume")
            print("  C     - Clear buffer")
            print("\nHow to use:")
            print("  Continuous prediction on rolling 30-frame window")

        print("\nStarting webcam...")

        # Open webcam
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {camera_id}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DataCollectionConfig.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DataCollectionConfig.FRAME_HEIGHT)

        print("[INFO] Webcam initialized")
        print("[INFO] Starting recognition...\n")

        paused = False

        try:
            while True:
                success, frame = cap.read()

                if not success:
                    print("[ERROR] Failed to read frame")
                    break

                # Track time
                current_time = time.time()
                self.frame_times.append(current_time)

                if not paused:
                    # Detect hand
                    hands_detected, results = self.detector.detect_hands(frame)

                    if hands_detected:
                        # Draw landmarks
                        frame = self.detector.draw_landmarks(frame, results)

                        # Get landmarks
                        landmarks = self.detector.get_landmarks(results)

                        if landmarks is not None:
                            # Add frame to buffer
                            if self.mode == 'continuous' or self.recording:
                                self.add_frame(landmarks)

                            # Check if sequence complete
                            if self.recording and self.frames_recorded >= self.sequence_length:
                                # Predict
                                gesture, confidence, success = self.predict_sequence()

                                if success:
                                    self.current_prediction = gesture
                                    self.current_confidence = confidence
                                    self.prediction_cooldown = 90  # Show for 3 seconds at 30fps

                                    print(f"[PREDICTION] {gesture} ({confidence*100:.1f}%)")

                                # Reset recording
                                self.recording = False
                                self.frames_recorded = 0
                                self.frame_buffer.clear()

                            # Continuous mode prediction
                            elif self.mode == 'continuous' and len(self.frame_buffer) == self.sequence_length:
                                if self.prediction_cooldown == 0:
                                    gesture, confidence, success = self.predict_sequence()
                                    if success and confidence >= self.confidence_threshold:
                                        self.current_prediction = gesture
                                        self.current_confidence = confidence
                                        self.prediction_cooldown = 30  # Cooldown to avoid rapid re-prediction

                    # Decrement cooldown
                    if self.prediction_cooldown > 0:
                        self.prediction_cooldown -= 1

                    # Clear prediction after cooldown
                    if self.prediction_cooldown == 0:
                        self.current_prediction = None
                        self.current_confidence = 0.0

                    # Calculate FPS
                    fps = self.calculate_fps()

                    # Buffer fill
                    buffer_fill = len(self.frame_buffer) / self.sequence_length

                    # Draw UI
                    frame = self.draw_ui(
                        frame,
                        gesture=self.current_prediction,
                        confidence=self.current_confidence,
                        fps=fps,
                        hand_detected=hands_detected,
                        buffer_fill=buffer_fill
                    )
                else:
                    cv2.putText(
                        frame,
                        "PAUSED",
                        (frame.shape[1]//2 - 80, frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.0,
                        (0, 255, 255),
                        3
                    )

                # Display
                cv2.imshow('ASL Dynamic Gesture Recognition', frame)

                # Handle keys
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == ord('Q'):
                    print("\n[INFO] Quitting...")
                    break

                elif key == ord(' '):
                    paused = not paused
                    print(f"[INFO] {'Paused' if paused else 'Resumed'}")

                elif key == ord('r') or key == ord('R'):
                    if self.mode == 'triggered' and not self.recording:
                        self.recording = True
                        self.frames_recorded = 0
                        self.frame_buffer.clear()
                        print("[INFO] Recording started...")

                elif key == ord('c') or key == ord('C'):
                    if self.mode == 'continuous':
                        self.frame_buffer.clear()
                        print("[INFO] Buffer cleared")

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.detector.close()

            print("\n" + "="*70)
            print("RECOGNITION SESSION COMPLETE")
            print("="*70)


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Real-time dynamic gesture recognition")
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained model'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera ID'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['continuous', 'triggered'],
        default='triggered',
        help='Recognition mode'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=InferenceConfig.CONFIDENCE_THRESHOLD,
        help='Confidence threshold'
    )

    args = parser.parse_args()

    # Create recognizer
    recognizer = RealtimeDynamicRecognizer(
        model_path=args.model,
        confidence_threshold=args.threshold,
        mode=args.mode
    )

    # Run
    recognizer.run(camera_id=args.camera)


if __name__ == "__main__":
    main()
