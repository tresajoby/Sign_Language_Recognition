"""
Real-Time Static Gesture Recognition

This module provides real-time webcam-based recognition of static ASL gestures.
It integrates hand detection, feature extraction, and MLP classification into
a smooth, interactive interface.

Usage:
    python src/inference/realtime_static.py

Controls:
    Q - Quit
    SPACE - Pause/Resume
    S - Save screenshot
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
    MODELS_DIR,
    LABELS_DIR
)


class RealtimeStaticRecognizer:
    """
    Real-time static gesture recognition system.

    This class integrates:
    1. Hand detection (MediaPipe)
    2. Feature extraction (wrist-relative normalization)
    3. Gesture classification (trained MLP)
    4. Visualization and UI

    Attributes:
        detector: HandDetector instance
        extractor: FeatureExtractor instance
        model: Trained Keras model
        label_mapping: Dict mapping class indices to gesture names
        confidence_threshold: Minimum confidence for predictions
    """

    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = InferenceConfig.CONFIDENCE_THRESHOLD
    ):
        """
        Initialize the real-time recognizer.

        Args:
            model_path: Path to trained model (default: models/static_model_final.h5)
            confidence_threshold: Minimum confidence for displaying predictions
        """
        # Initialize hand detector
        self.detector = HandDetector(
            static_image_mode=False,  # Use tracking for video
            max_num_hands=1
        )

        # Initialize feature extractor
        self.extractor = FeatureExtractor()

        # Load trained model
        if model_path is None:
            model_path = MODELS_DIR / "static_model_final.h5"

        self.model = self._load_model(model_path)

        # Load label mapping
        self.label_mapping = self._load_label_mapping()

        # Configuration
        self.confidence_threshold = confidence_threshold

        # FPS tracking
        self.fps_history = deque(maxlen=30)
        self.frame_times = deque(maxlen=30)

        # Prediction smoothing
        self.prediction_history = deque(maxlen=5)

    def _load_model(self, model_path: Path):
        """Load trained Keras model."""
        from tensorflow import keras

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Please train the model first:\n"
                "  python src/training/train_static.py"
            )

        print(f"[INFO] Loading model from {model_path}")
        model = keras.models.load_model(str(model_path))
        print(f"[INFO] Model loaded successfully")

        return model

    def _load_label_mapping(self) -> dict:
        """Load label mapping from JSON."""
        mapping_path = LABELS_DIR / "static_label_mapping.json"

        if mapping_path.exists():
            with open(mapping_path, 'r') as f:
                mapping = {int(k): v for k, v in json.load(f).items()}
        else:
            # Default mapping
            mapping = {i: gesture for i, gesture in
                      enumerate(DataCollectionConfig.STATIC_CLASSES)}

        return mapping

    def predict(
        self,
        landmarks: np.ndarray,
        smooth: bool = True
    ) -> tuple:
        """
        Predict gesture from landmarks.

        Args:
            landmarks: Hand landmarks (21, 3)
            smooth: Whether to apply prediction smoothing

        Returns:
            Tuple of (gesture_name, confidence)
        """
        # Extract features
        features = self.extractor.extract_features(landmarks)

        # Reshape for model input
        features = features.reshape(1, -1)

        # Get prediction
        probabilities = self.model.predict(features, verbose=0)[0]

        # Get top prediction
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]

        # Prediction smoothing
        if smooth:
            self.prediction_history.append((predicted_class, confidence))

            # Use most common prediction in history
            if len(self.prediction_history) >= 3:
                classes = [p[0] for p in self.prediction_history]
                confidences = [p[1] for p in self.prediction_history]

                # Weighted voting based on confidence
                class_votes = {}
                for cls, conf in zip(classes, confidences):
                    if cls not in class_votes:
                        class_votes[cls] = 0
                    class_votes[cls] += conf

                predicted_class = max(class_votes, key=class_votes.get)
                confidence = np.mean([c for c, conf in zip(classes, confidences)
                                    if c == predicted_class])

        # Get gesture name
        gesture_name = self.label_mapping.get(predicted_class, "Unknown")

        return gesture_name, confidence

    def calculate_fps(self) -> float:
        """Calculate current FPS."""
        if len(self.frame_times) < 2:
            return 0.0

        time_diff = self.frame_times[-1] - self.frame_times[0]
        if time_diff == 0:
            return 0.0

        fps = (len(self.frame_times) - 1) / time_diff
        return fps

    def draw_ui(
        self,
        frame: np.ndarray,
        gesture: str = None,
        confidence: float = 0.0,
        fps: float = 0.0,
        hand_detected: bool = False
    ) -> np.ndarray:
        """
        Draw user interface on frame.

        Args:
            frame: Input frame
            gesture: Predicted gesture name
            confidence: Prediction confidence
            fps: Current FPS
            hand_detected: Whether hand is detected

        Returns:
            Frame with UI overlay
        """
        h, w = frame.shape[:2]

        # Create semi-transparent overlay for info panel
        overlay = frame.copy()

        # Top panel (title)
        cv2.rectangle(overlay, (0, 0), (w, 60), (50, 50, 50), -1)

        # Bottom panel (controls)
        cv2.rectangle(overlay, (0, h - 40), (w, h), (50, 50, 50), -1)

        # Blend overlay
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Title
        cv2.putText(
            frame,
            "ASL Static Gesture Recognition",
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

        # Hand detection status
        status_text = "Hand Detected" if hand_detected else "No Hand"
        status_color = (0, 255, 0) if hand_detected else (0, 0, 255)

        cv2.circle(frame, (w - 200, 35), 8, status_color, -1)
        cv2.putText(
            frame,
            status_text,
            (w - 180, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2
        )

        # Prediction box (center)
        if gesture and confidence >= self.confidence_threshold:
            box_height = 120
            box_width = 300
            box_x = (w - box_width) // 2
            box_y = h // 2 - 150

            # Draw prediction box
            cv2.rectangle(
                frame,
                (box_x, box_y),
                (box_x + box_width, box_y + box_height),
                (0, 255, 0),
                3
            )

            # Gesture name (large)
            cv2.putText(
                frame,
                gesture,
                (box_x + 20, box_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
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
        controls_text = "Q: Quit  |  SPACE: Pause  |  S: Screenshot"
        cv2.putText(
            frame,
            controls_text,
            (20, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        return frame

    def run(self, camera_id: int = 0):
        """
        Run the real-time recognition system.

        Args:
            camera_id: Webcam ID (default: 0)

        Controls:
            Q - Quit
            SPACE - Pause/Resume
            S - Save screenshot
        """
        print("\n" + "="*70)
        print("REAL-TIME STATIC GESTURE RECOGNITION")
        print("="*70)
        print("\nControls:")
        print("  Q     - Quit")
        print("  SPACE - Pause/Resume")
        print("  S     - Save screenshot")
        print("\nStarting webcam...")

        # Open webcam
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {camera_id}")
            return

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DataCollectionConfig.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DataCollectionConfig.FRAME_HEIGHT)

        print("[INFO] Webcam initialized")
        print("[INFO] Starting recognition...\n")

        paused = False
        screenshot_count = 0

        try:
            while True:
                # Read frame
                success, frame = cap.read()

                if not success:
                    print("[ERROR] Failed to read frame")
                    break

                # Track frame time
                current_time = time.time()
                self.frame_times.append(current_time)

                if not paused:
                    # Detect hand
                    hands_detected, results = self.detector.detect_hands(frame)

                    gesture = None
                    confidence = 0.0

                    if hands_detected:
                        # Draw landmarks
                        frame = self.detector.draw_landmarks(frame, results)

                        # Get landmarks
                        landmarks = self.detector.get_landmarks(results)

                        if landmarks is not None:
                            # Predict gesture
                            gesture, confidence = self.predict(landmarks, smooth=True)

                    # Calculate FPS
                    fps = self.calculate_fps()

                    # Draw UI
                    frame = self.draw_ui(
                        frame,
                        gesture=gesture,
                        confidence=confidence,
                        fps=fps,
                        hand_detected=hands_detected
                    )
                else:
                    # Paused - show message
                    cv2.putText(
                        frame,
                        "PAUSED",
                        (frame.shape[1]//2 - 80, frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.0,
                        (0, 255, 255),
                        3
                    )

                # Display frame
                cv2.imshow('ASL Static Gesture Recognition', frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == ord('Q'):
                    print("\n[INFO] Quitting...")
                    break

                elif key == ord(' '):  # Space
                    paused = not paused
                    print(f"[INFO] {'Paused' if paused else 'Resumed'}")

                elif key == ord('s') or key == ord('S'):
                    # Save screenshot
                    screenshot_path = f"screenshot_{screenshot_count}.png"
                    cv2.imwrite(screenshot_path, frame)
                    print(f"[INFO] Screenshot saved: {screenshot_path}")
                    screenshot_count += 1

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")

        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.detector.close()

            print("\n" + "="*70)
            print("RECOGNITION SESSION COMPLETE")
            print("="*70)


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Real-time static gesture recognition")
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained model (default: models/static_model_final.h5)'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera ID (default: 0)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=InferenceConfig.CONFIDENCE_THRESHOLD,
        help='Confidence threshold (default: 0.7)'
    )

    args = parser.parse_args()

    # Create recognizer
    recognizer = RealtimeStaticRecognizer(
        model_path=args.model,
        confidence_threshold=args.threshold
    )

    # Run
    recognizer.run(camera_id=args.camera)


if __name__ == "__main__":
    main()
