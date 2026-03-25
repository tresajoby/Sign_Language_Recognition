"""
Static Gesture Data Collection Module

Captures webcam frames, detects the hand with MediaPipe, extracts normalized
landmark features, and saves a .npy file per gesture class to data/raw/static/.

Usage:
    collector = StaticCollector()
    collector.collect('A')          # collect for one class
    collector.collect_all()         # collect for every STATIC_CLASS
"""

import cv2
import numpy as np
from pathlib import Path

from src.utils.config import DataCollectionConfig, RAW_DATA_DIR
from src.data_collection.hand_detector import HandDetector
from src.preprocessing.feature_extractor import FeatureExtractor


class StaticCollector:
    """Collects static (single-frame) ASL gesture samples from a webcam."""

    def __init__(self):
        self.detector = HandDetector()
        self.extractor = FeatureExtractor()

        # Ensure the output directory exists
        self.save_dir = RAW_DATA_DIR / "static"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.num_samples = DataCollectionConfig.STATIC_SAMPLES_PER_CLASS
        self.camera_id = DataCollectionConfig.CAMERA_ID

    def collect(self, gesture_class, num_samples=None):
        """
        Collect landmark samples for a single gesture class.

        Opens the webcam and displays a live feed with landmarks overlaid.
        Press 's' to toggle sample collection on/off.
        Press 'q' to quit early.

        Collected samples are saved to:
            data/raw/static/{gesture_class}.npy   shape: (num_samples, 63)

        Args:
            gesture_class: string label, e.g. 'A' or 'hello'.
            num_samples:   override for how many samples to collect.
                           Defaults to DataCollectionConfig.STATIC_SAMPLES_PER_CLASS.
        """
        if num_samples is None:
            num_samples = self.num_samples

        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DataCollectionConfig.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DataCollectionConfig.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, DataCollectionConfig.FPS)

        samples = []
        collecting = False

        print(f"\n[StaticCollector] Class: '{gesture_class}' | Target: {num_samples} samples")
        print("  Press 's' to start/stop collecting | Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[StaticCollector] Failed to read from camera.")
                break

            annotated, landmarks_list = self.detector.detect(frame)

            # If collecting and a hand is visible, grab the first hand's features
            if collecting and landmarks_list is not None:
                raw_array = self.detector.get_landmark_array(landmarks_list[0])
                features = self.extractor.extract(raw_array)
                samples.append(features)

            # Build status text for the overlay
            status = "COLLECTING" if collecting else "PAUSED - press 's' to start"
            count_text = f"{len(samples)}/{num_samples}"

            cv2.putText(annotated, f"Class: {gesture_class}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(annotated, f"Samples: {count_text}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(annotated, status, (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if not collecting else (0, 255, 0), 2)

            cv2.imshow("Static Collector", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                collecting = not collecting
            elif key == ord('q'):
                print("[StaticCollector] Quit early.")
                break

            # Auto-stop when target is reached
            if len(samples) >= num_samples:
                print(f"[StaticCollector] Collected {num_samples} samples for '{gesture_class}'.")
                break

        cap.release()
        cv2.destroyAllWindows()

        if samples:
            data = np.array(samples, dtype=np.float32)  # (N, 63)
            save_path = self.save_dir / f"{gesture_class}.npy"
            np.save(save_path, data)
            print(f"[StaticCollector] Saved {len(samples)} samples -> {save_path}")
        else:
            print(f"[StaticCollector] No samples collected for '{gesture_class}'. Nothing saved.")

    def collect_all(self):
        """Iterate through every class in STATIC_CLASSES and collect samples."""
        for gesture_class in DataCollectionConfig.STATIC_CLASSES:
            self.collect(gesture_class)

        self.detector.release()
        print("\n[StaticCollector] All static classes collected.")
