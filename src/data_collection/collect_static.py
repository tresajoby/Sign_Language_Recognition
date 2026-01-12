"""
Static Gesture Data Collection Script

This script provides an interactive interface for collecting static gesture data.
Static gestures are single-frame hand poses (e.g., ASL letters A-Z, numbers 0-9).

Usage:
    python src/data_collection/collect_static.py

Features:
- Interactive webcam interface
- Real-time hand landmark visualization
- Progress tracking for each gesture class
- Automatic feature extraction and saving
- Data validation

Academic Justification:
- Large, diverse dataset critical for model generalization
- Multiple samples per class capture natural variation
- Systematic collection ensures balanced dataset
- Feature extraction during collection saves processing time

Dataset Structure:
    data/raw/static/{gesture_name}/
        landmarks_{timestamp}.npy - Raw landmarks
    data/processed/
        static_features.npy - Processed feature vectors
        static_labels.npy - Corresponding labels
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.preprocessing.hand_detector import HandDetector
from src.preprocessing.feature_extractor import FeatureExtractor
from src.utils.config import DataCollectionConfig, RAW_DATA_DIR, PROCESSED_DATA_DIR, LABELS_DIR


class StaticGestureCollector:
    """
    Interactive tool for collecting static gesture data.

    This class handles:
    - Webcam capture
    - Hand detection and landmark extraction
    - User interaction (key presses, prompts)
    - Data saving in organized structure
    - Progress tracking
    """

    def __init__(
        self,
        gestures: List[str] = None,
        samples_per_gesture: int = DataCollectionConfig.STATIC_SAMPLES_PER_CLASS,
        camera_id: int = DataCollectionConfig.CAMERA_ID
    ):
        """
        Initialize the collector.

        Args:
            gestures: List of gesture names to collect. If None, uses default from config.
            samples_per_gesture: Number of samples to collect per gesture
            camera_id: Webcam ID
        """
        self.gestures = gestures or DataCollectionConfig.STATIC_CLASSES
        self.samples_per_gesture = samples_per_gesture
        self.camera_id = camera_id

        # Initialize detector and extractor
        self.detector = HandDetector(
            static_image_mode=False,  # Use tracking for smooth capture
            max_num_hands=1
        )
        self.extractor = FeatureExtractor()

        # Data storage
        self.data_dir = RAW_DATA_DIR / "static"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Progress tracking
        self.progress = self._load_progress()

    def _load_progress(self) -> dict:
        """Load existing progress from file."""
        progress_file = self.data_dir / "collection_progress.json"

        if progress_file.exists():
            with open(progress_file, 'r') as f:
                return json.load(f)
        else:
            return {gesture: 0 for gesture in self.gestures}

    def _save_progress(self):
        """Save current progress to file."""
        progress_file = self.data_dir / "collection_progress.json"

        with open(progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def collect_gesture(self, gesture_name: str):
        """
        Collect samples for a specific gesture.

        Args:
            gesture_name: Name of the gesture to collect

        Controls:
            SPACE - Capture current frame
            ENTER - Move to next gesture
            ESC - Quit collection
            R - Reset progress for current gesture
        """
        # Create directory for this gesture
        gesture_dir = self.data_dir / gesture_name
        gesture_dir.mkdir(parents=True, exist_ok=True)

        # Get current progress
        current_count = self.progress.get(gesture_name, 0)

        # Open webcam
        cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            print(f"Error: Cannot open camera {self.camera_id}")
            return False

        print(f"\n{'='*60}")
        print(f"Collecting gesture: {gesture_name}")
        print(f"Progress: {current_count}/{self.samples_per_gesture}")
        print(f"{'='*60}")
        print("\nControls:")
        print("  SPACE - Capture frame")
        print("  ENTER - Next gesture (if enough samples)")
        print("  R     - Reset progress for this gesture")
        print("  ESC   - Quit collection")

        while current_count < self.samples_per_gesture:
            success, frame = cap.read()

            if not success:
                print("Failed to read frame")
                break

            # Detect hand
            hands_detected, results = self.detector.detect_hands(frame)

            # Create display frame
            display_frame = frame.copy()

            if hands_detected:
                # Draw landmarks
                display_frame = self.detector.draw_landmarks(display_frame, results)

                # Get landmarks
                landmarks = self.detector.get_landmarks(results)

                if landmarks is not None:
                    # Draw bounding box
                    landmarks_pixel = self.detector.get_landmarks_pixel_coords(
                        results,
                        frame.shape
                    )
                    x_min, y_min, x_max, y_max = self.detector.get_bounding_box(landmarks_pixel)

                    cv2.rectangle(
                        display_frame,
                        (x_min, y_min),
                        (x_max, y_max),
                        (0, 255, 0),
                        2
                    )

                    # Show "Ready to capture" indicator
                    cv2.putText(
                        display_frame,
                        "Hand detected - Press SPACE to capture",
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
            else:
                # No hand detected
                cv2.putText(
                    display_frame,
                    "No hand detected",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

            # Display gesture info
            cv2.putText(
                display_frame,
                f"Gesture: {gesture_name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                f"Progress: {current_count}/{self.samples_per_gesture}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # Show progress bar
            bar_width = 400
            bar_height = 30
            bar_x = 10
            bar_y = 90

            # Background
            cv2.rectangle(
                display_frame,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (50, 50, 50),
                -1
            )

            # Progress
            progress_width = int((current_count / self.samples_per_gesture) * bar_width)
            cv2.rectangle(
                display_frame,
                (bar_x, bar_y),
                (bar_x + progress_width, bar_y + bar_height),
                (0, 255, 0),
                -1
            )

            # Border
            cv2.rectangle(
                display_frame,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (255, 255, 255),
                2
            )

            # Display frame
            cv2.imshow('Static Gesture Collection', display_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):  # SPACE - Capture
                if hands_detected and landmarks is not None:
                    # Save landmarks
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = gesture_dir / f"landmarks_{timestamp}.npy"
                    np.save(filename, landmarks)

                    current_count += 1
                    self.progress[gesture_name] = current_count
                    self._save_progress()

                    print(f"  Captured sample {current_count}/{self.samples_per_gesture}")

                    # Visual feedback
                    cv2.imshow('Static Gesture Collection', display_frame)
                    cv2.waitKey(100)  # Flash for 100ms
                else:
                    print("  No hand detected - cannot capture")

            elif key == 13:  # ENTER - Next gesture
                if current_count >= self.samples_per_gesture:
                    print(f"\nCompleted {gesture_name}!")
                    break
                else:
                    print(f"\n  Need {self.samples_per_gesture - current_count} more samples")

            elif key == ord('r') or key == ord('R'):  # Reset
                response = input("\n  Reset progress for this gesture? (yes/no): ")
                if response.lower() == 'yes':
                    current_count = 0
                    self.progress[gesture_name] = 0
                    self._save_progress()
                    print("  Progress reset!")

            elif key == 27:  # ESC - Quit
                print("\nQuitting collection...")
                cap.release()
                cv2.destroyAllWindows()
                return False

        cap.release()
        cv2.destroyAllWindows()

        return True

    def collect_all(self):
        """
        Collect data for all gestures sequentially.

        Automatically moves through all gesture classes,
        tracking progress and allowing resume if interrupted.
        """
        print("\n" + "="*60)
        print("STATIC GESTURE DATA COLLECTION")
        print("="*60)
        print(f"\nTotal gestures: {len(self.gestures)}")
        print(f"Samples per gesture: {self.samples_per_gesture}")
        print(f"Total samples to collect: {len(self.gestures) * self.samples_per_gesture}")

        # Show progress summary
        completed = sum(1 for count in self.progress.values() if count >= self.samples_per_gesture)
        print(f"\nProgress: {completed}/{len(self.gestures)} gestures completed")

        input("\nPress ENTER to start collection...")

        # Collect each gesture
        for gesture in self.gestures:
            if self.progress.get(gesture, 0) >= self.samples_per_gesture:
                print(f"\nSkipping {gesture} (already completed)")
                continue

            success = self.collect_gesture(gesture)

            if not success:
                print("\nCollection interrupted!")
                break

        print("\n" + "="*60)
        print("COLLECTION COMPLETE")
        print("="*60)

        # Final summary
        completed = sum(1 for count in self.progress.values() if count >= self.samples_per_gesture)
        total_samples = sum(self.progress.values())

        print(f"\nCompleted gestures: {completed}/{len(self.gestures)}")
        print(f"Total samples collected: {total_samples}")

        # Process and save dataset
        self.process_collected_data()

    def process_collected_data(self):
        """
        Process all collected landmarks into feature vectors and save.

        This creates the final dataset files:
        - data/processed/static_features.npy
        - data/processed/static_labels.npy
        """
        print("\n" + "="*60)
        print("PROCESSING COLLECTED DATA")
        print("="*60)

        all_features = []
        all_labels = []

        for gesture_idx, gesture in enumerate(self.gestures):
            gesture_dir = self.data_dir / gesture

            if not gesture_dir.exists():
                continue

            # Load all landmark files for this gesture
            landmark_files = list(gesture_dir.glob("landmarks_*.npy"))

            print(f"\nProcessing {gesture}: {len(landmark_files)} samples")

            for landmark_file in landmark_files:
                try:
                    # Load landmarks
                    landmarks = np.load(landmark_file)

                    # Extract features
                    features = self.extractor.extract_features(landmarks)

                    all_features.append(features)
                    all_labels.append(gesture_idx)

                except Exception as e:
                    print(f"  Error processing {landmark_file}: {e}")

        # Convert to numpy arrays
        features_array = np.array(all_features, dtype=np.float32)
        labels_array = np.array(all_labels, dtype=np.int32)

        # Save processed data
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        features_path = PROCESSED_DATA_DIR / "static_features.npy"
        labels_path = PROCESSED_DATA_DIR / "static_labels.npy"

        np.save(features_path, features_array)
        np.save(labels_path, labels_array)

        print(f"\nSaved features: {features_path}")
        print(f"  Shape: {features_array.shape}")

        print(f"\nSaved labels: {labels_path}")
        print(f"  Shape: {labels_array.shape}")
        print(f"  Classes: {len(np.unique(labels_array))}")

        # Save label mapping
        label_mapping = {idx: gesture for idx, gesture in enumerate(self.gestures)}
        mapping_path = LABELS_DIR / "static_label_mapping.json"
        LABELS_DIR.mkdir(parents=True, exist_ok=True)

        with open(mapping_path, 'w') as f:
            json.dump(label_mapping, f, indent=2)

        print(f"\nSaved label mapping: {mapping_path}")

        print("\n" + "="*60)
        print("PROCESSING COMPLETE!")
        print("="*60)


def main():
    """
    Main function to run static gesture data collection.
    """
    print("Static Gesture Data Collection Tool")
    print("=====================================\n")

    # Ask user what to collect
    print("What would you like to collect?")
    print("1. All gestures (A-Z, 0-9)")
    print("2. Only letters (A-Z)")
    print("3. Only numbers (0-9)")
    print("4. Custom selection")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == '1':
        gestures = DataCollectionConfig.STATIC_CLASSES
    elif choice == '2':
        gestures = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    elif choice == '3':
        gestures = [str(i) for i in range(10)]
    elif choice == '4':
        custom = input("Enter gestures (comma-separated): ").strip()
        gestures = [g.strip().upper() for g in custom.split(',')]
    else:
        print("Invalid choice. Using all gestures.")
        gestures = DataCollectionConfig.STATIC_CLASSES

    # Ask for samples per gesture
    samples_input = input(f"\nSamples per gesture (default {DataCollectionConfig.STATIC_SAMPLES_PER_CLASS}): ").strip()

    if samples_input:
        samples_per_gesture = int(samples_input)
    else:
        samples_per_gesture = DataCollectionConfig.STATIC_SAMPLES_PER_CLASS

    print(f"\nWill collect {len(gestures)} gestures with {samples_per_gesture} samples each")
    print(f"Total: {len(gestures) * samples_per_gesture} samples\n")

    # Initialize collector
    collector = StaticGestureCollector(
        gestures=gestures,
        samples_per_gesture=samples_per_gesture
    )

    # Start collection
    collector.collect_all()


if __name__ == "__main__":
    main()
