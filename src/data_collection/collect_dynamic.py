"""
Dynamic Gesture Data Collection Script

This script provides an interactive interface for collecting dynamic (motion-based) gesture data.
Dynamic gestures are temporal sequences of hand poses (e.g., "hello", "thanks", "please").

Usage:
    python src/data_collection/collect_dynamic.py

Features:
- Interactive webcam interface with countdown timer
- Sequence recording (30 frames per gesture)
- Real-time hand landmark visualization
- Progress tracking
- Automatic sequence padding/truncation

Academic Justification:
- Temporal sequences capture motion patterns crucial for dynamic gestures
- Fixed sequence length (30 frames) simplifies BiLSTM input
- Multiple sequences per class capture natural variation in speed and style
- Frame rate normalization ensures consistent temporal resolution

Dataset Structure:
    data/raw/dynamic/{gesture_name}/
        sequence_{timestamp}.npy - Landmark sequences (30, 21, 3)
    data/processed/
        dynamic_sequences.npy - Processed sequences (N, 30, 63)
        dynamic_labels.npy - Corresponding labels
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
from typing import List
import json
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.preprocessing.hand_detector import HandDetector
from src.preprocessing.feature_extractor import FeatureExtractor
from src.utils.config import DataCollectionConfig, RAW_DATA_DIR, PROCESSED_DATA_DIR, LABELS_DIR


class DynamicGestureCollector:
    """
    Interactive tool for collecting dynamic gesture sequences.

    This class handles:
    - Webcam capture
    - Sequence recording with countdown
    - Hand detection and landmark extraction
    - Sequence validation and saving
    - Progress tracking
    """

    def __init__(
        self,
        gestures: List[str] = None,
        sequences_per_gesture: int = DataCollectionConfig.DYNAMIC_SEQUENCES_PER_CLASS,
        sequence_length: int = DataCollectionConfig.DYNAMIC_SEQUENCE_LENGTH,
        camera_id: int = DataCollectionConfig.CAMERA_ID
    ):
        """
        Initialize the collector.

        Args:
            gestures: List of gesture names to collect
            sequences_per_gesture: Number of sequences to collect per gesture
            sequence_length: Number of frames per sequence
            camera_id: Webcam ID
        """
        self.gestures = gestures or DataCollectionConfig.DYNAMIC_CLASSES
        self.sequences_per_gesture = sequences_per_gesture
        self.sequence_length = sequence_length
        self.camera_id = camera_id

        # Initialize detector and extractor
        self.detector = HandDetector(
            static_image_mode=False,
            max_num_hands=1
        )
        self.extractor = FeatureExtractor()

        # Data storage
        self.data_dir = RAW_DATA_DIR / "dynamic"
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

    def record_sequence(self, gesture_name: str, cap) -> bool:
        """
        Record a single gesture sequence.

        Args:
            gesture_name: Name of the gesture being recorded
            cap: OpenCV VideoCapture object

        Returns:
            True if sequence successfully recorded, False otherwise

        Process:
        1. Countdown (3 seconds)
        2. Record sequence (capture landmarks for sequence_length frames)
        3. Validate sequence (check for hand presence)
        4. Save if valid
        """
        print(f"\n  Preparing to record sequence...")

        # Countdown phase
        countdown_seconds = 3

        for i in range(countdown_seconds, 0, -1):
            success, frame = cap.read()

            if not success:
                return False

            # Display countdown
            display_frame = frame.copy()

            # Large countdown text
            cv2.putText(
                display_frame,
                f"Get ready: {i}",
                (frame.shape[1]//2 - 150, frame.shape[0]//2),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (0, 255, 255),
                4
            )

            cv2.putText(
                display_frame,
                f"Gesture: {gesture_name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )

            cv2.imshow('Dynamic Gesture Collection', display_frame)
            cv2.waitKey(1000)  # Wait 1 second

        # Recording phase
        print(f"  Recording...")

        sequence = []
        start_time = time.time()

        for frame_idx in range(self.sequence_length):
            success, frame = cap.read()

            if not success:
                print("  Failed to read frame")
                return False

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
                    sequence.append(landmarks)
            else:
                # No hand detected - add None placeholder
                sequence.append(None)

            # Display recording indicator
            cv2.circle(
                display_frame,
                (30, 30),
                15,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                display_frame,
                "RECORDING",
                (55, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

            # Progress bar
            progress = (frame_idx + 1) / self.sequence_length
            bar_width = 400
            bar_x = (frame.shape[1] - bar_width) // 2
            bar_y = frame.shape[0] - 50

            # Background
            cv2.rectangle(
                display_frame,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + 30),
                (50, 50, 50),
                -1
            )

            # Progress
            progress_width = int(progress * bar_width)
            cv2.rectangle(
                display_frame,
                (bar_x, bar_y),
                (bar_x + progress_width, bar_y + 30),
                (0, 255, 0),
                -1
            )

            # Border
            cv2.rectangle(
                display_frame,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + 30),
                (255, 255, 255),
                2
            )

            cv2.imshow('Dynamic Gesture Collection', display_frame)
            cv2.waitKey(1)

        recording_time = time.time() - start_time

        print(f"  Recorded {len(sequence)} frames in {recording_time:.2f}s")

        # Validate sequence
        valid_frames = sum(1 for landmarks in sequence if landmarks is not None)
        validity_ratio = valid_frames / len(sequence)

        print(f"  Valid frames: {valid_frames}/{len(sequence)} ({validity_ratio*100:.1f}%)")

        if validity_ratio < 0.7:  # At least 70% of frames must have hand detected
            print("  Sequence invalid (too many missing frames)")
            return False

        # Interpolate missing frames
        sequence = self._interpolate_sequence(sequence)

        if sequence is None:
            print("  Failed to interpolate sequence")
            return False

        # Save sequence
        gesture_dir = self.data_dir / gesture_name
        gesture_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = gesture_dir / f"sequence_{timestamp}.npy"

        # Convert to array (sequence_length, 21, 3)
        sequence_array = np.array(sequence, dtype=np.float32)
        np.save(filename, sequence_array)

        print(f"  Sequence saved: {filename.name}")

        return True

    def _interpolate_sequence(self, sequence: List) -> List:
        """
        Interpolate missing frames in a sequence.

        Args:
            sequence: List of landmarks (some may be None)

        Returns:
            Interpolated sequence with no None values

        Method:
        - Linear interpolation between valid frames
        - If first/last frames are None, use nearest valid frame
        """
        # Find valid indices
        valid_indices = [i for i, landmarks in enumerate(sequence) if landmarks is not None]

        if len(valid_indices) == 0:
            return None

        # Convert to numpy array for valid frames
        valid_landmarks = np.array([sequence[i] for i in valid_indices])

        # Interpolate
        interpolated_sequence = []

        for i in range(len(sequence)):
            if sequence[i] is not None:
                # Valid frame - use as is
                interpolated_sequence.append(sequence[i])
            else:
                # Invalid frame - interpolate
                # Find nearest valid frames
                prev_valid = max([idx for idx in valid_indices if idx < i], default=None)
                next_valid = min([idx for idx in valid_indices if idx > i], default=None)

                if prev_valid is not None and next_valid is not None:
                    # Interpolate between prev and next
                    weight = (i - prev_valid) / (next_valid - prev_valid)
                    interpolated = (1 - weight) * sequence[prev_valid] + weight * sequence[next_valid]
                elif prev_valid is not None:
                    # Use previous valid frame
                    interpolated = sequence[prev_valid]
                elif next_valid is not None:
                    # Use next valid frame
                    interpolated = sequence[next_valid]
                else:
                    # No valid frames (shouldn't happen due to validity check)
                    return None

                interpolated_sequence.append(interpolated)

        return interpolated_sequence

    def collect_gesture(self, gesture_name: str):
        """
        Collect sequences for a specific gesture.

        Args:
            gesture_name: Name of the gesture to collect

        Controls:
            SPACE - Start recording sequence
            ENTER - Move to next gesture
            ESC - Quit collection
            R - Reset progress for current gesture
        """
        current_count = self.progress.get(gesture_name, 0)

        # Open webcam
        cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            print(f"Error: Cannot open camera {self.camera_id}")
            return False

        print(f"\n{'='*60}")
        print(f"Collecting gesture: {gesture_name}")
        print(f"Progress: {current_count}/{self.sequences_per_gesture}")
        print(f"{'='*60}")
        print("\nInstructions:")
        print("1. Position your hand in view")
        print("2. Press SPACE to start recording")
        print("3. Perform the gesture during the countdown")
        print("4. Sequence will auto-record for ~1 second")
        print("\nControls:")
        print("  SPACE - Start recording sequence")
        print("  ENTER - Next gesture (if enough sequences)")
        print("  R     - Reset progress for this gesture")
        print("  ESC   - Quit collection")

        while current_count < self.sequences_per_gesture:
            success, frame = cap.read()

            if not success:
                print("Failed to read frame")
                break

            # Detect hand
            hands_detected, results = self.detector.detect_hands(frame)

            # Create display frame
            display_frame = frame.copy()

            if hands_detected:
                display_frame = self.detector.draw_landmarks(display_frame, results)

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
                f"Progress: {current_count}/{self.sequences_per_gesture}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                "Press SPACE to start recording",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            # Show progress bar
            bar_width = 400
            bar_height = 30
            bar_x = 10
            bar_y = 90

            cv2.rectangle(
                display_frame,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (50, 50, 50),
                -1
            )

            progress_width = int((current_count / self.sequences_per_gesture) * bar_width)
            cv2.rectangle(
                display_frame,
                (bar_x, bar_y),
                (bar_x + progress_width, bar_y + bar_height),
                (0, 255, 0),
                -1
            )

            cv2.rectangle(
                display_frame,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (255, 255, 255),
                2
            )

            cv2.imshow('Dynamic Gesture Collection', display_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):  # SPACE - Record sequence
                success = self.record_sequence(gesture_name, cap)

                if success:
                    current_count += 1
                    self.progress[gesture_name] = current_count
                    self._save_progress()

                    print(f"  Sequence {current_count}/{self.sequences_per_gesture} complete!")
                else:
                    print("  Sequence failed - try again")

                time.sleep(1)  # Brief pause

            elif key == 13:  # ENTER
                if current_count >= self.sequences_per_gesture:
                    print(f"\nCompleted {gesture_name}!")
                    break
                else:
                    print(f"\n  Need {self.sequences_per_gesture - current_count} more sequences")

            elif key == ord('r') or key == ord('R'):
                response = input("\n  Reset progress for this gesture? (yes/no): ")
                if response.lower() == 'yes':
                    current_count = 0
                    self.progress[gesture_name] = 0
                    self._save_progress()
                    print("  Progress reset!")

            elif key == 27:  # ESC
                print("\nQuitting collection...")
                cap.release()
                cv2.destroyAllWindows()
                return False

        cap.release()
        cv2.destroyAllWindows()

        return True

    def collect_all(self):
        """Collect data for all dynamic gestures sequentially."""
        print("\n" + "="*60)
        print("DYNAMIC GESTURE DATA COLLECTION")
        print("="*60)
        print(f"\nTotal gestures: {len(self.gestures)}")
        print(f"Sequences per gesture: {self.sequences_per_gesture}")
        print(f"Frames per sequence: {self.sequence_length}")
        print(f"Total sequences: {len(self.gestures) * self.sequences_per_gesture}")

        completed = sum(1 for count in self.progress.values() if count >= self.sequences_per_gesture)
        print(f"\nProgress: {completed}/{len(self.gestures)} gestures completed")

        input("\nPress ENTER to start collection...")

        for gesture in self.gestures:
            if self.progress.get(gesture, 0) >= self.sequences_per_gesture:
                print(f"\nSkipping {gesture} (already completed)")
                continue

            success = self.collect_gesture(gesture)

            if not success:
                print("\nCollection interrupted!")
                break

        print("\n" + "="*60)
        print("COLLECTION COMPLETE")
        print("="*60)

        # Process collected data
        self.process_collected_data()

    def process_collected_data(self):
        """Process all collected sequences into feature arrays."""
        print("\n" + "="*60)
        print("PROCESSING COLLECTED DATA")
        print("="*60)

        all_sequences = []
        all_labels = []

        for gesture_idx, gesture in enumerate(self.gestures):
            gesture_dir = self.data_dir / gesture

            if not gesture_dir.exists():
                continue

            sequence_files = list(gesture_dir.glob("sequence_*.npy"))

            print(f"\nProcessing {gesture}: {len(sequence_files)} sequences")

            for seq_file in sequence_files:
                try:
                    # Load sequence (30, 21, 3)
                    sequence = np.load(seq_file)

                    # Extract features for each frame
                    feature_sequence = []
                    for frame_landmarks in sequence:
                        features = self.extractor.extract_features(frame_landmarks)
                        feature_sequence.append(features)

                    # Convert to array (30, 63)
                    feature_sequence = np.array(feature_sequence)

                    all_sequences.append(feature_sequence)
                    all_labels.append(gesture_idx)

                except Exception as e:
                    print(f"  Error processing {seq_file}: {e}")

        # Convert to arrays
        sequences_array = np.array(all_sequences, dtype=np.float32)
        labels_array = np.array(all_labels, dtype=np.int32)

        # Save
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        sequences_path = PROCESSED_DATA_DIR / "dynamic_sequences.npy"
        labels_path = PROCESSED_DATA_DIR / "dynamic_labels.npy"

        np.save(sequences_path, sequences_array)
        np.save(labels_path, labels_array)

        print(f"\nSaved sequences: {sequences_path}")
        print(f"  Shape: {sequences_array.shape}")

        print(f"\nSaved labels: {labels_path}")
        print(f"  Shape: {labels_array.shape}")

        # Save label mapping
        label_mapping = {idx: gesture for idx, gesture in enumerate(self.gestures)}
        mapping_path = LABELS_DIR / "dynamic_label_mapping.json"
        LABELS_DIR.mkdir(parents=True, exist_ok=True)

        with open(mapping_path, 'w') as f:
            json.dump(label_mapping, f, indent=2)

        print(f"\nSaved label mapping: {mapping_path}")
        print("\nPROCESSING COMPLETE!")


def main():
    """Main function to run dynamic gesture collection."""
    print("Dynamic Gesture Data Collection Tool")
    print("======================================\n")

    gestures = DataCollectionConfig.DYNAMIC_CLASSES
    sequences_per_gesture = DataCollectionConfig.DYNAMIC_SEQUENCES_PER_CLASS

    print(f"Will collect {len(gestures)} gestures: {', '.join(gestures)}")
    print(f"Sequences per gesture: {sequences_per_gesture}\n")

    collector = DynamicGestureCollector(
        gestures=gestures,
        sequences_per_gesture=sequences_per_gesture
    )

    collector.collect_all()


if __name__ == "__main__":
    main()
