"""
Dynamic Gesture Data Collection Module

Records sequences of landmark frames for motion-based ASL signs (e.g. 'hello',
'thanks').  Each recorded sequence is DYNAMIC_SEQUENCE_LENGTH frames long.

Saved file layout:
    data/raw/dynamic/{gesture_class}.npy   shape: (num_sequences, seq_len, 63)

Usage:
    collector = DynamicCollector()
    collector.collect('hello')
    collector.collect_all()
"""

import cv2
import numpy as np
from pathlib import Path

from src.utils.config import DataCollectionConfig, RAW_DATA_DIR
from src.data_collection.hand_detector import HandDetector
from src.preprocessing.feature_extractor import FeatureExtractor


class DynamicCollector:
    """Collects dynamic (multi-frame sequence) ASL gesture samples from a webcam."""

    def __init__(self):
        self.detector = HandDetector()
        self.extractor = FeatureExtractor()

        self.save_dir = RAW_DATA_DIR / "dynamic"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.seq_length = DataCollectionConfig.DYNAMIC_SEQUENCE_LENGTH
        self.num_sequences = DataCollectionConfig.DYNAMIC_SEQUENCES_PER_CLASS
        self.camera_id = DataCollectionConfig.CAMERA_ID

    def collect(self, gesture_class, num_sequences=None):
        """
        Collect landmark sequences for one dynamic gesture class.

        Press 's' to begin recording a single sequence.  Recording stops
        automatically after DYNAMIC_SEQUENCE_LENGTH frames have been captured.
        Repeat until the target number of sequences is reached or press 'q'
        to quit early.

        Args:
            gesture_class: string label, e.g. 'hello'.
            num_sequences: override for how many sequences to collect.
                           Defaults to DataCollectionConfig.DYNAMIC_SEQUENCES_PER_CLASS.
        """
        if num_sequences is None:
            num_sequences = self.num_sequences

        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DataCollectionConfig.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DataCollectionConfig.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, DataCollectionConfig.FPS)

        sequences = []       # completed sequences
        current_seq = []     # frames for the sequence being recorded
        recording = False

        print(f"\n[DynamicCollector] Class: '{gesture_class}' | Target: {num_sequences} sequences")
        print(f"  Each sequence = {self.seq_length} frames")
        print("  Press 's' to start a sequence | Press 'q' to quit early")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[DynamicCollector] Failed to read from camera.")
                break

            annotated, landmarks_list = self.detector.detect(frame)

            if recording:
                # Capture frame into the current sequence if a hand is visible
                if landmarks_list is not None:
                    raw_array = self.detector.get_landmark_array(landmarks_list[0])
                    features = self.extractor.extract(raw_array)
                    current_seq.append(features)
                else:
                    # No hand detected: pad with zeros to keep fixed length
                    current_seq.append(np.zeros(63, dtype=np.float32))

                # Check if the sequence is complete
                if len(current_seq) >= self.seq_length:
                    sequences.append(np.array(current_seq, dtype=np.float32))
                    current_seq = []
                    recording = False
                    print(f"  Sequence {len(sequences)}/{num_sequences} recorded.")

            # Overlay status information
            status = f"RECORDING {len(current_seq)}/{self.seq_length}" if recording else "READY - press 's'"
            cv2.putText(annotated, f"Class: {gesture_class}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(annotated, f"Sequences: {len(sequences)}/{num_sequences}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(annotated, status, (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if recording else (0, 255, 0), 2)

            cv2.imshow("Dynamic Collector", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and not recording:
                current_seq = []
                recording = True
            elif key == ord('q'):
                print("[DynamicCollector] Quit early.")
                break

            if len(sequences) >= num_sequences:
                print(f"[DynamicCollector] Collected {num_sequences} sequences for '{gesture_class}'.")
                break

        cap.release()
        cv2.destroyAllWindows()

        if sequences:
            # Stack into shape (num_sequences, seq_length, 63)
            data = np.stack(sequences, axis=0).astype(np.float32)
            save_path = self.save_dir / f"{gesture_class}.npy"
            np.save(save_path, data)
            print(f"[DynamicCollector] Saved {len(sequences)} sequences -> {save_path}")
        else:
            print(f"[DynamicCollector] No sequences collected for '{gesture_class}'. Nothing saved.")

    def collect_all(self):
        """Iterate through every class in DYNAMIC_CLASSES and collect sequences."""
        for gesture_class in DataCollectionConfig.DYNAMIC_CLASSES:
            self.collect(gesture_class)

        self.detector.release()
        print("\n[DynamicCollector] All dynamic classes collected.")
