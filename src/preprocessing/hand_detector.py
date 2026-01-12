"""
Hand Detector Module using MediaPipe

This module provides a clean wrapper around MediaPipe Hands for hand detection
and landmark extraction. It abstracts away the complexity of MediaPipe's API
and provides a simple interface for the rest of the system.

Academic Justification:
- MediaPipe Hands uses a two-stage pipeline:
  1. Palm detection using a CNN-based detector
  2. Hand landmark regression using another CNN
- Achieves real-time performance (>30 FPS) on CPU
- Provides 21 3D landmarks per hand with high accuracy
- Published by Google Research with extensive validation

References:
- Zhang et al., "MediaPipe Hands: On-device Real-time Hand Tracking" (2020)
- https://arxiv.org/abs/2006.10214
"""

import cv2
import mediapipe as mp
from mediapipe import tasks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from typing import Optional, Tuple, List
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import MediaPipeConfig


class HandDetector:
    """
    Wrapper class for MediaPipe Hands detection (API 0.10.30+).

    This class encapsulates MediaPipe's hand detection functionality,
    providing a cleaner interface and handling common operations like:
    - Hand detection in images
    - Landmark extraction
    - Visualization of results

    Attributes:
        detector: MediaPipe HandLandmarker instance
        running_mode: LIVE_STREAM or IMAGE mode
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = MediaPipeConfig.MAX_NUM_HANDS,
        min_detection_confidence: float = MediaPipeConfig.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MediaPipeConfig.MIN_TRACKING_CONFIDENCE,
        model_complexity: int = MediaPipeConfig.MODEL_COMPLEXITY
    ):
        """
        Initialize the HandDetector with new MediaPipe API (0.10.30+).

        Args:
            static_image_mode: If True, treats each image as independent.
                             If False, uses temporal information for tracking.
            max_num_hands: Maximum number of hands to detect (1 or 2)
            min_detection_confidence: Minimum confidence for hand detection (0.0-1.0)
            min_tracking_confidence: Minimum confidence for hand tracking (0.0-1.0)
            model_complexity: Model complexity (0 for lite, 1 for full)
        """
        try:
            # Try legacy API first (0.10.8 and earlier)
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles

            self.hands = self.mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                model_complexity=model_complexity
            )

            self.static_image_mode = static_image_mode
            self.use_legacy_api = True
            print("[INFO] Using MediaPipe legacy API")

        except AttributeError:
            # New API (0.10.30+)
            print("[INFO] Using MediaPipe new API (0.10.30+)")
            print("[WARNING] New API has limited functionality. Consider using Python 3.10 with mediapipe==0.10.8")
            self.use_legacy_api = False
            # For now, we'll provide a simplified implementation
            # The new API requires downloading model files and is more complex
            raise NotImplementedError(
                "MediaPipe 0.10.30+ requires new API setup. "
                "Please install Python 3.10 or 3.11 and use: pip install mediapipe==0.10.8"
            )

    def detect_hands(self, image: np.ndarray) -> Tuple[bool, Optional[any]]:
        """
        Detect hands in an image and return results.

        Args:
            image: Input image in BGR format (OpenCV standard)

        Returns:
            Tuple of (success, results):
                - success: Boolean indicating if hand was detected
                - results: MediaPipe results object containing landmarks

        Technical Details:
        - Converts BGR to RGB (MediaPipe requirement)
        - Processes image to detect hands and extract landmarks
        - Returns None if no hands detected

        Example:
            >>> detector = HandDetector()
            >>> success, results = detector.detect_hands(frame)
            >>> if success:
            >>>     landmarks = results.multi_hand_landmarks[0]
        """
        # Convert BGR to RGB (MediaPipe uses RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Improve performance by marking image as not writeable
        image_rgb.flags.writeable = False

        # Process the image
        results = self.hands.process(image_rgb)

        # Check if any hands were detected
        if results.multi_hand_landmarks:
            return True, results
        else:
            return False, None

    def get_landmarks(
        self,
        results: any,
        hand_index: int = 0
    ) -> Optional[np.ndarray]:
        """
        Extract hand landmarks from MediaPipe results.

        Args:
            results: MediaPipe results object from detect_hands()
            hand_index: Index of hand to extract (0 for first hand)

        Returns:
            NumPy array of shape (21, 3) containing [x, y, z] for each landmark,
            or None if no landmarks available

        Landmark Indices (MediaPipe Hand Landmark Model):
            0: WRIST
            1-4: THUMB (CMC, MCP, IP, TIP)
            5-8: INDEX FINGER (MCP, PIP, DIP, TIP)
            9-12: MIDDLE FINGER (MCP, PIP, DIP, TIP)
            13-16: RING FINGER (MCP, PIP, DIP, TIP)
            17-20: PINKY (MCP, PIP, DIP, TIP)

        Coordinate System:
        - x: Horizontal position (0 = left, 1 = right)
        - y: Vertical position (0 = top, 1 = bottom)
        - z: Depth relative to wrist (negative = closer to camera)
        - All coordinates normalized to [0, 1] based on image dimensions

        Academic Note:
        - Normalization ensures scale invariance
        - Z-coordinate provides depth information for 3D gestures
        """
        if not results or not results.multi_hand_landmarks:
            return None

        if hand_index >= len(results.multi_hand_landmarks):
            return None

        hand_landmarks = results.multi_hand_landmarks[hand_index]

        # Extract landmarks as numpy array
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])

        return np.array(landmarks, dtype=np.float32)

    def get_landmarks_pixel_coords(
        self,
        results: any,
        image_shape: Tuple[int, int],
        hand_index: int = 0
    ) -> Optional[np.ndarray]:
        """
        Extract hand landmarks in pixel coordinates.

        Args:
            results: MediaPipe results object
            image_shape: (height, width) of the image
            hand_index: Index of hand to extract

        Returns:
            NumPy array of shape (21, 3) containing [x_pixel, y_pixel, z_depth]
            or None if no landmarks available

        Note:
        - x and y are converted to pixel coordinates
        - z remains as depth value (not scaled)
        """
        landmarks_normalized = self.get_landmarks(results, hand_index)

        if landmarks_normalized is None:
            return None

        height, width = image_shape[:2]

        # Convert normalized coordinates to pixel coordinates
        landmarks_pixel = landmarks_normalized.copy()
        landmarks_pixel[:, 0] *= width   # x: 0-1 → 0-width
        landmarks_pixel[:, 1] *= height  # y: 0-1 → 0-height
        # z remains unchanged (depth)

        return landmarks_pixel.astype(np.int32)

    def draw_landmarks(
        self,
        image: np.ndarray,
        results: any,
        draw_connections: bool = True
    ) -> np.ndarray:
        """
        Draw hand landmarks and connections on image.

        Args:
            image: Input image in BGR format
            results: MediaPipe results object
            draw_connections: If True, draws lines connecting landmarks

        Returns:
            Image with landmarks drawn

        Visualization Details:
        - Landmarks are drawn as colored circles
        - Connections show hand skeleton structure
        - Uses MediaPipe's default styling
        """
        if not results or not results.multi_hand_landmarks:
            return image

        annotated_image = image.copy()

        for hand_landmarks in results.multi_hand_landmarks:
            if draw_connections:
                # Draw hand connections (skeleton)
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            else:
                # Draw only landmarks (no connections)
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    None
                )

        return annotated_image

    def get_bounding_box(
        self,
        landmarks_pixel: np.ndarray,
        margin: int = 10
    ) -> Tuple[int, int, int, int]:
        """
        Calculate bounding box around hand landmarks.

        Args:
            landmarks_pixel: Hand landmarks in pixel coordinates
            margin: Margin to add around bounding box (pixels)

        Returns:
            Tuple (x_min, y_min, x_max, y_max) defining bounding box

        Use Case:
        - Cropping hand region for processing
        - Visualizing hand detection area
        - Region of interest extraction
        """
        x_coords = landmarks_pixel[:, 0]
        y_coords = landmarks_pixel[:, 1]

        x_min = max(0, int(np.min(x_coords)) - margin)
        y_min = max(0, int(np.min(y_coords)) - margin)
        x_max = int(np.max(x_coords)) + margin
        y_max = int(np.max(y_coords)) + margin

        return x_min, y_min, x_max, y_max

    def close(self):
        """
        Release MediaPipe resources.

        Call this when done using the detector to free up resources.
        """
        self.hands.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def test_hand_detector(camera_id: int = 0):
    """
    Test the HandDetector with live webcam feed.

    This function demonstrates basic usage of the HandDetector class.
    Press 'q' to quit.

    Args:
        camera_id: ID of camera to use (default 0)
    """
    print("Testing Hand Detector...")
    print("Press 'q' to quit")

    # Initialize detector
    detector = HandDetector(
        static_image_mode=False,  # Enable tracking for video
        max_num_hands=1
    )

    # Open webcam
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    while True:
        success, frame = cap.read()

        if not success:
            print("Failed to read frame")
            break

        # Detect hands
        hands_detected, results = detector.detect_hands(frame)

        if hands_detected:
            # Draw landmarks
            frame = detector.draw_landmarks(frame, results)

            # Get landmarks in pixel coordinates
            landmarks = detector.get_landmarks_pixel_coords(
                results,
                frame.shape
            )

            if landmarks is not None:
                # Draw bounding box
                x_min, y_min, x_max, y_max = detector.get_bounding_box(landmarks)
                cv2.rectangle(
                    frame,
                    (x_min, y_min),
                    (x_max, y_max),
                    (0, 255, 255),
                    2
                )

                # Display landmark count
                cv2.putText(
                    frame,
                    f"Landmarks: {len(landmarks)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
        else:
            # No hand detected
            cv2.putText(
                frame,
                "No hand detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # Display frame
        cv2.imshow('Hand Detector Test', frame)

        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    print("Test complete!")


if __name__ == "__main__":
    # Run test when executed directly
    test_hand_detector()
