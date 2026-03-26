"""
Hand Detection Module

Wraps MediaPipe Hands to provide a clean interface for landmark detection.
MediaPipe returns normalized (x, y, z) coordinates relative to the image frame.
"""

import cv2
import mediapipe as mp
import numpy as np

from src.utils.config import MediaPipeConfig


class HandDetector:
    """
    Wrapper around MediaPipe Hands for hand landmark detection.

    Detects up to MAX_NUM_HANDS hands per frame and returns their 21
    landmarks as (x, y, z) tuples in image-normalized coordinates.
    """

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MediaPipeConfig.MAX_NUM_HANDS,
            model_complexity=MediaPipeConfig.MODEL_COMPLEXITY,
            min_detection_confidence=MediaPipeConfig.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MediaPipeConfig.MIN_TRACKING_CONFIDENCE,
        )

    def detect(self, frame):
        """
        Run hand detection on a BGR frame.

        Args:
            frame: BGR image as a numpy array (from cv2.VideoCapture).

        Returns:
            (annotated_frame, landmarks_list) where:
              - annotated_frame is a copy of frame with landmarks drawn on it.
              - landmarks_list is a list of lists; each inner list contains
                21 (x, y, z) tuples for one detected hand.
            Returns (annotated_frame, None) when no hand is detected.
        """
        # MediaPipe requires RGB input
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        annotated_frame = frame.copy()

        if not results.multi_hand_landmarks:
            return annotated_frame, None

        landmarks_list = []
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the annotated copy
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style(),
            )

            # Extract the 21 (x, y, z) tuples for this hand
            hand_coords = [
                (lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark
            ]
            landmarks_list.append(hand_coords)

        return annotated_frame, landmarks_list

    def get_landmark_array(self, landmarks):
        """
        Flatten one hand's landmark list into a 1-D numpy array.

        Args:
            landmarks: list of 21 (x, y, z) tuples for a single hand.

        Returns:
            numpy array of shape (63,) — [x0, y0, z0, x1, y1, z1, ...].
        """
        return np.array(landmarks, dtype=np.float32).flatten()  # (21*3,) = (63,)

    def draw_landmarks(self, frame, landmarks):
        """
        Draw hand landmarks on a frame without running detection again.

        Args:
            frame:     BGR image array (modified in-place).
            landmarks: list of 21 (x, y, z) tuples for one hand.

        Returns:
            The modified frame.
        """
        # Reconstruct a NormalizedLandmarkList so we can reuse mp_drawing
        hand_landmarks_proto = self.mp_hands.HandLandmark
        landmark_list = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
        for x, y, z in landmarks:
            lm = landmark_list.landmark.add()
            lm.x, lm.y, lm.z = float(x), float(y), float(z)

        self.mp_drawing.draw_landmarks(
            frame,
            landmark_list,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style(),
        )
        return frame

    def release(self):
        """Release MediaPipe resources."""
        self.hands.close()
