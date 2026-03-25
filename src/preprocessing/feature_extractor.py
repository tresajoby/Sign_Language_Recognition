"""
Feature Extraction Module

Normalizes raw MediaPipe landmark coordinates so the model is invariant to
hand position within the camera frame.  The default strategy ('wrist')
subtracts landmark 0 (the wrist) from every other landmark, making all
coordinates relative to the wrist position.
"""

import numpy as np

from src.utils.config import FeatureConfig


class FeatureExtractor:
    """
    Transforms raw 63-dimensional landmark arrays into normalized features.

    Normalization method is controlled by FeatureConfig.NORMALIZATION_METHOD.
    Currently supports 'wrist' normalization; 'none' passes the array through
    unchanged.
    """

    def extract(self, landmark_array):
        """
        Normalize a single frame's landmark array.

        Args:
            landmark_array: numpy array of shape (63,) —
                            [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20].

        Returns:
            Normalized numpy array of shape (63,).
        """
        features = landmark_array.copy().astype(np.float32)

        if FeatureConfig.NORMALIZATION_METHOD == 'wrist':
            # Wrist is landmark 0, stored at indices 0, 1, 2
            wrist_x = features[0]
            wrist_y = features[1]
            wrist_z = features[2]

            # Subtract wrist coords from every landmark's (x, y, z)
            features[0::3] -= wrist_x  # every x
            features[1::3] -= wrist_y  # every y
            features[2::3] -= wrist_z  # every z

        # 'none' or any other value: return as-is
        return features

    def extract_from_sequence(self, sequence):
        """
        Normalize an entire sequence of frames.

        Args:
            sequence: numpy array of shape (T, 63).

        Returns:
            Normalized numpy array of shape (T, 63).
        """
        return np.array([self.extract(frame) for frame in sequence],
                        dtype=np.float32)
