"""
Feature Extraction Module

This module handles the conversion of raw hand landmarks into normalized feature
vectors suitable for machine learning models. It implements various normalization
strategies to achieve scale and position invariance.

Academic Justification:
- Normalization is critical for gesture recognition robustness
- Wrist-relative normalization provides:
  * Scale invariance: Works with different hand sizes
  * Position invariance: Hand location doesn't affect features
  * Rotation partial invariance: Preserves hand shape
- Feature dimension: 21 landmarks × 3 coords = 63 features

Technical References:
- Hand pose normalization techniques in gesture recognition literature
- Feature engineering for temporal sequence modeling
"""

import numpy as np
from typing import Optional, List, Tuple
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import FeatureConfig


class FeatureExtractor:
    """
    Extract and normalize features from hand landmarks.

    This class handles the preprocessing of raw MediaPipe landmarks into
    normalized feature vectors that are invariant to hand size, position,
    and partially invariant to rotation.

    Methods:
        - normalize_landmarks: Apply normalization to landmarks
        - extract_features: Convert landmarks to feature vector
        - extract_distance_features: Calculate inter-landmark distances
        - extract_angle_features: Calculate joint angles
    """

    def __init__(
        self,
        normalization_method: str = FeatureConfig.NORMALIZATION_METHOD,
        include_distance_features: bool = FeatureConfig.INCLUDE_DISTANCE_FEATURES,
        include_angle_features: bool = FeatureConfig.INCLUDE_ANGLE_FEATURES
    ):
        """
        Initialize FeatureExtractor.

        Args:
            normalization_method: 'wrist', 'bbox', or 'none'
            include_distance_features: Whether to add distance features
            include_angle_features: Whether to add angle features

        Academic Note:
        - Different normalization strategies have trade-offs
        - Wrist-relative: Good for most gestures, fast computation
        - BBox: More invariant to rotation, slightly more complex
        - None: Use for pre-normalized data or debugging
        """
        self.normalization_method = normalization_method
        self.include_distance_features = include_distance_features
        self.include_angle_features = include_angle_features

    def normalize_landmarks_wrist(
        self,
        landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Normalize landmarks relative to wrist position.

        This is the recommended normalization method for ASL recognition.

        Algorithm:
        1. Translate all landmarks so wrist is at origin
        2. Scale by hand size (distance from wrist to middle finger MCP)
        3. Maintains hand shape while removing position and scale

        Args:
            landmarks: Array of shape (21, 3) with [x, y, z] coordinates

        Returns:
            Normalized landmarks of shape (21, 3)

        Mathematical Formulation:
            For landmark i with coordinates (x_i, y_i, z_i):

            x_norm_i = (x_i - x_wrist) / hand_size
            y_norm_i = (y_i - y_wrist) / hand_size
            z_norm_i = (z_i - z_wrist) / hand_size

            where hand_size = ||landmark_9 - landmark_0||

        Technical Justification:
        - Landmark 0: Wrist (base reference point)
        - Landmark 9: Middle finger MCP (stable, representative of hand size)
        - Division by hand_size provides scale invariance
        - Subtraction of wrist provides translation invariance
        """
        if landmarks.shape[0] != 21:
            raise ValueError(f"Expected 21 landmarks, got {landmarks.shape[0]}")

        # Get wrist position (landmark 0)
        wrist = landmarks[0].copy()

        # Translate to wrist origin
        landmarks_translated = landmarks - wrist

        # Calculate hand size (wrist to middle finger MCP)
        hand_size = np.linalg.norm(landmarks_translated[9])

        # Avoid division by zero
        if hand_size < 1e-6:
            hand_size = 1.0

        # Normalize by hand size
        landmarks_normalized = landmarks_translated / hand_size

        return landmarks_normalized

    def normalize_landmarks_bbox(
        self,
        landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Normalize landmarks using bounding box.

        This method normalizes landmarks to fit within [0, 1] range
        based on the bounding box of the hand.

        Algorithm:
        1. Find min and max coordinates in each dimension
        2. Translate so minimum is at origin
        3. Scale to [0, 1] range

        Args:
            landmarks: Array of shape (21, 3)

        Returns:
            Normalized landmarks of shape (21, 3)

        Use Case:
        - More invariant to rotation than wrist-relative
        - Useful when hand orientation varies significantly
        - Slightly more computationally expensive

        Limitation:
        - May distort aspect ratio if hand dimensions differ greatly
        """
        # Find bounding box
        min_coords = np.min(landmarks, axis=0)
        max_coords = np.max(landmarks, axis=0)

        # Calculate range
        coord_range = max_coords - min_coords

        # Avoid division by zero
        coord_range[coord_range < 1e-6] = 1.0

        # Normalize to [0, 1]
        landmarks_normalized = (landmarks - min_coords) / coord_range

        return landmarks_normalized

    def normalize_landmarks(
        self,
        landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Normalize landmarks using configured method.

        Args:
            landmarks: Array of shape (21, 3)

        Returns:
            Normalized landmarks

        Raises:
            ValueError: If landmarks shape is invalid or method unknown
        """
        if landmarks is None or len(landmarks) == 0:
            raise ValueError("Landmarks cannot be None or empty")

        if self.normalization_method == 'wrist':
            return self.normalize_landmarks_wrist(landmarks)
        elif self.normalization_method == 'bbox':
            return self.normalize_landmarks_bbox(landmarks)
        elif self.normalization_method == 'none':
            return landmarks
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")

    def extract_basic_features(
        self,
        landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Extract basic feature vector from landmarks.

        This converts the (21, 3) landmark array into a flattened
        feature vector of size 63.

        Args:
            landmarks: Normalized landmarks of shape (21, 3)

        Returns:
            Feature vector of shape (63,)

        Feature Vector Structure:
            [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]

        This is the standard representation used for:
        - Static gesture classification (MLP input)
        - Dynamic gesture sequences (BiLSTM input per frame)
        """
        return landmarks.flatten()

    def extract_distance_features(
        self,
        landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Extract distance features between key landmarks.

        Calculates Euclidean distances between important landmark pairs.
        These features can capture hand configurations that may be invariant
        to certain transformations.

        Args:
            landmarks: Normalized landmarks of shape (21, 3)

        Returns:
            Array of distance features

        Key Distances (examples):
        - Thumb tip to index tip (pinch detection)
        - Thumb tip to middle tip
        - Index tip to middle tip
        - Finger lengths (tip to base)

        Academic Note:
        - Distance features are rotation-invariant
        - Useful for detecting specific hand configurations
        - Increases feature dimensionality
        """
        distances = []

        # Thumb to other fingertips
        for tip_idx in [8, 12, 16, 20]:  # Index, middle, ring, pinky tips
            dist = np.linalg.norm(landmarks[4] - landmarks[tip_idx])
            distances.append(dist)

        # Consecutive fingertip distances
        fingertips = [4, 8, 12, 16, 20]
        for i in range(len(fingertips) - 1):
            dist = np.linalg.norm(landmarks[fingertips[i]] - landmarks[fingertips[i+1]])
            distances.append(dist)

        # Finger lengths (tip to MCP)
        finger_pairs = [(4, 2), (8, 5), (12, 9), (16, 13), (20, 17)]
        for tip, base in finger_pairs:
            dist = np.linalg.norm(landmarks[tip] - landmarks[base])
            distances.append(dist)

        return np.array(distances, dtype=np.float32)

    def extract_angle_features(
        self,
        landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Extract angle features from hand joints.

        Calculates angles at key joints using vectors between landmarks.
        Useful for capturing joint configurations.

        Args:
            landmarks: Normalized landmarks of shape (21, 3)

        Returns:
            Array of angle features (in radians)

        Example Angles:
        - Finger joint angles (PIP, DIP)
        - Palm angles
        - Finger spread angles

        Mathematical Note:
        - Angle between vectors A and B:
          θ = arccos((A · B) / (||A|| ||B||))
        """
        angles = []

        # Finger joint angles (simplified example)
        # For each finger, calculate angle at middle joint

        fingers = [
            (2, 3, 4),   # Thumb
            (5, 6, 7),   # Index proximal joints
            (9, 10, 11), # Middle proximal joints
            (13, 14, 15),# Ring proximal joints
            (17, 18, 19) # Pinky proximal joints
        ]

        for p1, p2, p3 in fingers:
            # Vectors
            v1 = landmarks[p1] - landmarks[p2]
            v2 = landmarks[p3] - landmarks[p2]

            # Normalize
            v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
            v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)

            # Angle (dot product)
            cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
            angle = np.arccos(cos_angle)

            angles.append(angle)

        return np.array(angles, dtype=np.float32)

    def extract_features(
        self,
        landmarks: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Extract complete feature vector from raw landmarks.

        This is the main feature extraction function that combines
        all configured feature types.

        Args:
            landmarks: Raw landmarks from MediaPipe, shape (21, 3)
            normalize: Whether to normalize landmarks first

        Returns:
            Complete feature vector

        Feature Vector Composition:
        - Basic: 63 features (always included)
        - Distance: ~14 features (if enabled)
        - Angle: ~5 features (if enabled)

        Total dimension varies based on configuration.

        Usage Example:
            >>> extractor = FeatureExtractor()
            >>> features = extractor.extract_features(raw_landmarks)
            >>> # features.shape = (63,) for default config
        """
        if landmarks is None:
            raise ValueError("Landmarks cannot be None")

        # Normalize landmarks
        if normalize:
            landmarks_normalized = self.normalize_landmarks(landmarks)
        else:
            landmarks_normalized = landmarks

        # Extract basic features (flattened landmarks)
        features = [self.extract_basic_features(landmarks_normalized)]

        # Add distance features if enabled
        if self.include_distance_features:
            distance_feats = self.extract_distance_features(landmarks_normalized)
            features.append(distance_feats)

        # Add angle features if enabled
        if self.include_angle_features:
            angle_feats = self.extract_angle_features(landmarks_normalized)
            features.append(angle_feats)

        # Concatenate all features
        feature_vector = np.concatenate(features)

        return feature_vector

    def get_feature_dimension(self) -> int:
        """
        Get the total feature dimension based on current configuration.

        Returns:
            Total number of features in output vector

        Use Case:
        - Determining model input size
        - Validating feature vectors
        - Documentation
        """
        dim = FeatureConfig.FEATURE_DIM  # Basic: 63

        if self.include_distance_features:
            dim += 14  # Distance features

        if self.include_angle_features:
            dim += 5  # Angle features

        return dim


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def test_feature_extractor():
    """
    Test the FeatureExtractor with sample data.

    Creates synthetic landmarks and demonstrates feature extraction.
    """
    print("Testing Feature Extractor...")

    # Create sample landmarks (21 points)
    # In practice, these come from MediaPipe
    np.random.seed(42)
    sample_landmarks = np.random.rand(21, 3).astype(np.float32)

    # Initialize extractor
    extractor = FeatureExtractor(
        normalization_method='wrist',
        include_distance_features=False,
        include_angle_features=False
    )

    # Extract features
    features = extractor.extract_features(sample_landmarks)

    print(f"Input landmarks shape: {sample_landmarks.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Feature dimension: {extractor.get_feature_dimension()}")
    print(f"\nFirst 10 features: {features[:10]}")
    print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")

    # Test with distance features
    print("\n" + "="*50)
    print("Testing with distance features...")

    extractor_with_dist = FeatureExtractor(
        normalization_method='wrist',
        include_distance_features=True,
        include_angle_features=False
    )

    features_with_dist = extractor_with_dist.extract_features(sample_landmarks)

    print(f"Feature dimension with distances: {extractor_with_dist.get_feature_dimension()}")
    print(f"Output shape: {features_with_dist.shape}")

    print("\nTest complete!")


if __name__ == "__main__":
    test_feature_extractor()
