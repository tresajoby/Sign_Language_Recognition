"""
Configuration Management Module

This module centralizes all configuration parameters for the ASL Recognition System.
Using a configuration file ensures reproducibility and makes hyperparameter tuning systematic.

Academic Justification:
- Centralized configuration prevents magic numbers scattered across code
- Easy to document all experimental settings for thesis methodology
- Supports reproducibility of results
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Root directory
ROOT_DIR = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LABELS_DIR = DATA_DIR / "labels"

# Model directories
MODELS_DIR = ROOT_DIR / "models"

# Source directories
SRC_DIR = ROOT_DIR / "src"

# ============================================================================
# MEDIAPIPE CONFIGURATION
# ============================================================================

class MediaPipeConfig:
    """MediaPipe Hand Detector Configuration"""

    # Detection confidence threshold (0.0 to 1.0)
    # Higher values = fewer false positives, may miss hands
    # Lower values = more detections, may have false positives
    MIN_DETECTION_CONFIDENCE = 0.7

    # Tracking confidence threshold
    # Higher values = more stable tracking but may lose track easily
    MIN_TRACKING_CONFIDENCE = 0.5

    # Maximum number of hands to detect
    # For ASL: typically 1 or 2 hands
    MAX_NUM_HANDS = 1

    # Model complexity (0 or 1)
    # 0 = Lite model (faster, less accurate)
    # 1 = Full model (slower, more accurate)
    MODEL_COMPLEXITY = 1


# ============================================================================
# DATA COLLECTION CONFIGURATION
# ============================================================================

class DataCollectionConfig:
    """Configuration for data collection phase"""

    # Static gestures (A-Z letters and 0-9 digits)
    STATIC_CLASSES = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
    ]

    # Dynamic gestures (motion-based signs)
    DYNAMIC_CLASSES = [
        'hello', 'thanks', 'please', 'sorry', 'yes', 'no',
        'help', 'stop', 'more', 'finish'
    ]

    # Samples per class for static gestures
    STATIC_SAMPLES_PER_CLASS = 300

    # Sequence length for dynamic gestures (frames)
    DYNAMIC_SEQUENCE_LENGTH = 30

    # Sequences per class for dynamic gestures
    DYNAMIC_SEQUENCES_PER_CLASS = 100

    # Webcam configuration
    CAMERA_ID = 0  # Default webcam
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS = 30


# ============================================================================
# FEATURE EXTRACTION CONFIGURATION
# ============================================================================

class FeatureConfig:
    """Configuration for feature extraction and preprocessing"""

    # MediaPipe returns 21 landmarks per hand
    # Each landmark has (x, y, z) coordinates
    NUM_LANDMARKS = 21
    COORDS_PER_LANDMARK = 3  # x, y, z
    FEATURE_DIM = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 63

    # Normalization strategy
    # Options: 'wrist', 'bbox', 'none'
    # 'wrist': Normalize relative to wrist position (landmark 0)
    # 'bbox': Normalize to bounding box
    NORMALIZATION_METHOD = 'wrist'

    # Whether to include distance features
    # (distances between key landmarks)
    INCLUDE_DISTANCE_FEATURES = False

    # Whether to include angle features
    INCLUDE_ANGLE_FEATURES = False


# ============================================================================
# MODEL ARCHITECTURE CONFIGURATION
# ============================================================================

class StaticModelConfig:
    """Configuration for static gesture MLP model"""

    # Input dimension (21 landmarks × 3 coords)
    INPUT_DIM = FeatureConfig.FEATURE_DIM

    # Number of output classes
    NUM_CLASSES = len(DataCollectionConfig.STATIC_CLASSES)

    # Hidden layer sizes
    HIDDEN_LAYERS = [128, 64, 32]

    # Dropout rate for regularization
    DROPOUT_RATE = 0.3

    # Activation function
    ACTIVATION = 'relu'

    # Output activation
    OUTPUT_ACTIVATION = 'softmax'


class DynamicModelConfig:
    """Configuration for dynamic gesture BiLSTM model"""

    # Input shape: (sequence_length, feature_dim)
    SEQUENCE_LENGTH = DataCollectionConfig.DYNAMIC_SEQUENCE_LENGTH
    FEATURE_DIM = FeatureConfig.FEATURE_DIM

    # Number of output classes
    NUM_CLASSES = len(DataCollectionConfig.DYNAMIC_CLASSES)

    # LSTM units per layer
    LSTM_UNITS = [64, 32]

    # Whether to use bidirectional LSTM
    BIDIRECTIONAL = True

    # Dropout rate
    DROPOUT_RATE = 0.3

    # Recurrent dropout
    RECURRENT_DROPOUT = 0.2


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Configuration for model training"""

    # Train/validation/test split
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15

    # Batch size
    BATCH_SIZE = 32

    # Number of epochs
    EPOCHS = 50

    # Learning rate
    LEARNING_RATE = 0.001

    # Optimizer
    OPTIMIZER = 'adam'

    # Loss function
    LOSS = 'categorical_crossentropy'

    # Metrics to track
    METRICS = ['accuracy']

    # Early stopping patience
    EARLY_STOPPING_PATIENCE = 10

    # Random seed for reproducibility
    RANDOM_SEED = 42


# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================

class InferenceConfig:
    """Configuration for real-time inference"""

    # Confidence threshold for predictions
    # Only show predictions above this threshold
    CONFIDENCE_THRESHOLD = 0.7

    # Buffer size for dynamic gesture detection
    # Number of frames to collect before making prediction
    DYNAMIC_BUFFER_SIZE = DataCollectionConfig.DYNAMIC_SEQUENCE_LENGTH

    # Display settings
    DISPLAY_LANDMARKS = True
    DISPLAY_BBOX = True
    DISPLAY_FPS = True

    # Colors (BGR format for OpenCV)
    LANDMARK_COLOR = (0, 255, 0)  # Green
    CONNECTION_COLOR = (255, 0, 0)  # Blue
    TEXT_COLOR = (255, 255, 255)  # White
    BBOX_COLOR = (0, 255, 255)  # Yellow


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

class EvaluationConfig:
    """Configuration for model evaluation"""

    # Metrics to compute
    COMPUTE_ACCURACY = True
    COMPUTE_PRECISION = True
    COMPUTE_RECALL = True
    COMPUTE_F1_SCORE = True
    COMPUTE_CONFUSION_MATRIX = True

    # Whether to save evaluation plots
    SAVE_PLOTS = True

    # Plot directory
    PLOTS_DIR = ROOT_DIR / "docs" / "plots"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_directories():
    """
    Create all necessary directories if they don't exist.

    This function should be called at the start of the project
    to ensure all required directories are present.
    """
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        RAW_DATA_DIR / "static",
        RAW_DATA_DIR / "dynamic",
        PROCESSED_DATA_DIR,
        LABELS_DIR,
        MODELS_DIR,
        EvaluationConfig.PLOTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        # Create .gitkeep to track empty directories
        gitkeep = directory / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()


def get_model_path(model_type: str) -> Path:
    """
    Get the path for saving/loading a trained model.

    Args:
        model_type: Either 'static' or 'dynamic'

    Returns:
        Path object pointing to the model file
    """
    if model_type == 'static':
        return MODELS_DIR / "static_model.h5"
    elif model_type == 'dynamic':
        return MODELS_DIR / "dynamic_model.h5"
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def print_configuration():
    """
    Print all configuration parameters.
    Useful for thesis documentation and reproducibility.
    """
    print("=" * 70)
    print("ASL RECOGNITION SYSTEM - CONFIGURATION")
    print("=" * 70)

    print("\n[MediaPipe Configuration]")
    for key, value in MediaPipeConfig.__dict__.items():
        if not key.startswith('_'):
            print(f"  {key}: {value}")

    print("\n[Data Collection Configuration]")
    print(f"  Static Classes: {len(DataCollectionConfig.STATIC_CLASSES)}")
    print(f"  Dynamic Classes: {len(DataCollectionConfig.DYNAMIC_CLASSES)}")
    print(f"  Static Samples/Class: {DataCollectionConfig.STATIC_SAMPLES_PER_CLASS}")
    print(f"  Dynamic Sequences/Class: {DataCollectionConfig.DYNAMIC_SEQUENCES_PER_CLASS}")

    print("\n[Feature Configuration]")
    print(f"  Feature Dimension: {FeatureConfig.FEATURE_DIM}")
    print(f"  Normalization: {FeatureConfig.NORMALIZATION_METHOD}")

    print("\n[Static Model Configuration]")
    print(f"  Input Dim: {StaticModelConfig.INPUT_DIM}")
    print(f"  Output Classes: {StaticModelConfig.NUM_CLASSES}")
    print(f"  Hidden Layers: {StaticModelConfig.HIDDEN_LAYERS}")

    print("\n[Dynamic Model Configuration]")
    print(f"  Sequence Length: {DynamicModelConfig.SEQUENCE_LENGTH}")
    print(f"  LSTM Units: {DynamicModelConfig.LSTM_UNITS}")
    print(f"  Bidirectional: {DynamicModelConfig.BIDIRECTIONAL}")

    print("\n[Training Configuration]")
    print(f"  Batch Size: {TrainingConfig.BATCH_SIZE}")
    print(f"  Epochs: {TrainingConfig.EPOCHS}")
    print(f"  Learning Rate: {TrainingConfig.LEARNING_RATE}")
    print(f"  Random Seed: {TrainingConfig.RANDOM_SEED}")

    print("=" * 70)


if __name__ == "__main__":
    # Create directories and print configuration
    create_directories()
    print_configuration()
