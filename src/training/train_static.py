"""
Training Script for Static Gesture Model

This script handles the complete training pipeline for the static gesture MLP:
1. Load preprocessed data
2. Split into train/validation/test sets
3. Train the model
4. Evaluate performance
5. Save trained model and results

Usage:
    python src/training/train_static.py
"""

import numpy as np
import json
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.static_model import create_static_model
from src.utils.config import (
    TrainingConfig,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    LABELS_DIR,
    DataCollectionConfig
)


class StaticModelTrainer:
    """
    Trainer class for static gesture model.

    Handles the entire training workflow from data loading to model saving.
    """

    def __init__(self):
        """Initialize the trainer."""
        self.model = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.label_mapping = None
        self.history = None

    def load_data(self):
        """
        Load preprocessed data from disk.

        Expected files:
        - data/processed/static_features.npy: (N, 63) feature array
        - data/processed/static_labels.npy: (N,) label array
        - data/labels/static_label_mapping.json: {0: 'A', 1: 'B', ...}

        Returns:
            Tuple of (features, labels, label_mapping)
        """
        print("\n" + "="*70)
        print("LOADING DATA")
        print("="*70)

        features_path = PROCESSED_DATA_DIR / "static_features.npy"
        labels_path = PROCESSED_DATA_DIR / "static_labels.npy"
        mapping_path = LABELS_DIR / "static_label_mapping.json"

        # Check if files exist
        if not features_path.exists():
            raise FileNotFoundError(
                f"Features file not found: {features_path}\n"
                "Please run data collection first:\n"
                "  python src/data_collection/collect_static.py"
            )

        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        # Load data
        features = np.load(features_path)
        labels = np.load(labels_path)

        # Load label mapping
        if mapping_path.exists():
            with open(mapping_path, 'r') as f:
                # Convert string keys to int
                label_mapping = {int(k): v for k, v in json.load(f).items()}
        else:
            # Create default mapping
            label_mapping = {i: chr(65 + i) if i < 26 else str(i - 26)
                           for i in range(len(DataCollectionConfig.STATIC_CLASSES))}

        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Number of classes: {len(np.unique(labels))}")
        print(f"Feature dimension: {features.shape[1]}")
        print(f"Label mapping: {label_mapping}")

        self.label_mapping = label_mapping

        return features, labels, label_mapping

    def split_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        train_split: float = TrainingConfig.TRAIN_SPLIT,
        val_split: float = TrainingConfig.VAL_SPLIT,
        test_split: float = TrainingConfig.TEST_SPLIT,
        random_seed: int = TrainingConfig.RANDOM_SEED
    ):
        """
        Split data into train/validation/test sets.

        Args:
            features: Feature array (N, 63)
            labels: Label array (N,)
            train_split: Proportion for training
            val_split: Proportion for validation
            test_split: Proportion for testing
            random_seed: Random seed for reproducibility

        Academic Note:
        - Stratified split ensures balanced class distribution across splits
        - Standard split: 70% train, 15% validation, 15% test
        - Validation set used for hyperparameter tuning and early stopping
        - Test set held out for final unbiased evaluation
        """
        print("\n" + "="*70)
        print("SPLITTING DATA")
        print("="*70)

        # Verify splits sum to 1.0
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
            "Train/val/test splits must sum to 1.0"

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            features,
            labels,
            test_size=test_split,
            random_state=random_seed,
            stratify=labels  # Maintain class distribution
        )

        # Second split: separate train and validation
        val_proportion = val_split / (train_split + val_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_proportion,
            random_state=random_seed,
            stratify=y_temp
        )

        print(f"Train set: {X_train.shape[0]} samples ({train_split*100:.1f}%)")
        print(f"Validation set: {X_val.shape[0]} samples ({val_split*100:.1f}%)")
        print(f"Test set: {X_test.shape[0]} samples ({test_split*100:.1f}%)")

        # Convert labels to one-hot encoding
        num_classes = len(np.unique(labels))
        y_train_onehot = keras.utils.to_categorical(y_train, num_classes)
        y_val_onehot = keras.utils.to_categorical(y_val, num_classes)
        y_test_onehot = keras.utils.to_categorical(y_test, num_classes)

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train_onehot
        self.y_val = y_val_onehot
        self.y_test = y_test_onehot

        return X_train, X_val, X_test, y_train_onehot, y_val_onehot, y_test_onehot

    def create_model(self, num_classes: int):
        """
        Create and compile the static gesture model.

        Args:
            num_classes: Number of gesture classes
        """
        print("\n" + "="*70)
        print("CREATING MODEL")
        print("="*70)

        self.model = create_static_model(num_classes=num_classes, compile=True)
        self.model.summary()

    def train(
        self,
        epochs: int = TrainingConfig.EPOCHS,
        batch_size: int = TrainingConfig.BATCH_SIZE
    ):
        """
        Train the model.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")

        self.history = self.model.train(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            epochs=epochs,
            batch_size=batch_size
        )

    def evaluate(self):
        """
        Evaluate the model on test set.

        Returns:
            Tuple of (loss, accuracy)
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")

        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)

        return loss, accuracy

    def save_model(self, filename: str = "static_model.h5"):
        """
        Save the trained model.

        Args:
            filename: Name of model file
        """
        if self.model is None:
            raise ValueError("No model to save.")

        filepath = MODELS_DIR / filename
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        self.model.save(str(filepath))
        print(f"\n[INFO] Model saved to {filepath}")

    def plot_training_history(self, save_path: str = None):
        """
        Plot training history (loss and accuracy curves).

        Args:
            save_path: Path to save plot (optional)
        """
        if self.history is None:
            print("[WARNING] No training history to plot.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot loss
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Plot accuracy
        axes[1].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[1].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Training plot saved to {save_path}")

        plt.show()


def main():
    """
    Main training workflow.
    """
    print("\n" + "="*70)
    print("STATIC GESTURE MODEL TRAINING")
    print("="*70)

    # Initialize trainer
    trainer = StaticModelTrainer()

    # Load data
    features, labels, label_mapping = trainer.load_data()

    # Split data
    trainer.split_data(features, labels)

    # Create model
    num_classes = len(np.unique(labels))
    trainer.create_model(num_classes)

    # Train model
    trainer.train(epochs=50, batch_size=32)

    # Evaluate model
    test_loss, test_accuracy = trainer.evaluate()

    # Save model
    trainer.save_model("static_model_final.h5")

    # Plot training history
    from src.utils.config import EvaluationConfig
    plot_path = EvaluationConfig.PLOTS_DIR / "static_training_history.png"
    EvaluationConfig.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    trainer.plot_training_history(save_path=plot_path)

    # Print final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Model saved to: {MODELS_DIR / 'static_model_final.h5'}")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(TrainingConfig.RANDOM_SEED)

    # Run training
    main()
