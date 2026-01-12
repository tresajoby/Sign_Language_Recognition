"""
Training Script for Dynamic Gesture Model

This script handles the complete training pipeline for the dynamic gesture BiLSTM.

Usage:
    python src/training/train_dynamic.py
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

from src.models.dynamic_model import create_dynamic_model
from src.utils.config import (
    TrainingConfig,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    LABELS_DIR,
    DataCollectionConfig
)


class DynamicModelTrainer:
    """Trainer class for dynamic gesture model."""

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
        """Load preprocessed sequences from disk."""
        print("\n" + "="*70)
        print("LOADING DATA")
        print("="*70)

        sequences_path = PROCESSED_DATA_DIR / "dynamic_sequences.npy"
        labels_path = PROCESSED_DATA_DIR / "dynamic_labels.npy"
        mapping_path = LABELS_DIR / "dynamic_label_mapping.json"

        if not sequences_path.exists():
            raise FileNotFoundError(
                f"Sequences file not found: {sequences_path}\n"
                "Please run data collection first:\n"
                "  python src/data_collection/collect_dynamic.py"
            )

        # Load data
        sequences = np.load(sequences_path)
        labels = np.load(labels_path)

        # Load label mapping
        if mapping_path.exists():
            with open(mapping_path, 'r') as f:
                label_mapping = {int(k): v for k, v in json.load(f).items()}
        else:
            label_mapping = {i: gesture
                           for i, gesture in enumerate(DataCollectionConfig.DYNAMIC_CLASSES)}

        print(f"Sequences shape: {sequences.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Number of classes: {len(np.unique(labels))}")
        print(f"Sequence length: {sequences.shape[1]} frames")
        print(f"Feature dimension: {sequences.shape[2]}")
        print(f"Label mapping: {label_mapping}")

        self.label_mapping = label_mapping

        return sequences, labels, label_mapping

    def split_data(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        train_split: float = TrainingConfig.TRAIN_SPLIT,
        val_split: float = TrainingConfig.VAL_SPLIT,
        test_split: float = TrainingConfig.TEST_SPLIT,
        random_seed: int = TrainingConfig.RANDOM_SEED
    ):
        """Split data into train/validation/test sets."""
        print("\n" + "="*70)
        print("SPLITTING DATA")
        print("="*70)

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            sequences,
            labels,
            test_size=test_split,
            random_state=random_seed,
            stratify=labels
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

        print(f"Train set: {X_train.shape[0]} sequences ({train_split*100:.1f}%)")
        print(f"Validation set: {X_val.shape[0]} sequences ({val_split*100:.1f}%)")
        print(f"Test set: {X_test.shape[0]} sequences ({test_split*100:.1f}%)")

        # Convert labels to one-hot
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
        """Create and compile the dynamic gesture model."""
        print("\n" + "="*70)
        print("CREATING MODEL")
        print("="*70)

        self.model = create_dynamic_model(num_classes=num_classes, compile=True)
        self.model.summary()

    def train(
        self,
        epochs: int = TrainingConfig.EPOCHS,
        batch_size: int = 16  # Smaller batch size for sequences
    ):
        """Train the model."""
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")

        self.history = self.model.train(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            epochs=epochs,
            batch_size=batch_size
        )

    def evaluate(self):
        """Evaluate the model on test set."""
        if self.model is None:
            raise ValueError("Model not trained yet.")

        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)

        return loss, accuracy

    def save_model(self, filename: str = "dynamic_model.h5"):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save.")

        filepath = MODELS_DIR / filename
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        self.model.save(str(filepath))
        print(f"\n[INFO] Model saved to {filepath}")

    def plot_training_history(self, save_path: str = None):
        """Plot training history."""
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
    """Main training workflow."""
    print("\n" + "="*70)
    print("DYNAMIC GESTURE MODEL TRAINING")
    print("="*70)

    # Initialize trainer
    trainer = DynamicModelTrainer()

    # Load data
    sequences, labels, label_mapping = trainer.load_data()

    # Split data
    trainer.split_data(sequences, labels)

    # Create model
    num_classes = len(np.unique(labels))
    trainer.create_model(num_classes)

    # Train model
    trainer.train(epochs=50, batch_size=16)

    # Evaluate model
    test_loss, test_accuracy = trainer.evaluate()

    # Save model
    trainer.save_model("dynamic_model_final.h5")

    # Plot training history
    from src.utils.config import EvaluationConfig
    plot_path = EvaluationConfig.PLOTS_DIR / "dynamic_training_history.png"
    EvaluationConfig.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    trainer.plot_training_history(save_path=plot_path)

    # Print final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Model saved to: {MODELS_DIR / 'dynamic_model_final.h5'}")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Set random seed
    np.random.seed(TrainingConfig.RANDOM_SEED)

    # Run training
    main()
