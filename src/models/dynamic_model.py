"""
Dynamic Gesture BiLSTM Model

This module implements a Bidirectional LSTM for dynamic gesture classification.
Dynamic gestures are temporal sequences of hand movements (e.g., "hello", "thanks").

Academic Justification:
- LSTM captures temporal dependencies in gesture sequences
- Bidirectional processing: learns from both past and future context
- Solves vanishing gradient problem in standard RNNs
- Well-suited for variable-length sequential patterns

Architecture:
    Input (30, 63) → BiLSTM(64) → Dropout(0.3) → BiLSTM(32) →
    Dropout(0.3) → Dense(num_classes)

References:
- Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997)
- Schuster & Paliwal, "Bidirectional Recurrent Neural Networks" (1997)
- Graves et al., "Speech Recognition with Deep Recurrent Neural Networks" (2013)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from typing import Tuple, List
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import DynamicModelConfig, TrainingConfig


class DynamicGestureModel:
    """
    Bidirectional LSTM for dynamic gesture classification.

    This model takes a sequence of feature vectors (30 frames × 63 features)
    and classifies it into one of the dynamic gesture classes.

    LSTM Architecture Explanation:
    - Each LSTM cell has a memory cell and three gates:
      * Forget gate: Decides what to remove from memory
      * Input gate: Decides what new information to store
      * Output gate: Decides what to output
    - Bidirectional: Processes sequence forwards AND backwards
    - This captures both "what came before" and "what comes after"

    Attributes:
        model: Keras Sequential model
        history: Training history
        sequence_length: Length of input sequences (default: 30)
        feature_dim: Feature dimension per frame (default: 63)
        num_classes: Number of output classes
    """

    def __init__(
        self,
        sequence_length: int = DynamicModelConfig.SEQUENCE_LENGTH,
        feature_dim: int = DynamicModelConfig.FEATURE_DIM,
        num_classes: int = DynamicModelConfig.NUM_CLASSES,
        lstm_units: List[int] = None,
        bidirectional: bool = DynamicModelConfig.BIDIRECTIONAL,
        dropout_rate: float = DynamicModelConfig.DROPOUT_RATE,
        recurrent_dropout: float = DynamicModelConfig.RECURRENT_DROPOUT
    ):
        """
        Initialize the Dynamic Gesture BiLSTM model.

        Args:
            sequence_length: Number of frames in sequence (30)
            feature_dim: Feature dimension per frame (63)
            num_classes: Number of gesture classes to classify
            lstm_units: List of LSTM layer sizes (default: [64, 32])
            bidirectional: Whether to use bidirectional LSTM
            dropout_rate: Dropout rate after LSTM layers
            recurrent_dropout: Dropout rate within LSTM cells

        Academic Note:
        - Bidirectional LSTM doubles the number of parameters but improves accuracy
        - Recurrent dropout helps prevent overfitting in LSTM cells
        - Layer stacking (64→32) creates hierarchical temporal features
        """
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.lstm_units = lstm_units or DynamicModelConfig.LSTM_UNITS
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout

        self.model = self._build_model()
        self.history = None

    def _build_model(self) -> keras.Model:
        """
        Build the BiLSTM architecture.

        Architecture Design Rationale:
        1. Input Layer (30, 63): Sequence of 30 frames, each with 63 features
        2. BiLSTM Layers: Process sequence in both directions
           - Forward pass: Learns patterns from frame 0 → 29
           - Backward pass: Learns patterns from frame 29 → 0
           - Combined: Captures full temporal context
        3. Dropout: Regularization
        4. Dense Output: Classification layer

        LSTM Cell Mathematics:
        For time step t:
        - Forget gate: fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)
        - Input gate: iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)
        - Output gate: oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)
        - Cell state: Cₜ = fₜ * Cₜ₋₁ + iₜ * tanh(Wc·[hₜ₋₁, xₜ] + bc)
        - Hidden state: hₜ = oₜ * tanh(Cₜ)

        Returns:
            Compiled Keras model
        """
        model = models.Sequential(name='Dynamic_Gesture_BiLSTM')

        # Input layer
        model.add(layers.Input(
            shape=(self.sequence_length, self.feature_dim),
            name='input_sequence'
        ))

        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            # Return sequences for all layers except the last
            return_sequences = (i < len(self.lstm_units) - 1)

            if self.bidirectional:
                model.add(layers.Bidirectional(
                    layers.LSTM(
                        units,
                        return_sequences=return_sequences,
                        recurrent_dropout=self.recurrent_dropout,
                        name=f'lstm_{i+1}'
                    ),
                    name=f'bidirectional_{i+1}'
                ))
            else:
                model.add(layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    recurrent_dropout=self.recurrent_dropout,
                    name=f'lstm_{i+1}'
                ))

            # Dropout after each LSTM layer
            model.add(layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}'))

        # Output layer
        model.add(layers.Dense(
            self.num_classes,
            activation='softmax',
            name='output'
        ))

        return model

    def compile_model(
        self,
        learning_rate: float = TrainingConfig.LEARNING_RATE,
        optimizer: str = TrainingConfig.OPTIMIZER
    ):
        """
        Compile the model with optimizer and loss function.

        Args:
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer name ('adam', 'sgd', 'rmsprop')

        Note:
        - Adam optimizer with default lr=0.001 works well for RNNs
        - Categorical crossentropy for multi-class classification
        """
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"[INFO] Model compiled with {optimizer} optimizer (lr={learning_rate})")

    def summary(self):
        """Print model architecture summary."""
        return self.model.summary()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = TrainingConfig.EPOCHS,
        batch_size: int = TrainingConfig.BATCH_SIZE,
        callbacks: List = None,
        verbose: int = 1
    ):
        """
        Train the model.

        Args:
            X_train: Training sequences (N, 30, 63)
            y_train: Training labels (N, num_classes) - one-hot encoded
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            callbacks: List of Keras callbacks
            verbose: Verbosity mode

        Returns:
            Training history

        Training Process for RNNs:
        1. Forward pass through time (frame 0 → 29)
        2. Compute loss at final time step
        3. Backpropagation Through Time (BPTT)
        4. Update weights with optimizer
        """
        print("\n" + "="*70)
        print("TRAINING DYNAMIC GESTURE MODEL")
        print("="*70)
        print(f"Training sequences: {len(X_train)}")
        print(f"Validation sequences: {len(X_val)}")
        print(f"Sequence length: {self.sequence_length} frames")
        print(f"Feature dim: {self.feature_dim}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")
        print("="*70 + "\n")

        # Default callbacks if none provided
        if callbacks is None:
            callbacks = self._get_default_callbacks()

        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)

        return self.history

    def _get_default_callbacks(self) -> List:
        """Get default training callbacks."""
        from src.utils.config import MODELS_DIR

        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=TrainingConfig.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),

            # Save best model
            ModelCheckpoint(
                filepath=str(MODELS_DIR / 'dynamic_model_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),

            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        return callbacks

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: int = 1
    ) -> Tuple[float, float]:
        """
        Evaluate model on test data.

        Args:
            X_test: Test sequences (N, 30, 63)
            y_test: Test labels (N, num_classes)
            verbose: Verbosity mode

        Returns:
            Tuple of (loss, accuracy)
        """
        results = self.model.evaluate(X_test, y_test, verbose=verbose)
        loss, accuracy = results[0], results[1]

        print("\n" + "="*70)
        print("TEST SET EVALUATION")
        print("="*70)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("="*70 + "\n")

        return loss, accuracy

    def predict(self, X: np.ndarray, return_probabilities: bool = False):
        """
        Make predictions on new sequences.

        Args:
            X: Input sequences (N, 30, 63)
            return_probabilities: If True, return class probabilities

        Returns:
            Class predictions or probability distributions
        """
        predictions = self.model.predict(X, verbose=0)

        if return_probabilities:
            return predictions
        else:
            return np.argmax(predictions, axis=1)

    def predict_single(
        self,
        sequence: np.ndarray,
        return_confidence: bool = True
    ) -> Tuple[int, float]:
        """
        Predict a single gesture sequence.

        Args:
            sequence: Single sequence (30, 63)
            return_confidence: If True, return confidence score

        Returns:
            Tuple of (predicted_class, confidence)
        """
        # Ensure correct shape
        if sequence.ndim == 2:
            sequence = sequence.reshape(1, self.sequence_length, self.feature_dim)

        # Get prediction
        probs = self.model.predict(sequence, verbose=0)[0]
        predicted_class = np.argmax(probs)
        confidence = probs[predicted_class]

        if return_confidence:
            return predicted_class, confidence
        else:
            return predicted_class

    def save(self, filepath: str):
        """Save model to disk."""
        self.model.save(filepath)
        print(f"[INFO] Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model from disk."""
        self.model = keras.models.load_model(filepath)
        print(f"[INFO] Model loaded from {filepath}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_dynamic_model(
    num_classes: int = 10,
    compile: bool = True
) -> DynamicGestureModel:
    """
    Factory function to create and compile a dynamic gesture model.

    Args:
        num_classes: Number of gesture classes
        compile: Whether to compile the model

    Returns:
        DynamicGestureModel instance
    """
    model = DynamicGestureModel(num_classes=num_classes)

    if compile:
        model.compile_model()

    return model


def test_dynamic_model():
    """Test the dynamic model with synthetic data."""
    print("Testing Dynamic Gesture Model...")
    print("="*70)

    # Create model
    model = create_dynamic_model(num_classes=10)

    # Print architecture
    print("\nModel Architecture:")
    model.summary()

    # Create synthetic data
    print("\nGenerating synthetic data...")
    X_train = np.random.randn(500, 30, 63).astype(np.float32)
    y_train = keras.utils.to_categorical(np.random.randint(0, 10, 500), 10)

    X_val = np.random.randn(100, 30, 63).astype(np.float32)
    y_val = keras.utils.to_categorical(np.random.randint(0, 10, 100), 10)

    # Train for a few epochs
    print("\nTraining on synthetic data (quick test)...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=3,
        batch_size=16,
        verbose=1
    )

    # Test prediction
    print("\nTesting prediction...")
    test_sample = np.random.randn(1, 30, 63).astype(np.float32)
    pred_class, confidence = model.predict_single(test_sample)
    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.2%}")

    print("\n" + "="*70)
    print("Dynamic model test complete!")
    print("="*70)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(TrainingConfig.RANDOM_SEED)
    tf.random.set_seed(TrainingConfig.RANDOM_SEED)

    # Run test
    test_dynamic_model()
