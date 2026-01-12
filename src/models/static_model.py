"""
Static Gesture MLP Model

This module implements a Multi-Layer Perceptron (MLP) for static gesture classification.
Static gestures are single-frame hand configurations (e.g., ASL letters A-Z, numbers 0-9).

Academic Justification:
- MLP is suitable because spatial relationships are already encoded in normalized landmarks
- No need for convolutional layers (CNNs) since we have structured feature vectors
- Fully connected layers learn non-linear decision boundaries in 63-dimensional space
- Dropout regularization prevents overfitting on limited data

Architecture:
    Input (63) → Dense(128) → Dropout(0.3) → Dense(64) → Dropout(0.3) →
    Dense(32) → Dropout(0.3) → Output(36)

References:
- Goodfellow et al., "Deep Learning" (2016), Chapter 6: Deep Feedforward Networks
- Rumelhart et al., "Learning representations by back-propagating errors" (1986)
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
from src.utils.config import StaticModelConfig, TrainingConfig


class StaticGestureModel:
    """
    Multi-Layer Perceptron for static gesture classification.

    This model takes a 63-dimensional feature vector (21 landmarks × 3 coords)
    and classifies it into one of 36 gesture classes (A-Z, 0-9).

    Attributes:
        model: Keras Sequential model
        history: Training history
        input_dim: Input feature dimension (default: 63)
        num_classes: Number of output classes (default: 36)
    """

    def __init__(
        self,
        input_dim: int = StaticModelConfig.INPUT_DIM,
        num_classes: int = StaticModelConfig.NUM_CLASSES,
        hidden_layers: List[int] = None,
        dropout_rate: float = StaticModelConfig.DROPOUT_RATE,
        activation: str = StaticModelConfig.ACTIVATION
    ):
        """
        Initialize the Static Gesture MLP model.

        Args:
            input_dim: Input feature dimension (63 for hand landmarks)
            num_classes: Number of gesture classes to classify
            hidden_layers: List of hidden layer sizes (default: [128, 64, 32])
            dropout_rate: Dropout rate for regularization
            activation: Activation function for hidden layers

        Academic Note:
        - ReLU activation prevents vanishing gradients
        - Dropout randomly drops units during training to prevent co-adaptation
        - Gradual layer size reduction (128→64→32) creates hierarchical features
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers or StaticModelConfig.HIDDEN_LAYERS
        self.dropout_rate = dropout_rate
        self.activation = activation

        self.model = self._build_model()
        self.history = None

    def _build_model(self) -> keras.Model:
        """
        Build the MLP architecture.

        Architecture Design Rationale:
        1. Input Layer (63): Normalized hand landmarks
        2. Hidden Layers: Progressive dimensionality reduction
           - Layer 1 (128): Learns low-level hand configurations
           - Layer 2 (64): Learns mid-level gesture components
           - Layer 3 (32): Learns high-level gesture representations
        3. Dropout: Regularization to prevent overfitting
        4. Output Layer (36): Softmax for multi-class classification

        Returns:
            Compiled Keras model
        """
        model = models.Sequential(name='Static_Gesture_MLP')

        # Input layer
        model.add(layers.Input(shape=(self.input_dim,), name='input_landmarks'))

        # Hidden layers with dropout
        for i, units in enumerate(self.hidden_layers):
            model.add(layers.Dense(
                units,
                activation=self.activation,
                kernel_initializer='he_normal',  # Good for ReLU
                name=f'dense_{i+1}'
            ))
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

        Loss Function:
        - Categorical Crossentropy: Standard for multi-class classification
        - Formula: L = -Σ yᵢ log(ŷᵢ)
          where yᵢ is true label (one-hot), ŷᵢ is predicted probability

        Optimizer:
        - Adam: Adaptive learning rate, combines momentum and RMSprop
        - Default lr=0.001 is standard for most vision tasks
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
            X_train: Training features (N, 63)
            y_train: Training labels (N, num_classes) - one-hot encoded
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch gradient descent
            callbacks: List of Keras callbacks
            verbose: Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch)

        Returns:
            Training history

        Training Process:
        1. Mini-batch gradient descent
        2. Forward pass: compute predictions
        3. Compute loss
        4. Backward pass: compute gradients via backpropagation
        5. Update weights: θ = θ - α∇L(θ)
        """
        print("\n" + "="*70)
        print("TRAINING STATIC GESTURE MODEL")
        print("="*70)
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
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
        """
        Get default training callbacks.

        Callbacks:
        1. EarlyStopping: Stop training if validation loss doesn't improve
        2. ModelCheckpoint: Save best model based on validation accuracy
        3. ReduceLROnPlateau: Reduce learning rate when validation loss plateaus

        Returns:
            List of Keras callbacks
        """
        from src.utils.config import MODELS_DIR

        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=TrainingConfig.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),

            # Save best model
            ModelCheckpoint(
                filepath=str(MODELS_DIR / 'static_model_best.h5'),
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
            X_test: Test features (N, 63)
            y_test: Test labels (N, num_classes) - one-hot encoded
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
        Make predictions on new data.

        Args:
            X: Input features (N, 63)
            return_probabilities: If True, return class probabilities
                                 If False, return class indices

        Returns:
            Class predictions or probability distributions
        """
        predictions = self.model.predict(X, verbose=0)

        if return_probabilities:
            return predictions  # (N, num_classes)
        else:
            return np.argmax(predictions, axis=1)  # (N,)

    def predict_single(
        self,
        features: np.ndarray,
        return_confidence: bool = True
    ) -> Tuple[int, float]:
        """
        Predict a single gesture.

        Args:
            features: Single feature vector (63,)
            return_confidence: If True, return confidence score

        Returns:
            Tuple of (predicted_class, confidence)
        """
        # Ensure correct shape
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Get prediction
        probs = self.model.predict(features, verbose=0)[0]
        predicted_class = np.argmax(probs)
        confidence = probs[predicted_class]

        if return_confidence:
            return predicted_class, confidence
        else:
            return predicted_class

    def save(self, filepath: str):
        """
        Save model to disk.

        Args:
            filepath: Path to save model (e.g., 'models/static_model.h5')
        """
        self.model.save(filepath)
        print(f"[INFO] Model saved to {filepath}")

    def load(self, filepath: str):
        """
        Load model from disk.

        Args:
            filepath: Path to saved model
        """
        self.model = keras.models.load_model(filepath)
        print(f"[INFO] Model loaded from {filepath}")

    def get_layer_outputs(self, X: np.ndarray, layer_name: str) -> np.ndarray:
        """
        Get intermediate layer outputs (for visualization/analysis).

        Args:
            X: Input features
            layer_name: Name of layer to extract

        Returns:
            Layer activations
        """
        intermediate_model = keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(layer_name).output
        )
        return intermediate_model.predict(X, verbose=0)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_static_model(
    num_classes: int = 36,
    compile: bool = True
) -> StaticGestureModel:
    """
    Factory function to create and compile a static gesture model.

    Args:
        num_classes: Number of gesture classes
        compile: Whether to compile the model

    Returns:
        StaticGestureModel instance
    """
    model = StaticGestureModel(num_classes=num_classes)

    if compile:
        model.compile_model()

    return model


def test_static_model():
    """Test the static model with synthetic data."""
    print("Testing Static Gesture Model...")
    print("="*70)

    # Create model
    model = create_static_model(num_classes=36)

    # Print architecture
    print("\nModel Architecture:")
    model.summary()

    # Create synthetic data
    print("\nGenerating synthetic data...")
    X_train = np.random.randn(1000, 63).astype(np.float32)
    y_train = keras.utils.to_categorical(np.random.randint(0, 36, 1000), 36)

    X_val = np.random.randn(200, 63).astype(np.float32)
    y_val = keras.utils.to_categorical(np.random.randint(0, 36, 200), 36)

    # Train for a few epochs
    print("\nTraining on synthetic data (quick test)...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=3,
        batch_size=32,
        verbose=1
    )

    # Test prediction
    print("\nTesting prediction...")
    test_sample = np.random.randn(1, 63).astype(np.float32)
    pred_class, confidence = model.predict_single(test_sample)
    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.2%}")

    print("\n" + "="*70)
    print("Static model test complete!")
    print("="*70)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(TrainingConfig.RANDOM_SEED)
    tf.random.set_seed(TrainingConfig.RANDOM_SEED)

    # Run test
    test_static_model()
