import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.models.static_model import StaticModel
from src.models.dynamic_model import DynamicModel
from src.utils.config import TrainingConfig, get_model_path


class Trainer:

    def __init__(self, model_type):
        self.model_type = model_type
        self.model_path = get_model_path(model_type)

        if model_type == 'static':
            builder = StaticModel()
        elif model_type == 'dynamic':
            builder = DynamicModel()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.model = builder.build()
        builder.compile(self.model)
        self._builder = builder

    def train(self, X_train, y_train, X_val, y_val):
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=TrainingConfig.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
            ),
            ModelCheckpoint(
                filepath=str(self.model_path),
                monitor='val_loss',
                save_best_only=True,
            ),
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=TrainingConfig.EPOCHS,
            batch_size=TrainingConfig.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
        )
        return history

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return {'loss': loss, 'accuracy': accuracy}

    def predict(self, X):
        probs = self.model.predict(X, verbose=0)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        return self.model.predict(X, verbose=0)

    def save_model(self):
        self._builder.save(self.model, self.model_path)

    def load_model(self):
        self.model = self._builder.load(self.model_path)
