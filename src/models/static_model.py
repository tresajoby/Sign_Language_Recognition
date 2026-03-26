from tensorflow import keras
from tensorflow.keras import layers
from src.utils.config import StaticModelConfig, TrainingConfig


class StaticModel:

    def build(self):
        cfg = StaticModelConfig
        model = keras.Sequential([
            layers.Input(shape=(cfg.INPUT_DIM,)),
            layers.Dense(cfg.HIDDEN_LAYERS[0], activation='relu'),
            layers.Dropout(cfg.DROPOUT_RATE),
            layers.Dense(cfg.HIDDEN_LAYERS[1], activation='relu'),
            layers.Dropout(cfg.DROPOUT_RATE),
            layers.Dense(cfg.HIDDEN_LAYERS[2], activation='relu'),
            layers.Dropout(cfg.DROPOUT_RATE),
            layers.Dense(cfg.NUM_CLASSES, activation='softmax'),
        ], name='static_mlp')
        return model

    def compile(self, model):
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=TrainingConfig.LEARNING_RATE),
            loss=TrainingConfig.LOSS,
            metrics=TrainingConfig.METRICS,
        )

    def summary(self, model):
        model.summary()

    def save(self, model, path):
        model.save(str(path))

    def load(self, path):
        return keras.models.load_model(str(path))
