from tensorflow import keras
from tensorflow.keras import layers
from src.utils.config import DynamicModelConfig, TrainingConfig


class DynamicModel:

    def build(self):
        cfg = DynamicModelConfig
        model = keras.Sequential([
            layers.Input(shape=(cfg.SEQUENCE_LENGTH, cfg.FEATURE_DIM)),
            layers.Bidirectional(
                layers.LSTM(cfg.LSTM_UNITS[0], return_sequences=True,
                            recurrent_dropout=cfg.RECURRENT_DROPOUT)
            ),
            layers.Dropout(cfg.DROPOUT_RATE),
            layers.Bidirectional(
                layers.LSTM(cfg.LSTM_UNITS[1], return_sequences=False,
                            recurrent_dropout=cfg.RECURRENT_DROPOUT)
            ),
            layers.Dropout(cfg.DROPOUT_RATE),
            layers.Dense(64, activation='relu'),
            layers.Dense(cfg.NUM_CLASSES, activation='softmax'),
        ], name='dynamic_bilstm')
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
