import numpy as np

from src.utils.config import InferenceConfig, get_model_path
from src.models.static_model import StaticModel
from src.models.dynamic_model import DynamicModel
from src.preprocessing.feature_extractor import FeatureExtractor
from src.preprocessing.dataset_manager import DatasetManager


class Predictor:

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.static_model = None
        self.dynamic_model = None
        self.static_label_encoder = {}
        self.dynamic_label_encoder = {}
        self.static_ready = False
        self.dynamic_ready = False

        static_path = get_model_path('static')
        if static_path.exists():
            try:
                self.static_model = StaticModel().load(static_path)
                self.static_ready = True
            except Exception as e:
                print(f"Failed to load static model: {e}")

        dynamic_path = get_model_path('dynamic')
        if dynamic_path.exists():
            try:
                self.dynamic_model = DynamicModel().load(dynamic_path)
                self.dynamic_ready = True
            except Exception as e:
                print(f"Failed to load dynamic model: {e}")

        dm = DatasetManager()
        try:
            *_, self.static_label_encoder = dm.load_processed('static')
        except Exception:
            self.static_label_encoder = {}

        try:
            *_, self.dynamic_label_encoder = dm.load_processed('dynamic')
        except Exception:
            self.dynamic_label_encoder = {}

    def predict_static(self, landmark_array_63):
        if not self.static_ready:
            return None
        features = self.feature_extractor.extract(landmark_array_63)
        x = features.reshape(1, -1)
        probs = self.static_model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        if confidence < InferenceConfig.CONFIDENCE_THRESHOLD:
            return None
        class_name = self.static_label_encoder.get(idx, str(idx))
        return class_name, confidence

    def predict_dynamic(self, sequence):
        if not self.dynamic_ready:
            return None
        x = np.array(sequence).reshape(1, sequence.shape[0], sequence.shape[1])
        probs = self.dynamic_model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        if confidence < InferenceConfig.CONFIDENCE_THRESHOLD:
            return None
        class_name = self.dynamic_label_encoder.get(idx, str(idx))
        return class_name, confidence
