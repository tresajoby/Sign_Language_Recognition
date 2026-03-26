import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.models.static_model import StaticModel
from src.models.dynamic_model import DynamicModel
from src.preprocessing.feature_extractor import FeatureExtractor
from src.utils.config import StaticModelConfig, DynamicModelConfig


class TestStaticModel:

    def test_build_returns_model(self):
        model = StaticModel().build()
        assert model is not None

    def test_output_shape(self):
        model = StaticModel().build()
        batch_size = 4
        x = np.random.rand(batch_size, StaticModelConfig.INPUT_DIM).astype(np.float32)
        output = model.predict(x, verbose=0)
        assert output.shape == (batch_size, StaticModelConfig.NUM_CLASSES)

    def test_predict_probabilities_sum_to_one(self):
        model = StaticModel().build()
        x = np.random.rand(3, StaticModelConfig.INPUT_DIM).astype(np.float32)
        probs = model.predict(x, verbose=0)
        sums = probs.sum(axis=1)
        np.testing.assert_allclose(sums, np.ones(3), atol=1e-5)


class TestDynamicModel:

    def test_build_returns_model(self):
        model = DynamicModel().build()
        assert model is not None

    def test_output_shape(self):
        model = DynamicModel().build()
        batch_size = 4
        x = np.random.rand(
            batch_size,
            DynamicModelConfig.SEQUENCE_LENGTH,
            DynamicModelConfig.FEATURE_DIM,
        ).astype(np.float32)
        output = model.predict(x, verbose=0)
        assert output.shape == (batch_size, DynamicModelConfig.NUM_CLASSES)

    def test_predict_probabilities_sum_to_one(self):
        model = DynamicModel().build()
        x = np.random.rand(
            3,
            DynamicModelConfig.SEQUENCE_LENGTH,
            DynamicModelConfig.FEATURE_DIM,
        ).astype(np.float32)
        probs = model.predict(x, verbose=0)
        sums = probs.sum(axis=1)
        np.testing.assert_allclose(sums, np.ones(3), atol=1e-5)


class TestFeatureExtractor:

    def test_extract_output_shape(self):
        extractor = FeatureExtractor()
        raw = np.random.rand(63).astype(np.float32)
        result = extractor.extract(raw)
        assert result.shape == (63,)

    def test_wrist_normalization(self):
        extractor = FeatureExtractor()
        raw = np.random.rand(63).astype(np.float32)
        result = extractor.extract(raw)
        # After wrist normalization, landmark 0 coords (indices 0,1,2) should be 0
        np.testing.assert_allclose(result[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(result[1], 0.0, atol=1e-6)
        np.testing.assert_allclose(result[2], 0.0, atol=1e-6)

    def test_extract_from_sequence_shape(self):
        extractor = FeatureExtractor()
        seq_len = 30
        sequence = np.random.rand(seq_len, 63).astype(np.float32)
        result = extractor.extract_from_sequence(sequence)
        assert result.shape == (seq_len, 63)


class TestPredictor:

    def test_instantiates_without_crash(self):
        from src.inference.predictor import Predictor
        try:
            predictor = Predictor()
        except Exception as e:
            pytest.fail(f"Predictor() raised an exception: {e}")

    def test_predict_static_returns_none_when_no_model(self):
        from src.inference.predictor import Predictor
        predictor = Predictor()
        if not predictor.static_ready:
            result = predictor.predict_static(np.zeros(63, dtype=np.float32))
            assert result is None

    def test_predict_dynamic_returns_none_when_no_model(self):
        from src.inference.predictor import Predictor
        predictor = Predictor()
        if not predictor.dynamic_ready:
            seq = np.zeros((30, 63), dtype=np.float32)
            result = predictor.predict_dynamic(seq)
            assert result is None
