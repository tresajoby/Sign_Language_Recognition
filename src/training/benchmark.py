import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
import numpy as np

from src.models.static_model import StaticModel
from src.models.dynamic_model import DynamicModel
from src.utils.config import get_model_path, DynamicModelConfig
from src.data_collection.hand_detector import HandDetector
from src.preprocessing.feature_extractor import FeatureExtractor


class RuntimeBenchmark:

    def __init__(self):
        static_path = get_model_path('static')
        dynamic_path = get_model_path('dynamic')

        self.static_model = None
        self.dynamic_model = None

        if static_path.exists():
            try:
                self.static_model = StaticModel().load(static_path)
                print(f"Loaded static model from {static_path}")
            except Exception as e:
                print(f"Could not load static model: {e}")

        if dynamic_path.exists():
            try:
                self.dynamic_model = DynamicModel().load(dynamic_path)
                print(f"Loaded dynamic model from {dynamic_path}")
            except Exception as e:
                print(f"Could not load dynamic model: {e}")

        self.hand_detector = HandDetector()
        self.feature_extractor = FeatureExtractor()

    def benchmark_inference_speed(self, n_trials=100):
        results = {}

        static_input = np.random.rand(1, 63).astype(np.float32)
        dynamic_input = np.random.rand(1, DynamicModelConfig.SEQUENCE_LENGTH, 63).astype(np.float32)

        if self.static_model is not None:
            # Warm-up
            for _ in range(5):
                self.static_model.predict(static_input, verbose=0)

            times = []
            for _ in range(n_trials):
                t0 = time.perf_counter()
                self.static_model.predict(static_input, verbose=0)
                times.append((time.perf_counter() - t0) * 1000)

            mean_ms = float(np.mean(times))
            std_ms = float(np.std(times))
            results['static_mean_ms'] = mean_ms
            results['static_std_ms'] = std_ms
            results['static_fps'] = 1000.0 / mean_ms if mean_ms > 0 else 0.0
        else:
            results['static_mean_ms'] = None
            results['static_std_ms'] = None
            results['static_fps'] = None

        if self.dynamic_model is not None:
            for _ in range(5):
                self.dynamic_model.predict(dynamic_input, verbose=0)

            times = []
            for _ in range(n_trials):
                t0 = time.perf_counter()
                self.dynamic_model.predict(dynamic_input, verbose=0)
                times.append((time.perf_counter() - t0) * 1000)

            mean_ms = float(np.mean(times))
            std_ms = float(np.std(times))
            results['dynamic_mean_ms'] = mean_ms
            results['dynamic_std_ms'] = std_ms
            results['dynamic_fps'] = 1000.0 / mean_ms if mean_ms > 0 else 0.0
        else:
            results['dynamic_mean_ms'] = None
            results['dynamic_std_ms'] = None
            results['dynamic_fps'] = None

        return results

    def benchmark_mediapipe(self, n_frames=100):
        frames = [
            np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            for _ in range(n_frames)
        ]

        times = []
        for frame in frames:
            t0 = time.perf_counter()
            self.hand_detector.detect(frame)
            times.append((time.perf_counter() - t0) * 1000)

        mean_ms = float(np.mean(times))
        std_ms = float(np.std(times))
        fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

        return {'mean_ms': mean_ms, 'std_ms': std_ms, 'fps': fps}

    def print_report(self, inference_results, mediapipe_results):
        print("\n" + "=" * 60)
        print("RUNTIME BENCHMARK REPORT")
        print("=" * 60)

        print(f"\n{'Component':<30} {'Mean (ms)':>10} {'Std (ms)':>10} {'FPS':>8}")
        print("-" * 62)

        def fmt(val):
            return f"{val:>10.2f}" if val is not None else f"{'N/A':>10}"

        s_mean = inference_results.get('static_mean_ms')
        s_std = inference_results.get('static_std_ms')
        s_fps = inference_results.get('static_fps')
        print(f"{'Static Model Inference':<30}{fmt(s_mean)}{fmt(s_std)}{fmt(s_fps)}")

        d_mean = inference_results.get('dynamic_mean_ms')
        d_std = inference_results.get('dynamic_std_ms')
        d_fps = inference_results.get('dynamic_fps')
        print(f"{'Dynamic Model Inference':<30}{fmt(d_mean)}{fmt(d_std)}{fmt(d_fps)}")

        mp = mediapipe_results
        print(f"{'MediaPipe Detection':<30}{mp['mean_ms']:>10.2f}{mp['std_ms']:>10.2f}{mp['fps']:>8.2f}")

        print("-" * 62)
        print()

    def run(self):
        print("Benchmarking inference speed...")
        inference_results = self.benchmark_inference_speed(n_trials=100)

        print("Benchmarking MediaPipe detection speed...")
        mediapipe_results = self.benchmark_mediapipe(n_frames=100)

        self.print_report(inference_results, mediapipe_results)
        return inference_results, mediapipe_results


def main():
    RuntimeBenchmark().run()


if __name__ == "__main__":
    main()
