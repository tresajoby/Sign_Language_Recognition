"""
Preprocessor Module

Orchestrates the full preprocessing pipeline for both static and dynamic
gesture datasets:

    raw .npy files
        -> load with DatasetManager
        -> normalize with FeatureExtractor
        -> split into train/val/test
        -> save to data/processed/

Usage:
    preprocessor = Preprocessor()
    preprocessor.run_static_pipeline()
    preprocessor.run_dynamic_pipeline()
"""

from src.preprocessing.feature_extractor import FeatureExtractor
from src.preprocessing.dataset_manager import DatasetManager


class Preprocessor:
    """
    High-level pipeline that ties FeatureExtractor and DatasetManager together.
    """

    def __init__(self):
        self.extractor = FeatureExtractor()
        self.manager = DatasetManager()

    def run_static_pipeline(self):
        """
        End-to-end pipeline for static gesture data.

        Steps:
            1. Load raw (N, 63) arrays from data/raw/static/
            2. Normalize every sample with FeatureExtractor
            3. Split into train/val/test
            4. Save processed splits and label encoder to data/processed/
        """
        print("=" * 55)
        print("STATIC PIPELINE")
        print("=" * 55)

        # Step 1: Load raw data
        print("\nStep 1/4 — Loading raw static data...")
        X, y, label_encoder = self.manager.load_static_dataset()

        # Step 2: Normalize features
        print("\nStep 2/4 — Extracting and normalizing features...")
        X_normalized = self.extractor.extract_from_sequence(X)
        print(f"  Normalized shape: {X_normalized.shape}")

        # Step 3: Split
        print("\nStep 3/4 — Splitting dataset...")
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.manager.split_dataset(X_normalized, y)

        # Step 4: Save
        print("\nStep 4/4 — Saving processed data...")
        self.manager.save_processed(X_train, X_val, X_test,
                                    y_train, y_val, y_test,
                                    dataset_type='static')
        self.manager.save_label_encoder(label_encoder, dataset_type='static')

        print("\nStatic pipeline complete.")
        print("=" * 55)

    def run_dynamic_pipeline(self):
        """
        End-to-end pipeline for dynamic gesture data.

        Steps:
            1. Load raw (N, seq_len, 63) arrays from data/raw/dynamic/
            2. Normalize every frame in every sequence
            3. Split into train/val/test
            4. Save processed splits and label encoder to data/processed/
        """
        print("=" * 55)
        print("DYNAMIC PIPELINE")
        print("=" * 55)

        # Step 1: Load raw data
        print("\nStep 1/4 — Loading raw dynamic data...")
        X, y, label_encoder = self.manager.load_dynamic_dataset()

        # Step 2: Normalize — reshape to (N*T, 63), normalize, reshape back
        print("\nStep 2/4 — Extracting and normalizing features...")
        N, T, F = X.shape
        X_flat = X.reshape(N * T, F)                        # (N*T, 63)
        X_flat_norm = self.extractor.extract_from_sequence(X_flat)
        X_normalized = X_flat_norm.reshape(N, T, F)         # (N, T, 63)
        print(f"  Normalized shape: {X_normalized.shape}")

        # Step 3: Split
        print("\nStep 3/4 — Splitting dataset...")
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.manager.split_dataset(X_normalized, y)

        # Step 4: Save
        print("\nStep 4/4 — Saving processed data...")
        self.manager.save_processed(X_train, X_val, X_test,
                                    y_train, y_val, y_test,
                                    dataset_type='dynamic')
        self.manager.save_label_encoder(label_encoder, dataset_type='dynamic')

        print("\nDynamic pipeline complete.")
        print("=" * 55)
