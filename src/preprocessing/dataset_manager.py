"""
Dataset Manager Module

Handles loading raw .npy landmark files, encoding class labels, splitting into
train/val/test sets, and persisting processed arrays to disk.

Expected raw file layout:
    data/raw/static/{class_name}.npy   shape: (N, 63)
    data/raw/dynamic/{class_name}.npy  shape: (N, seq_len, 63)

Processed outputs go to:
    data/processed/{dataset_type}_X_{split}.npy
    data/processed/{dataset_type}_y_{split}.npy
    data/processed/{dataset_type}_labels.npy
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, TrainingConfig


class DatasetManager:
    """
    Loads, splits, and persists ASL landmark datasets.

    label_encoder is a plain dict mapping integer index -> class name string,
    so it survives numpy save/load without pickle concerns.
    """

    def __init__(self):
        self.processed_dir = PROCESSED_DATA_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_raw(self, raw_dir):
        """
        Load all .npy files in raw_dir and build X, y arrays.

        Args:
            raw_dir: Path to the directory containing per-class .npy files.

        Returns:
            (X, y, label_encoder) — X is stacked samples, y is integer labels,
            label_encoder is a dict {int: class_name}.
        """
        npy_files = sorted(raw_dir.glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No .npy files found in {raw_dir}")

        X_parts = []
        y_parts = []
        class_names = []

        for class_idx, npy_path in enumerate(npy_files):
            class_name = npy_path.stem          # filename without extension
            data = np.load(npy_path)            # (N, ...) loaded array
            num_samples = data.shape[0]

            X_parts.append(data)
            y_parts.append(np.full(num_samples, class_idx, dtype=np.int32))
            class_names.append(class_name)

            print(f"  Loaded '{class_name}': {num_samples} samples")

        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        label_encoder = {idx: name for idx, name in enumerate(class_names)}

        return X, y, label_encoder

    # ------------------------------------------------------------------
    # Public loading methods
    # ------------------------------------------------------------------

    def load_static_dataset(self):
        """
        Load all static gesture samples.

        Returns:
            X: shape (N, 63)
            y: shape (N,) — integer class indices
            label_encoder: dict {int: class_name}
        """
        print("[DatasetManager] Loading static dataset...")
        raw_dir = RAW_DATA_DIR / "static"
        X, y, label_encoder = self._load_raw(raw_dir)
        print(f"  Total samples: {X.shape[0]} | Classes: {len(label_encoder)}")
        return X, y, label_encoder

    def load_dynamic_dataset(self):
        """
        Load all dynamic gesture sequences.

        Returns:
            X: shape (N, seq_len, 63)
            y: shape (N,) — integer class indices
            label_encoder: dict {int: class_name}
        """
        print("[DatasetManager] Loading dynamic dataset...")
        raw_dir = RAW_DATA_DIR / "dynamic"
        X, y, label_encoder = self._load_raw(raw_dir)
        print(f"  Total sequences: {X.shape[0]} | Classes: {len(label_encoder)}")
        return X, y, label_encoder

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def split_dataset(self, X, y):
        """
        Split X and y into train, validation, and test sets.

        Ratios are taken from TrainingConfig.  The split is stratified so
        class distribution is preserved across all three sets.

        Args:
            X: feature array (any shape, first dim = samples)
            y: integer label array of shape (N,)

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        seed = TrainingConfig.RANDOM_SEED
        test_ratio = TrainingConfig.TEST_SPLIT
        # val ratio relative to the train+val portion
        val_ratio = TrainingConfig.VAL_SPLIT / (TrainingConfig.TRAIN_SPLIT + TrainingConfig.VAL_SPLIT)

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=test_ratio,
            stratify=y,
            random_state=seed,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_ratio,
            stratify=y_trainval,
            random_state=seed,
        )

        print(f"[DatasetManager] Split -> train: {len(X_train)} | "
              f"val: {len(X_val)} | test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_processed(self, X_train, X_val, X_test,
                       y_train, y_val, y_test, dataset_type):
        """
        Save split arrays to data/processed/.

        Args:
            dataset_type: 'static' or 'dynamic' — used as a filename prefix.
        """
        prefix = self.processed_dir / dataset_type
        np.save(f"{prefix}_X_train.npy", X_train)
        np.save(f"{prefix}_X_val.npy",   X_val)
        np.save(f"{prefix}_X_test.npy",  X_test)
        np.save(f"{prefix}_y_train.npy", y_train)
        np.save(f"{prefix}_y_val.npy",   y_val)
        np.save(f"{prefix}_y_test.npy",  y_test)
        print(f"[DatasetManager] Processed '{dataset_type}' data saved to {self.processed_dir}")

    def save_label_encoder(self, label_encoder, dataset_type):
        """
        Persist the label encoder dict as a numpy object array.

        Saved as: data/processed/{dataset_type}_labels.npy
        Reload with: np.load(..., allow_pickle=True).item()
        """
        save_path = self.processed_dir / f"{dataset_type}_labels.npy"
        np.save(save_path, label_encoder)
        print(f"[DatasetManager] Label encoder saved -> {save_path}")

    def load_processed(self, dataset_type):
        """
        Load previously saved processed arrays from data/processed/.

        Args:
            dataset_type: 'static' or 'dynamic'.

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test, label_encoder)
        """
        prefix = self.processed_dir / dataset_type
        X_train = np.load(f"{prefix}_X_train.npy")
        X_val   = np.load(f"{prefix}_X_val.npy")
        X_test  = np.load(f"{prefix}_X_test.npy")
        y_train = np.load(f"{prefix}_y_train.npy")
        y_val   = np.load(f"{prefix}_y_val.npy")
        y_test  = np.load(f"{prefix}_y_test.npy")

        label_path = self.processed_dir / f"{dataset_type}_labels.npy"
        label_encoder = np.load(label_path, allow_pickle=True).item()

        print(f"[DatasetManager] Loaded processed '{dataset_type}' data from {self.processed_dir}")
        return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder
