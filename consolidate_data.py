"""
Data Consolidation Script

Converts collected data from OneDrive subdirectory format into
the flat .npy format expected by DatasetManager.

Source format:
  static/{class}/landmarks_*.npy  -> shape (21, 3) per file
  dynamic/{class}/sequence_*.npy  -> shape (30, 21, 3) per file

Target format:
  data/raw/static/{class}.npy     -> shape (N, 63)
  data/raw/dynamic/{class}.npy    -> shape (N, 30, 63)
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

SOURCE_ROOT = Path("C:/Users/Adven/OneDrive/Documents/My files/Sign_Language_Recognition/data/raw")
TARGET_ROOT = Path("C:/Users/Adven/data/raw")


def consolidate_static():
    src = SOURCE_ROOT / "static"
    dst = TARGET_ROOT / "static"
    dst.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([d for d in src.iterdir() if d.is_dir()])
    print(f"Found {len(class_dirs)} static classes: {[d.name for d in class_dirs]}")

    for class_dir in class_dirs:
        files = sorted(class_dir.glob("landmarks_*.npy"))
        if not files:
            print(f"  [SKIP] {class_dir.name}: no landmark files")
            continue

        samples = []
        for f in files:
            arr = np.load(f)          # (21, 3)
            samples.append(arr.flatten())  # (63,)

        X = np.array(samples, dtype=np.float32)  # (N, 63)
        out_path = dst / f"{class_dir.name}.npy"
        np.save(out_path, X)
        print(f"  {class_dir.name}: {X.shape} -> {out_path.name}")

    print(f"Static consolidation complete.\n")


def consolidate_dynamic():
    src = SOURCE_ROOT / "dynamic"
    dst = TARGET_ROOT / "dynamic"
    dst.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([d for d in src.iterdir() if d.is_dir()])
    print(f"Found {len(class_dirs)} dynamic classes: {[d.name for d in class_dirs]}")

    for class_dir in class_dirs:
        files = sorted(class_dir.glob("sequence_*.npy"))
        if not files:
            files = sorted(class_dir.glob("*.npy"))
        if not files:
            print(f"  [SKIP] {class_dir.name}: no sequence files")
            continue

        sequences = []
        for f in files:
            arr = np.load(f)          # (30, 21, 3)
            sequences.append(arr.reshape(arr.shape[0], -1))  # (30, 63)

        X = np.array(sequences, dtype=np.float32)  # (N, 30, 63)
        out_path = dst / f"{class_dir.name}.npy"
        np.save(out_path, X)
        print(f"  {class_dir.name}: {X.shape} -> {out_path.name}")

    print(f"Dynamic consolidation complete.\n")


if __name__ == "__main__":
    print("=" * 60)
    print("DATA CONSOLIDATION")
    print("=" * 60)
    print(f"Source: {SOURCE_ROOT}")
    print(f"Target: {TARGET_ROOT}\n")

    print("--- STATIC ---")
    consolidate_static()

    print("--- DYNAMIC ---")
    consolidate_dynamic()

    print("Done. Run preprocessing next:")
    print("  python -m src.preprocessing.preprocessor")
