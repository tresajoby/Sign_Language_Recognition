import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from tensorflow.keras.utils import to_categorical

from src.preprocessing.dataset_manager import DatasetManager
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator
from src.utils.config import DynamicModelConfig


def main():
    dm = DatasetManager()

    try:
        X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = dm.load_processed('dynamic')
    except FileNotFoundError:
        print("[train_dynamic] No processed dynamic data found.")
        print("  Run the preprocessing pipeline first to generate:")
        print("  data/processed/dynamic_X_train.npy (and related files)")
        return

    num_classes = DynamicModelConfig.NUM_CLASSES
    y_train_oh = to_categorical(y_train, num_classes)
    y_val_oh   = to_categorical(y_val,   num_classes)
    y_test_oh  = to_categorical(y_test,  num_classes)

    trainer = Trainer('dynamic')
    history = trainer.train(X_train, y_train_oh, X_val, y_val_oh)

    evaluator = Evaluator('dynamic', label_encoder)
    evaluator.plot_training_history(history)

    metrics = evaluator.evaluate(trainer.model, X_test, y_test_oh)

    print("\n[train_dynamic] Final test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()
