import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

from src.utils.config import EvaluationConfig


class Evaluator:

    def __init__(self, model_type, label_encoder):
        self.model_type = model_type
        self.label_encoder = label_encoder
        EvaluationConfig.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    def evaluate(self, model, X_test, y_test_onehot):
        y_true = np.argmax(y_test_onehot, axis=1)
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        class_names = [self.label_encoder[i] for i in sorted(self.label_encoder)]

        acc = accuracy_score(y_true, y_pred)
        prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        print(f"\n[Evaluator] {self.model_type} model results:")
        print(f"  Accuracy:           {acc:.4f}")
        print(f"  Precision (macro):  {prec_macro:.4f}")
        print(f"  Recall    (macro):  {rec_macro:.4f}")
        print(f"  F1        (macro):  {f1_macro:.4f}")
        print(f"  Precision (weighted): {prec_weighted:.4f}")
        print(f"  Recall    (weighted): {rec_weighted:.4f}")
        print(f"  F1        (weighted): {f1_weighted:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

        self._save_confusion_matrix(y_true, y_pred, class_names)

        return {
            'accuracy': acc,
            'precision_macro': prec_macro,
            'recall_macro': rec_macro,
            'f1_macro': f1_macro,
            'precision_weighted': prec_weighted,
            'recall_weighted': rec_weighted,
            'f1_weighted': f1_weighted,
        }

    def _save_confusion_matrix(self, y_true, y_pred, class_names):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names))))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
        ax.set_title(f'Confusion Matrix — {self.model_type}')
        plt.tight_layout()
        save_path = EvaluationConfig.PLOTS_DIR / f'confusion_matrix_{self.model_type}.png'
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[Evaluator] Confusion matrix saved -> {save_path}")

    def plot_training_history(self, history):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(history.history['loss'], label='train')
        if 'val_loss' in history.history:
            axes[0].plot(history.history['val_loss'], label='val')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()

        axes[1].plot(history.history['accuracy'], label='train')
        if 'val_accuracy' in history.history:
            axes[1].plot(history.history['val_accuracy'], label='val')
        axes[1].set_title('Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()

        plt.suptitle(f'Training History — {self.model_type}')
        plt.tight_layout()
        save_path = EvaluationConfig.PLOTS_DIR / f'training_history_{self.model_type}.png'
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[Evaluator] Training history saved -> {save_path}")
