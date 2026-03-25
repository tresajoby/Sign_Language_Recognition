import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

from src.preprocessing.dataset_manager import DatasetManager
from src.models.static_model import StaticModel
from src.models.dynamic_model import DynamicModel
from src.utils.config import get_model_path, EvaluationConfig

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('ggplot')


class ModelEvaluator:

    def __init__(self, model_type):
        self.model_type = model_type
        EvaluationConfig.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        model_path = get_model_path(model_type)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if model_type == 'static':
            self.model = StaticModel().load(model_path)
        else:
            self.model = DynamicModel().load(model_path)

        dm = DatasetManager()
        _, _, X_test, _, _, y_test, label_encoder = dm.load_processed(model_type)
        self.X_test = X_test
        self.y_test = y_test.astype(int)
        self.label_encoder = label_encoder
        self.class_names = [label_encoder[i] for i in sorted(label_encoder)]

    def _get_predictions(self):
        raw = self.model.predict(self.X_test, verbose=0)
        return np.argmax(raw, axis=1)

    def compute_metrics(self):
        y_pred = self._get_predictions()
        y_true = self.y_test

        per_class_f1_arr = f1_score(y_true, y_pred, average=None,
                                    labels=sorted(self.label_encoder),
                                    zero_division=0)
        per_class_f1 = {
            self.label_encoder[i]: float(per_class_f1_arr[idx])
            for idx, i in enumerate(sorted(self.label_encoder))
        }

        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'precision_weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall_weighted': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            'per_class_f1': per_class_f1,
        }

    def print_classification_report(self):
        y_pred = self._get_predictions()
        print(classification_report(self.y_test, y_pred,
                                    target_names=self.class_names,
                                    zero_division=0))

    def plot_confusion_matrix(self, normalize=True):
        y_pred = self._get_predictions()
        cm = confusion_matrix(self.y_test, y_pred,
                              labels=sorted(self.label_encoder))

        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_plot = np.where(row_sums == 0, 0, cm / row_sums.astype(float))
            fmt = '.2f'
            vmax = 1.0
        else:
            cm_plot = cm
            fmt = 'd'
            vmax = cm.max()

        n = len(self.class_names)
        fig_size = max(10, n * 0.45)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

        sns.heatmap(
            cm_plot,
            annot=(n <= 20),
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            vmin=0,
            vmax=vmax,
            linewidths=0.3,
            ax=ax,
        )
        ax.set_title(f'Confusion Matrix — {self.model_type.upper()} Model',
                     fontsize=14, pad=12)
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)
        plt.xticks(rotation=45, ha='right', fontsize=8 if n > 20 else 10)
        plt.yticks(rotation=0, fontsize=8 if n > 20 else 10)
        plt.tight_layout()

        save_path = EvaluationConfig.PLOTS_DIR / f'confusion_matrix_{self.model_type}.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Confusion matrix saved -> {save_path}")

    def plot_per_class_accuracy(self):
        y_pred = self._get_predictions()
        cm = confusion_matrix(self.y_test, y_pred,
                              labels=sorted(self.label_encoder))
        row_sums = cm.sum(axis=1)
        per_class_acc = np.where(row_sums == 0, 0.0,
                                 cm.diagonal() / row_sums.astype(float))

        pairs = list(zip(self.class_names, per_class_acc))
        pairs.sort(key=lambda x: x[1], reverse=True)
        names_sorted, accs_sorted = zip(*pairs)

        colors = ['green' if a >= 0.7 else 'red' for a in accs_sorted]

        n = len(names_sorted)
        fig_width = max(10, n * 0.45)
        fig, ax = plt.subplots(figsize=(fig_width, 5))
        ax.bar(range(n), accs_sorted, color=colors)
        ax.axhline(0.7, color='black', linestyle='--', linewidth=1, label='0.7 threshold')
        ax.set_xticks(range(n))
        ax.set_xticklabels(names_sorted, rotation=45, ha='right',
                           fontsize=8 if n > 20 else 10)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Class', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title(f'Per-Class Accuracy — {self.model_type.upper()} Model',
                     fontsize=13)
        ax.legend(fontsize=9)
        plt.tight_layout()

        save_path = EvaluationConfig.PLOTS_DIR / f'per_class_accuracy_{self.model_type}.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Per-class accuracy saved -> {save_path}")

    def plot_error_analysis(self):
        y_pred = self._get_predictions()
        cm = confusion_matrix(self.y_test, y_pred,
                              labels=sorted(self.label_encoder))
        cm_off = cm.copy()
        np.fill_diagonal(cm_off, 0)

        flat_indices = np.argsort(cm_off.ravel())[::-1]
        pairs_seen = []
        confusion_pairs = []
        for idx in flat_indices:
            if len(confusion_pairs) >= 5:
                break
            i, j = divmod(idx, cm_off.shape[1])
            if cm_off[i, j] > 0:
                pairs_seen.append((i, j))
                confusion_pairs.append((self.class_names[i], self.class_names[j], cm_off[i, j]))

        if confusion_pairs:
            print(f"\nTop confused pairs ({self.model_type}):")
            for true_c, pred_c, count in confusion_pairs:
                print(f"  True={true_c:>8}  Predicted={pred_c:>8}  Count={int(count)}")

        involved_indices = sorted({idx for pair in pairs_seen for idx in pair})
        if len(involved_indices) < 2:
            print("Not enough confusion data to plot error analysis.")
            return

        sub_cm = cm[np.ix_(involved_indices, involved_indices)]
        sub_names = [self.class_names[i] for i in involved_indices]

        row_sums = sub_cm.sum(axis=1, keepdims=True)
        sub_cm_norm = np.where(row_sums == 0, 0.0,
                               sub_cm / row_sums.astype(float))

        fig, ax = plt.subplots(figsize=(max(6, len(sub_names)), max(5, len(sub_names) * 0.8)))
        sns.heatmap(
            sub_cm_norm,
            annot=True,
            fmt='.2f',
            cmap='Reds',
            xticklabels=sub_names,
            yticklabels=sub_names,
            vmin=0,
            vmax=1.0,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title(f'Error Analysis (Top Confused Classes) — {self.model_type.upper()}',
                     fontsize=12)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True', fontsize=10)
        plt.tight_layout()

        save_path = EvaluationConfig.PLOTS_DIR / f'error_analysis_{self.model_type}.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Error analysis saved -> {save_path}")

    def run_full_evaluation(self):
        metrics = self.compute_metrics()

        print("\nClassification Report:")
        self.print_classification_report()

        print("\nSummary:")
        print(f"  {'Metric':<25} {'Value':>8}")
        print(f"  {'-'*35}")
        for key in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
                    'precision_weighted', 'recall_weighted', 'f1_weighted']:
            print(f"  {key:<25} {metrics[key]:>8.4f}")

        self.plot_confusion_matrix(normalize=True)
        self.plot_per_class_accuracy()
        self.plot_error_analysis()

        return metrics


def main():
    for model_type in ['static', 'dynamic']:
        print(f"\n{'='*60}")
        print(f"Evaluating {model_type.upper()} model")
        print('='*60)
        try:
            evaluator = ModelEvaluator(model_type)
            metrics = evaluator.run_full_evaluation()
        except FileNotFoundError as e:
            print(f"Skipping {model_type}: {e}")


if __name__ == "__main__":
    main()
