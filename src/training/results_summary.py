import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.training.evaluate_models import ModelEvaluator
from src.utils.config import EvaluationConfig, ROOT_DIR

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('ggplot')

DOCS_DIR = ROOT_DIR / "docs"


class ResultsSummary:

    def __init__(self):
        self.metrics = {}
        EvaluationConfig.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        DOCS_DIR.mkdir(parents=True, exist_ok=True)

        for model_type in ['static', 'dynamic']:
            try:
                evaluator = ModelEvaluator(model_type)
                self.metrics[model_type] = evaluator.run_full_evaluation()
                print(f"Loaded metrics for {model_type} model.")
            except FileNotFoundError as e:
                print(f"Skipping {model_type}: {e}")
                self.metrics[model_type] = None

    def generate_latex_table(self):
        metric_labels = {
            'accuracy': 'Accuracy',
            'precision_macro': 'Precision (Macro)',
            'recall_macro': 'Recall (Macro)',
            'f1_macro': 'F1 Score (Macro)',
            'precision_weighted': 'Precision (Weighted)',
            'recall_weighted': 'Recall (Weighted)',
            'f1_weighted': 'F1 Score (Weighted)',
        }

        lines = [
            r'\begin{table}[h]',
            r'\centering',
            r'\caption{ASL Recognition System — Model Evaluation Results}',
            r'\label{tab:model_results}',
            r'\begin{tabular}{lcc}',
            r'\hline',
            r'\textbf{Metric} & \textbf{Static Model (36 classes)} & \textbf{Dynamic Model (10 classes)} \\',
            r'\hline',
        ]

        for key, label in metric_labels.items():
            s_val = self.metrics.get('static')
            d_val = self.metrics.get('dynamic')
            s_str = f"{s_val[key]:.4f}" if s_val else 'N/A'
            d_str = f"{d_val[key]:.4f}" if d_val else 'N/A'
            lines.append(f"{label} & {s_str} & {d_str} \\\\")

        lines += [
            r'\hline',
            r'\end{tabular}',
            r'\end{table}',
        ]

        latex = '\n'.join(lines)
        save_path = DOCS_DIR / 'results_table.tex'
        save_path.write_text(latex)
        print(f"LaTeX table saved -> {save_path}")
        return latex

    def generate_markdown_table(self):
        metric_labels = {
            'accuracy': 'Accuracy',
            'precision_macro': 'Precision (Macro)',
            'recall_macro': 'Recall (Macro)',
            'f1_macro': 'F1 Score (Macro)',
            'precision_weighted': 'Precision (Weighted)',
            'recall_weighted': 'Recall (Weighted)',
            'f1_weighted': 'F1 Score (Weighted)',
        }

        lines = [
            '| Metric | Static Model (36 classes) | Dynamic Model (10 classes) |',
            '|--------|--------------------------|---------------------------|',
        ]

        for key, label in metric_labels.items():
            s_val = self.metrics.get('static')
            d_val = self.metrics.get('dynamic')
            s_str = f"{s_val[key]:.4f}" if s_val else 'N/A'
            d_str = f"{d_val[key]:.4f}" if d_val else 'N/A'
            lines.append(f"| {label} | {s_str} | {d_str} |")

        md = '\n'.join(lines)
        save_path = DOCS_DIR / 'results_table.md'
        save_path.write_text(md)
        print(f"Markdown table saved -> {save_path}")
        return md

    def generate_summary_figure(self):
        metric_keys = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        metric_display = ['Accuracy', 'F1 (Macro)', 'Precision (Macro)', 'Recall (Macro)']

        static_vals = []
        dynamic_vals = []

        for key in metric_keys:
            s = self.metrics.get('static')
            d = self.metrics.get('dynamic')
            static_vals.append(s[key] if s else 0.0)
            dynamic_vals.append(d[key] if d else 0.0)

        x = np.arange(len(metric_keys))
        width = 0.35

        fig, ax = plt.subplots(figsize=(9, 5))
        bars1 = ax.bar(x - width / 2, static_vals, width,
                       label='Static (36 classes)', color='steelblue', alpha=0.85)
        bars2 = ax.bar(x + width / 2, dynamic_vals, width,
                       label='Dynamic (10 classes)', color='darkorange', alpha=0.85)

        for bar in list(bars1) + list(bars2):
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords='offset points',
                            ha='center', va='bottom', fontsize=8)

        ax.set_ylim(0, 1.12)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_display, fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('ASL Recognition — Static vs. Dynamic Model Comparison',
                     fontsize=13, pad=12)
        ax.legend(fontsize=10)
        plt.tight_layout()

        save_path = EvaluationConfig.PLOTS_DIR / 'model_comparison.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Comparison figure saved -> {save_path}")

    def run(self):
        self.generate_latex_table()
        self.generate_markdown_table()
        self.generate_summary_figure()
        print("\nResults summary generation complete.")


def main():
    ResultsSummary().run()


if __name__ == "__main__":
    main()
