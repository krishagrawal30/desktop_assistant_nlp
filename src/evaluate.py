import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import LinearSVC

from executor import execute
from intent_classifier import load_data
from ner import extract_entities

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


SYSTEM_COMMANDS = {"SHUTDOWN", "RESTART", "SUSPEND"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate intent, NER, and end-to-end execution performance")
    parser.add_argument(
        "--scope",
        choices=["intent", "e2e", "ner", "both"],
        default="both",
        help="Evaluation scope to run",
    )
    parser.add_argument(
        "--dataset-path",
        default="data/intent_dataset.json",
        help="Path to intent dataset",
    )
    parser.add_argument(
        "--e2e-dataset-path",
        default="data/e2e_eval_dataset.json",
        help="Path to end-to-end evaluation dataset",
    )
    parser.add_argument(
        "--model-path",
        default="models/intent_model.pkl",
        help="Path to trained runtime model",
    )
    parser.add_argument(
        "--vectorizer-path",
        default="models/vectorizer.pkl",
        help="Path to trained runtime vectorizer",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Directory where evaluation reports will be generated",
    )
    parser.add_argument(
        "--display",
        choices=["terminal", "gui", "both", "files"],
        default="terminal",
        help="How to show results (files means save reports only)",
    )
    parser.add_argument(
        "--save-reports",
        action="store_true",
        help="Save JSON/CSV/plot reports even when display mode is terminal/gui",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-plots", action="store_true", help="Skip graph generation")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def normalize_entities(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize_entities(val) for key, val in sorted(value.items())}
    if isinstance(value, list):
        normalized = [normalize_entities(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    return value


def flatten_entity_items(value: Any, prefix: str = "") -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []

    if isinstance(value, dict):
        for key, nested_value in sorted(value.items()):
            nested_prefix = f"{prefix}.{key}" if prefix else key
            items.extend(flatten_entity_items(nested_value, nested_prefix))
        return items

    if isinstance(value, list):
        list_prefix = f"{prefix}[]" if prefix else "item[]"
        for item in value:
            items.extend(flatten_entity_items(item, list_prefix))
        return items

    if prefix:
        items.append((prefix, str(value)))
    else:
        items.append(("value", str(value)))

    return items


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def get_top_confusions(labels: List[str], matrix: List[List[int]], limit: int = 5) -> List[Dict[str, Any]]:
    confusions: List[Dict[str, Any]] = []
    for row_idx, actual_label in enumerate(labels):
        for col_idx, predicted_label in enumerate(labels):
            if row_idx == col_idx:
                continue
            count = int(matrix[row_idx][col_idx])
            if count > 0:
                confusions.append(
                    {
                        "actual": actual_label,
                        "predicted": predicted_label,
                        "count": count,
                    }
                )

    confusions.sort(key=lambda item: item["count"], reverse=True)
    return confusions[:limit]


def format_ratio(value: float) -> str:
    return f"{value:.4f} ({value * 100:.2f}%)"


def build_terminal_summary(
    scope: str,
    seed: int,
    intent_summary: Optional[Dict[str, Any]],
    e2e_summary: Optional[Dict[str, Any]],
    ner_summary: Optional[Dict[str, Any]],
    reports_path: Optional[Path],
    persisted: bool,
) -> str:
    lines: List[str] = []
    lines.append("=" * 64)
    lines.append("Evaluation Summary")
    lines.append("=" * 64)
    lines.append(f"Scope: {scope} | Seed: {seed}")

    if intent_summary:
        selected_metrics = intent_summary["selected_metrics"]
        lines.append("")
        lines.append("[Intent Evaluation]")
        lines.append(f"Selected model: {intent_summary['selected_model']}")
        lines.append(
            "Split sizes: "
            f"train={intent_summary['sample_counts']['train']}, "
            f"val={intent_summary['sample_counts']['validation']}, "
            f"test={intent_summary['sample_counts']['test']}"
        )
        lines.append(
            "Accuracy: "
            f"{format_ratio(selected_metrics['accuracy'])} | "
            f"Macro F1: {format_ratio(selected_metrics['macro_f1'])}"
        )
        lines.append(
            "Precision: "
            f"{format_ratio(selected_metrics['macro_precision'])} | "
            f"Recall: {format_ratio(selected_metrics['macro_recall'])}"
        )
        lines.append(f"Baseline accuracy: {format_ratio(intent_summary['baseline']['accuracy'])}")

        top_confusions = get_top_confusions(
            intent_summary["label_order"],
            intent_summary["confusion_matrix"],
            limit=3,
        )
        if top_confusions:
            lines.append("Top confusions:")
            for item in top_confusions:
                lines.append(
                    "  "
                    f"- actual={item['actual']} predicted={item['predicted']} count={item['count']}"
                )

    if e2e_summary:
        lines.append("")
        lines.append("[End-to-End Evaluation]")
        lines.append(f"Sample count: {e2e_summary['sample_count']}")
        lines.append(
            "Intent accuracy: "
            f"{format_ratio(e2e_summary['intent_accuracy'])} | "
            f"Entity exact match: {format_ratio(e2e_summary['entity_exact_match_rate'])}"
        )
        lines.append(
            "Entity exact match (intent-correct only): "
            f"{format_ratio(e2e_summary['entity_exact_match_when_intent_correct'])}"
        )
        lines.append(
            "Validation pass rate: "
            f"{format_ratio(e2e_summary['validation_pass_rate'])} | "
            f"Execution success rate: {format_ratio(e2e_summary['execution_success_rate'])}"
        )
        lines.append(f"End-to-end success rate: {format_ratio(e2e_summary['end_to_end_success_rate'])}")

        stage_breakdown = e2e_summary.get("stage_breakdown", {})
        if stage_breakdown:
            lines.append("Stage breakdown:")
            for stage_name, stage_count in stage_breakdown.items():
                lines.append(f"  - {stage_name}: {stage_count}")

    if ner_summary:
        lines.append("")
        lines.append("[NER Evaluation]")
        lines.append(f"Sample count: {ner_summary['sample_count']}")
        lines.append(
            "Micro Precision: "
            f"{format_ratio(ner_summary['micro_precision'])} | "
            f"Micro Recall: {format_ratio(ner_summary['micro_recall'])}"
        )
        lines.append(
            "Micro F1: "
            f"{format_ratio(ner_summary['micro_f1'])} | "
            f"Entity exact match: {format_ratio(ner_summary['entity_exact_match_rate'])}"
        )
        lines.append(
            "Entity counts: "
            f"expected={ner_summary['expected_entity_count']}, "
            f"predicted={ner_summary['predicted_entity_count']}, "
            f"tp={ner_summary['true_positive_entities']}, "
            f"fp={ner_summary['false_positive_entities']}, "
            f"fn={ner_summary['false_negative_entities']}"
        )

    lines.append("")
    if persisted and reports_path is not None:
        lines.append(f"Reports saved to: {reports_path}")
    else:
        lines.append("Reports were not written to disk.")
        lines.append("Tip: use --save-reports or --display files to persist artifacts.")

    return "\n".join(lines)


def run_evaluation(
    scope: str = "both",
    dataset_path: str = "data/intent_dataset.json",
    e2e_dataset_path: str = "data/e2e_eval_dataset.json",
    model_path: str = "models/intent_model.pkl",
    vectorizer_path: str = "models/vectorizer.pkl",
    reports_dir: str = "reports",
    seed: int = 42,
    persist_reports: bool = False,
    no_plots: bool = False,
) -> Dict[str, Any]:
    should_plot_reports = persist_reports and not no_plots

    if should_plot_reports and plt is None:
        raise RuntimeError("matplotlib is not installed. Install requirements or use no_plots=True")

    run_dir: Optional[Path] = None
    if persist_reports:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(reports_dir) / f"run_{timestamp}"
        ensure_dir(run_dir)

    selected_model = None
    selected_vectorizer = None

    intent_summary = None
    e2e_summary = None
    ner_summary = None

    if scope in {"intent", "both"}:
        selected_model, selected_vectorizer, intent_summary = run_intent_evaluation(
            dataset_path=dataset_path,
            seed=seed,
            run_dir=run_dir,
            no_plots=no_plots,
            persist_reports=persist_reports,
        )

    if scope in {"e2e", "ner", "both"}:
        if selected_model is None or selected_vectorizer is None:
            resolved_model_path = Path(model_path)
            resolved_vectorizer_path = Path(vectorizer_path)
            if not resolved_model_path.exists() or not resolved_vectorizer_path.exists():
                raise FileNotFoundError(
                    "Model/vectorizer not found for e2e evaluation. Run training first or use scope='both'."
                )
            selected_model = joblib.load(resolved_model_path)
            selected_vectorizer = joblib.load(resolved_vectorizer_path)

        e2e_summary, ner_summary = run_e2e_evaluation(
            model=selected_model,
            vectorizer=selected_vectorizer,
            e2e_dataset_path=e2e_dataset_path,
            run_dir=run_dir,
            no_plots=no_plots,
            persist_reports=persist_reports,
        )

        if scope == "ner":
            e2e_summary = None

    summary_payload = {
        "scope": scope,
        "seed": seed,
        "reports_dir": str(run_dir) if run_dir is not None else None,
        "persisted_reports": persist_reports,
        "intent": intent_summary,
        "e2e": e2e_summary,
        "ner": ner_summary,
    }

    if persist_reports and run_dir is not None:
        write_json(run_dir / "summary.json", summary_payload)

    return summary_payload


def show_results_gui(summary_text: str) -> None:
    import tkinter as tk
    from tkinter import ttk
    from tkinter.scrolledtext import ScrolledText

    root = tk.Tk()
    root.title("Desktop Assistant Evaluation Results")
    root.geometry("980x720")

    frame = ttk.Frame(root, padding=12)
    frame.pack(fill="both", expand=True)

    title = ttk.Label(frame, text="Evaluation Summary", font=("Segoe UI", 14, "bold"))
    title.pack(anchor="w")

    text_box = ScrolledText(frame, wrap="word", font=("Consolas", 10))
    text_box.pack(fill="both", expand=True, pady=(10, 0))
    text_box.insert("1.0", summary_text)
    text_box.configure(state="disabled")

    close_button = ttk.Button(frame, text="Close", command=root.destroy)
    close_button.pack(anchor="e", pady=(10, 0))

    root.mainloop()


def split_data(
    texts: List[str],
    labels: List[str],
    seed: int,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> Dict[str, List[str]]:
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    val_ratio_inside_train_val = val_size / (1.0 - test_size)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts,
        train_val_labels,
        test_size=val_ratio_inside_train_val,
        random_state=seed,
        stratify=train_val_labels,
    )

    return {
        "train_texts": train_texts,
        "train_labels": train_labels,
        "val_texts": val_texts,
        "val_labels": val_labels,
        "test_texts": test_texts,
        "test_labels": test_labels,
    }


def get_cv_folds(labels: List[str], desired_folds: int = 5) -> int:
    counts = Counter(labels)
    min_count = min(counts.values())
    if min_count < 2:
        raise ValueError("Not enough samples per class for cross-validation")
    return min(desired_folds, min_count)


def compute_metrics(y_true: List[str], y_pred: List[str], label_order: List[str]) -> Dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=label_order,
        output_dict=True,
        zero_division=0,
    )
    counts_matrix = confusion_matrix(y_true, y_pred, labels=label_order)
    normalized_matrix = confusion_matrix(y_true, y_pred, labels=label_order, normalize="true")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": counts_matrix.tolist(),
        "normalized_confusion_matrix": np.nan_to_num(normalized_matrix).tolist(),
    }


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: List[str],
    output_path: Path,
    title: str,
    annotate: bool,
    float_format: str,
) -> None:
    fig_size = max(10, int(len(labels) * 0.7))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    if annotate:
        for row_idx in range(len(labels)):
            for col_idx in range(len(labels)):
                ax.text(
                    col_idx,
                    row_idx,
                    format(matrix[row_idx, col_idx], float_format),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_per_intent_scores(report: Dict[str, Any], labels: List[str], output_path: Path) -> None:
    f1_scores = [report[label]["f1-score"] for label in labels]
    recall_scores = [report[label]["recall"] for label in labels]

    x_pos = np.arange(len(labels))
    width = 0.4

    fig_width = max(12, int(len(labels) * 0.8))
    fig, ax = plt.subplots(figsize=(fig_width, 7))
    ax.bar(x_pos - width / 2, f1_scores, width=width, label="F1")
    ax.bar(x_pos + width / 2, recall_scores, width=width, label="Recall")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Per-Intent F1 and Recall")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_model_comparison(model_results: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    model_names = list(model_results.keys())
    metric_names = ["accuracy", "macro_f1", "weighted_f1"]

    x_pos = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    for idx, model_name in enumerate(model_names):
        offsets = x_pos + (idx - 0.5) * width
        values = [model_results[model_name][metric] for metric in metric_names]
        ax.bar(offsets, values, width=width, label=model_name)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison on Test Split")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_intent_evaluation(
    dataset_path: str,
    seed: int,
    run_dir: Optional[Path],
    no_plots: bool,
    persist_reports: bool,
) -> Tuple[Any, TfidfVectorizer, Dict[str, Any]]:
    texts, labels = load_data(dataset_path)
    split = split_data(texts, labels, seed=seed)

    label_order = sorted(set(labels))
    cv_folds = get_cv_folds(split["train_labels"])

    train_vectorizer = TfidfVectorizer()
    x_train = train_vectorizer.fit_transform(split["train_texts"])
    x_val = train_vectorizer.transform(split["val_texts"])

    logistic_grid = GridSearchCV(
        LogisticRegression(max_iter=2000, random_state=seed),
        {"C": [0.1, 1, 5, 10, 20, 50, 100]},
        cv=cv_folds,
        scoring="f1_macro",
        n_jobs=-1,
    )
    svm_grid = GridSearchCV(
        LinearSVC(max_iter=5000),
        {"C": [0.1, 1, 5, 10, 20, 50, 100]},
        cv=cv_folds,
        scoring="f1_macro",
        n_jobs=-1,
    )

    logistic_grid.fit(x_train, split["train_labels"])
    svm_grid.fit(x_train, split["train_labels"])

    logistic_val_pred = logistic_grid.best_estimator_.predict(x_val)
    svm_val_pred = svm_grid.best_estimator_.predict(x_val)

    logistic_val_macro_f1 = f1_score(split["val_labels"], logistic_val_pred, average="macro", zero_division=0)
    svm_val_macro_f1 = f1_score(split["val_labels"], svm_val_pred, average="macro", zero_division=0)

    selected_model_name = "logistic_regression" if logistic_val_macro_f1 >= svm_val_macro_f1 else "linear_svm"

    train_val_texts = split["train_texts"] + split["val_texts"]
    train_val_labels = split["train_labels"] + split["val_labels"]

    final_vectorizer = TfidfVectorizer()
    x_train_val = final_vectorizer.fit_transform(train_val_texts)
    x_test = final_vectorizer.transform(split["test_texts"])

    final_lr = LogisticRegression(max_iter=2000, random_state=seed, **logistic_grid.best_params_)
    final_lr.fit(x_train_val, train_val_labels)

    final_svm = LinearSVC(max_iter=5000, **svm_grid.best_params_)
    final_svm.fit(x_train_val, train_val_labels)

    lr_test_pred = final_lr.predict(x_test)
    svm_test_pred = final_svm.predict(x_test)

    lr_metrics = compute_metrics(split["test_labels"], list(lr_test_pred), label_order)
    svm_metrics = compute_metrics(split["test_labels"], list(svm_test_pred), label_order)

    model_results = {
        "logistic_regression": {
            "accuracy": lr_metrics["accuracy"],
            "macro_f1": lr_metrics["macro_f1"],
            "weighted_f1": lr_metrics["weighted_f1"],
            "val_macro_f1": float(logistic_val_macro_f1),
            "best_params": logistic_grid.best_params_,
            "cv_best_score": float(logistic_grid.best_score_),
        },
        "linear_svm": {
            "accuracy": svm_metrics["accuracy"],
            "macro_f1": svm_metrics["macro_f1"],
            "weighted_f1": svm_metrics["weighted_f1"],
            "val_macro_f1": float(svm_val_macro_f1),
            "best_params": svm_grid.best_params_,
            "cv_best_score": float(svm_grid.best_score_),
        },
    }

    selected_model = final_lr if selected_model_name == "logistic_regression" else final_svm
    selected_predictions = list(lr_test_pred) if selected_model_name == "logistic_regression" else list(svm_test_pred)
    selected_metrics = lr_metrics if selected_model_name == "logistic_regression" else svm_metrics

    majority_label = Counter(train_val_labels).most_common(1)[0][0]
    baseline_predictions = [majority_label for _ in split["test_labels"]]
    baseline_accuracy = float(accuracy_score(split["test_labels"], baseline_predictions))

    intent_summary = {
        "dataset_path": dataset_path,
        "sample_counts": {
            "train": len(split["train_texts"]),
            "validation": len(split["val_texts"]),
            "test": len(split["test_texts"]),
            "total": len(texts),
        },
        "label_count": len(label_order),
        "label_order": label_order,
        "cv_folds": cv_folds,
        "selection_metric": "macro_f1 on validation split",
        "selected_model": selected_model_name,
        "baseline": {
            "majority_label": majority_label,
            "accuracy": baseline_accuracy,
        },
        "selected_metrics": {
            "accuracy": selected_metrics["accuracy"],
            "macro_precision": selected_metrics["macro_precision"],
            "macro_recall": selected_metrics["macro_recall"],
            "macro_f1": selected_metrics["macro_f1"],
            "weighted_f1": selected_metrics["weighted_f1"],
        },
        "model_results": model_results,
        "classification_report": selected_metrics["classification_report"],
        "confusion_matrix": selected_metrics["confusion_matrix"],
        "normalized_confusion_matrix": selected_metrics["normalized_confusion_matrix"],
        "test_labels": split["test_labels"],
        "test_predictions": selected_predictions,
    }

    if persist_reports:
        if run_dir is None:
            raise ValueError("run_dir must be provided when persist_reports is enabled")

        intent_dir = run_dir / "intent"
        ensure_dir(intent_dir)

        write_json(intent_dir / "intent_summary.json", intent_summary)

        write_json(
            intent_dir / "model_comparison.json",
            {
                "models": model_results,
                "selected_model": selected_model_name,
            },
        )

        write_json(
            intent_dir / "classification_report.json",
            {
                "model": selected_model_name,
                "report": selected_metrics["classification_report"],
            },
        )

        write_json(
            intent_dir / "confusion_matrix.json",
            {
                "model": selected_model_name,
                "labels": label_order,
                "counts": selected_metrics["confusion_matrix"],
                "normalized": selected_metrics["normalized_confusion_matrix"],
            },
        )

        test_predictions_df = pd.DataFrame(
            {
                "text": split["test_texts"],
                "actual_intent": split["test_labels"],
                "predicted_intent": selected_predictions,
            }
        )
        test_predictions_df.to_csv(intent_dir / "test_predictions.csv", index=False)

        pd.DataFrame(logistic_grid.cv_results_).to_csv(intent_dir / "cv_results_logistic.csv", index=False)
        pd.DataFrame(svm_grid.cv_results_).to_csv(intent_dir / "cv_results_svm.csv", index=False)

        joblib.dump(selected_model, intent_dir / "selected_model.pkl")
        joblib.dump(final_vectorizer, intent_dir / "selected_vectorizer.pkl")

        if not no_plots:
            if plt is None:
                raise RuntimeError("matplotlib is required for plot generation. Install dependencies first.")

            plots_dir = intent_dir / "plots"
            ensure_dir(plots_dir)

            plot_confusion_matrix(
                np.array(selected_metrics["confusion_matrix"]),
                label_order,
                plots_dir / "confusion_matrix_counts.png",
                "Intent Confusion Matrix (Counts)",
                annotate=True,
                float_format=".0f",
            )
            plot_confusion_matrix(
                np.array(selected_metrics["normalized_confusion_matrix"]),
                label_order,
                plots_dir / "confusion_matrix_normalized.png",
                "Intent Confusion Matrix (Normalized)",
                annotate=True,
                float_format=".2f",
            )
            plot_per_intent_scores(
                selected_metrics["classification_report"],
                label_order,
                plots_dir / "per_intent_f1_recall.png",
            )
            plot_model_comparison(model_results, plots_dir / "model_comparison.png")

    return selected_model, final_vectorizer, intent_summary


def run_e2e_evaluation(
    model: Any,
    vectorizer: TfidfVectorizer,
    e2e_dataset_path: str,
    run_dir: Optional[Path],
    no_plots: bool,
    persist_reports: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    e2e_path = Path(e2e_dataset_path)
    if not e2e_path.exists():
        raise FileNotFoundError(f"End-to-end dataset not found: {e2e_dataset_path}")

    with e2e_path.open("r", encoding="utf-8") as handle:
        samples = json.load(handle)

    rows: List[Dict[str, Any]] = []
    total_tp_entities = 0
    total_fp_entities = 0
    total_fn_entities = 0
    entity_type_counts: Dict[str, Dict[str, int]] = {}
    per_intent_entity_counts: Dict[str, Dict[str, int]] = {}

    for sample in samples:
        text = sample["text"]
        expected_intent = sample["intent"]
        expected_entities = sample.get("expected_entities", {})

        x_input = vectorizer.transform([text])
        predicted_intent = model.predict(x_input)[0]

        extracted_entities = extract_entities(text, predicted_intent)
        normalized_expected_entities = normalize_entities(expected_entities)
        normalized_extracted_entities = normalize_entities(extracted_entities)

        expected_entity_items = set(flatten_entity_items(normalized_expected_entities))
        extracted_entity_items = set(flatten_entity_items(normalized_extracted_entities))
        matched_entity_items = expected_entity_items & extracted_entity_items
        fp_entity_items = extracted_entity_items - expected_entity_items
        fn_entity_items = expected_entity_items - extracted_entity_items

        for entity_type, _ in expected_entity_items:
            entity_type_bucket = entity_type_counts.setdefault(
                entity_type,
                {"tp": 0, "fp": 0, "fn": 0, "support": 0},
            )
            entity_type_bucket["support"] += 1

        for entity_type, _ in matched_entity_items:
            entity_type_bucket = entity_type_counts.setdefault(
                entity_type,
                {"tp": 0, "fp": 0, "fn": 0, "support": 0},
            )
            entity_type_bucket["tp"] += 1
            total_tp_entities += 1

        for entity_type, _ in fp_entity_items:
            entity_type_bucket = entity_type_counts.setdefault(
                entity_type,
                {"tp": 0, "fp": 0, "fn": 0, "support": 0},
            )
            entity_type_bucket["fp"] += 1
            total_fp_entities += 1

        for entity_type, _ in fn_entity_items:
            entity_type_bucket = entity_type_counts.setdefault(
                entity_type,
                {"tp": 0, "fp": 0, "fn": 0, "support": 0},
            )
            entity_type_bucket["fn"] += 1
            total_fn_entities += 1

        per_intent_entity_bucket = per_intent_entity_counts.setdefault(
            expected_intent,
            {"samples": 0, "exact_match_count": 0},
        )
        per_intent_entity_bucket["samples"] += 1

        intent_correct = predicted_intent == expected_intent
        entity_match = normalized_extracted_entities == normalized_expected_entities
        per_intent_entity_bucket["exact_match_count"] += int(entity_match)

        validation_pass = predicted_intent in SYSTEM_COMMANDS or bool(extracted_entities)

        execution_result = execute(predicted_intent, extracted_entities, dry_run=True)
        execution_success = execution_result.get("status") == "success"

        end_to_end_success = bool(intent_correct and entity_match and validation_pass and execution_success)

        if end_to_end_success:
            failure_stage = "success"
        elif not intent_correct:
            failure_stage = "intent_mismatch"
        elif not entity_match:
            failure_stage = "entity_mismatch"
        elif not validation_pass:
            failure_stage = "validation_reject"
        else:
            failure_stage = f"execution_{execution_result.get('status')}"

        rows.append(
            {
                "text": text,
                "expected_intent": expected_intent,
                "predicted_intent": predicted_intent,
                "intent_correct": intent_correct,
                "expected_entities": json.dumps(normalized_expected_entities, sort_keys=True),
                "extracted_entities": json.dumps(normalized_extracted_entities, sort_keys=True),
                "entity_match": entity_match,
                "validation_pass": validation_pass,
                "execution_status": execution_result.get("status"),
                "execution_message": execution_result.get("message"),
                "end_to_end_success": end_to_end_success,
                "failure_stage": failure_stage,
            }
        )

    df = pd.DataFrame(rows)

    intent_accuracy = float(df["intent_correct"].mean())
    entity_match_rate = float(df["entity_match"].mean())
    validation_pass_rate = float(df["validation_pass"].mean())
    execution_success_rate = float((df["execution_status"] == "success").mean())
    e2e_success_rate = float(df["end_to_end_success"].mean())

    correct_intent_mask = df["intent_correct"]
    if correct_intent_mask.any():
        entity_match_when_intent_correct = float(df.loc[correct_intent_mask, "entity_match"].mean())
    else:
        entity_match_when_intent_correct = 0.0

    per_intent = (
        df.groupby("expected_intent", as_index=False)
        .agg(
            samples=("expected_intent", "count"),
            intent_accuracy=("intent_correct", "mean"),
            entity_match_rate=("entity_match", "mean"),
            e2e_success_rate=("end_to_end_success", "mean"),
        )
        .sort_values(by="expected_intent")
    )

    label_order = sorted(set(df["expected_intent"]) | set(df["predicted_intent"]))
    e2e_counts = confusion_matrix(df["expected_intent"], df["predicted_intent"], labels=label_order)
    e2e_normalized = confusion_matrix(
        df["expected_intent"],
        df["predicted_intent"],
        labels=label_order,
        normalize="true",
    )

    stage_breakdown = df["failure_stage"].value_counts().sort_index()

    e2e_summary = {
        "sample_count": int(len(df)),
        "intent_accuracy": intent_accuracy,
        "entity_exact_match_rate": entity_match_rate,
        "entity_exact_match_when_intent_correct": entity_match_when_intent_correct,
        "validation_pass_rate": validation_pass_rate,
        "execution_success_rate": execution_success_rate,
        "end_to_end_success_rate": e2e_success_rate,
        "stage_breakdown": {str(key): int(value) for key, value in stage_breakdown.items()},
    }

    per_entity_type_rows: List[Dict[str, Any]] = []
    for entity_type in sorted(entity_type_counts.keys()):
        counts = entity_type_counts[entity_type]
        entity_precision = safe_ratio(counts["tp"], counts["tp"] + counts["fp"])
        entity_recall = safe_ratio(counts["tp"], counts["tp"] + counts["fn"])
        entity_f1 = safe_ratio(2 * entity_precision * entity_recall, entity_precision + entity_recall)

        per_entity_type_rows.append(
            {
                "entity_type": entity_type,
                "support": int(counts["support"]),
                "tp": int(counts["tp"]),
                "fp": int(counts["fp"]),
                "fn": int(counts["fn"]),
                "precision": entity_precision,
                "recall": entity_recall,
                "f1": entity_f1,
            }
        )

    per_intent_entity_rows: List[Dict[str, Any]] = []
    for intent_name in sorted(per_intent_entity_counts.keys()):
        counts = per_intent_entity_counts[intent_name]
        per_intent_entity_rows.append(
            {
                "intent": intent_name,
                "samples": int(counts["samples"]),
                "exact_match_count": int(counts["exact_match_count"]),
                "exact_match_rate": safe_ratio(counts["exact_match_count"], counts["samples"]),
            }
        )

    micro_precision = safe_ratio(total_tp_entities, total_tp_entities + total_fp_entities)
    micro_recall = safe_ratio(total_tp_entities, total_tp_entities + total_fn_entities)
    micro_f1 = safe_ratio(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    if per_entity_type_rows:
        macro_precision = float(np.mean([row["precision"] for row in per_entity_type_rows]))
        macro_recall = float(np.mean([row["recall"] for row in per_entity_type_rows]))
        macro_f1 = float(np.mean([row["f1"] for row in per_entity_type_rows]))
    else:
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0

    ner_summary = {
        "sample_count": int(len(df)),
        "entity_type_count": int(len(per_entity_type_rows)),
        "entity_exact_match_rate": entity_match_rate,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "expected_entity_count": int(total_tp_entities + total_fn_entities),
        "predicted_entity_count": int(total_tp_entities + total_fp_entities),
        "true_positive_entities": int(total_tp_entities),
        "false_positive_entities": int(total_fp_entities),
        "false_negative_entities": int(total_fn_entities),
        "per_entity_type": per_entity_type_rows,
        "per_intent_entity_exact_match": per_intent_entity_rows,
    }

    per_entity_type_df = pd.DataFrame(
        per_entity_type_rows,
        columns=["entity_type", "support", "tp", "fp", "fn", "precision", "recall", "f1"],
    )
    per_intent_entity_df = pd.DataFrame(
        per_intent_entity_rows,
        columns=["intent", "samples", "exact_match_count", "exact_match_rate"],
    )

    if persist_reports:
        if run_dir is None:
            raise ValueError("run_dir must be provided when persist_reports is enabled")

        e2e_dir = run_dir / "e2e"
        ensure_dir(e2e_dir)

        write_json(
            e2e_dir / "e2e_summary.json",
            {
                "dataset_path": e2e_dataset_path,
                **e2e_summary,
            },
        )

        write_json(
            e2e_dir / "ner_summary.json",
            {
                "dataset_path": e2e_dataset_path,
                **ner_summary,
            },
        )

        write_json(
            e2e_dir / "e2e_confusion_matrix.json",
            {
                "labels": label_order,
                "counts": e2e_counts.tolist(),
                "normalized": np.nan_to_num(e2e_normalized).tolist(),
            },
        )

        df.to_csv(e2e_dir / "e2e_samples.csv", index=False)
        per_intent.to_csv(e2e_dir / "e2e_per_intent.csv", index=False)
        per_entity_type_df.to_csv(e2e_dir / "ner_per_entity_type.csv", index=False)
        per_intent_entity_df.to_csv(e2e_dir / "ner_per_intent_entity_match.csv", index=False)

        if not no_plots:
            if plt is None:
                raise RuntimeError("matplotlib is required for plot generation. Install dependencies first.")

            plots_dir = e2e_dir / "plots"
            ensure_dir(plots_dir)

            plot_confusion_matrix(
                e2e_counts,
                label_order,
                plots_dir / "intent_confusion_matrix_counts.png",
                "End-to-End Intent Confusion Matrix (Counts)",
                annotate=True,
                float_format=".0f",
            )

            plot_confusion_matrix(
                np.nan_to_num(e2e_normalized),
                label_order,
                plots_dir / "intent_confusion_matrix_normalized.png",
                "End-to-End Intent Confusion Matrix (Normalized)",
                annotate=True,
                float_format=".2f",
            )

            stage_labels = list(stage_breakdown.index)
            stage_values = list(stage_breakdown.values)
            stage_positions = np.arange(len(stage_labels))
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(stage_positions, stage_values, color="#3b82f6")
            ax.set_ylabel("Count")
            ax.set_title("End-to-End Stage Breakdown")
            ax.set_xticks(stage_positions)
            ax.set_xticklabels(stage_labels, rotation=35, ha="right")
            fig.tight_layout()
            fig.savefig(plots_dir / "stage_breakdown.png", dpi=180)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(12, 6))
            x_pos = np.arange(len(per_intent))
            width = 0.4
            ax.bar(x_pos - width / 2, per_intent["entity_match_rate"], width=width, label="Entity Match")
            ax.bar(x_pos + width / 2, per_intent["e2e_success_rate"], width=width, label="E2E Success")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(per_intent["expected_intent"], rotation=45, ha="right")
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Rate")
            ax.set_title("Per-Intent Entity Match and End-to-End Success")
            ax.legend()
            fig.tight_layout()
            fig.savefig(plots_dir / "per_intent_entity_vs_e2e.png", dpi=180)
            plt.close(fig)

    return e2e_summary, ner_summary


def main() -> None:
    args = parse_args()

    persist_reports = args.save_reports or args.display == "files"
    summary_payload = run_evaluation(
        scope=args.scope,
        dataset_path=args.dataset_path,
        e2e_dataset_path=args.e2e_dataset_path,
        model_path=args.model_path,
        vectorizer_path=args.vectorizer_path,
        reports_dir=args.reports_dir,
        seed=args.seed,
        persist_reports=persist_reports,
        no_plots=args.no_plots,
    )

    run_dir = Path(summary_payload["reports_dir"]) if summary_payload["reports_dir"] else None

    summary_text = build_terminal_summary(
        scope=args.scope,
        seed=args.seed,
        intent_summary=summary_payload["intent"],
        e2e_summary=summary_payload["e2e"],
        ner_summary=summary_payload["ner"],
        reports_path=run_dir,
        persisted=persist_reports,
    )

    if args.display == "files":
        print(f"Evaluation complete. Reports generated at: {run_dir}")
        return

    if args.display in {"terminal", "both"}:
        print(summary_text)

    if args.display in {"gui", "both"}:
        try:
            show_results_gui(summary_text)
        except Exception as exc:
            print(f"GUI could not be opened: {exc}")
            if args.display == "gui":
                print(summary_text)


if __name__ == "__main__":
    main()
