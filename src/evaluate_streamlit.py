from typing import Sequence

import numpy as np
import pandas as pd
import streamlit as st

from evaluate import get_top_confusions, run_evaluation

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def as_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def as_count(rate: float, total: int) -> int:
    return max(0, min(total, int(round(rate * total))))


def render_heatmap(
    *,
    title: str,
    labels: Sequence[str],
    matrix: Sequence[Sequence[float]],
    value_format: str,
) -> None:
    if plt is None:
        st.warning("matplotlib is not available. Install dependencies to display heatmaps.")
        return

    if not labels or not matrix:
        return

    matrix_array = np.array(matrix, dtype=float)
    if matrix_array.size == 0:
        return

    size = max(6, min(14, int(len(labels) * 0.75) + 2))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(matrix_array, cmap="Blues", aspect="auto")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    axis_positions = np.arange(len(labels))
    ax.set_xticks(axis_positions)
    ax.set_yticks(axis_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    threshold = matrix_array.max() * 0.6 if matrix_array.max() > 0 else 0
    for row_idx in range(matrix_array.shape[0]):
        for col_idx in range(matrix_array.shape[1]):
            value = format(matrix_array[row_idx, col_idx], value_format)
            text_color = "white" if matrix_array[row_idx, col_idx] >= threshold else "black"
            ax.text(col_idx, row_idx, value, ha="center", va="center", color=text_color, fontsize=8)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_pie_chart(*, title: str, labels: Sequence[str], values: Sequence[int], donut: bool = False) -> None:
    if plt is None:
        st.warning("matplotlib is not available. Install dependencies to display pie charts.")
        return

    if not labels or not values or sum(values) <= 0:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    wedge_width = 0.45 if donut else 1.0
    ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"width": wedge_width, "edgecolor": "white"},
    )
    ax.axis("equal")
    ax.set_title(title)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_intent_section(intent_summary: dict) -> None:
    st.subheader("Intent Evaluation")

    selected_metrics = intent_summary["selected_metrics"]
    split = intent_summary["sample_counts"]

    st.caption(
        f"Model: {intent_summary['selected_model']} | "
        f"Train: {split['train']} | Val: {split['validation']} | Test: {split['test']}"
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", as_percent(selected_metrics["accuracy"]))
    col2.metric("Macro F1", as_percent(selected_metrics["macro_f1"]))
    col3.metric("Baseline", as_percent(intent_summary["baseline"]["accuracy"]))

    model_rows = []
    for model_name, metrics in intent_summary["model_results"].items():
        model_rows.append(
            {
                "model": model_name,
                "validation_macro_f1": metrics["val_macro_f1"],
                "test_accuracy": metrics["accuracy"],
                "test_macro_f1": metrics["macro_f1"],
            }
        )

    model_df = pd.DataFrame(model_rows).set_index("model")
    st.markdown("Graph 1: Model comparison")
    st.bar_chart(model_df)

    selected_profile_df = pd.DataFrame(
        [
            {"metric": "accuracy", "score": selected_metrics["accuracy"]},
            {"metric": "macro_precision", "score": selected_metrics["macro_precision"]},
            {"metric": "macro_recall", "score": selected_metrics["macro_recall"]},
            {"metric": "macro_f1", "score": selected_metrics["macro_f1"]},
            {"metric": "weighted_f1", "score": selected_metrics["weighted_f1"]},
        ]
    ).set_index("metric")
    st.markdown("Graph 2: Selected model metric profile")
    st.line_chart(selected_profile_df)

    label_order = intent_summary["label_order"]
    report = intent_summary.get("classification_report", {})
    per_intent_rows = []
    for label in label_order:
        label_metrics = report.get(label, {})
        per_intent_rows.append(
            {
                "intent": label,
                "precision": label_metrics.get("precision", 0.0),
                "recall": label_metrics.get("recall", 0.0),
                "f1_score": label_metrics.get("f1-score", 0.0),
            }
        )

    if per_intent_rows:
        per_intent_df = pd.DataFrame(per_intent_rows).set_index("intent")
        st.markdown("Graph 3: Per-intent precision / recall / F1")
        st.bar_chart(per_intent_df)

    st.markdown("Graph 4: Confusion matrix (counts)")
    render_heatmap(
        title="Intent Confusion Matrix (Counts)",
        labels=label_order,
        matrix=intent_summary["confusion_matrix"],
        value_format=".0f",
    )

    st.markdown("Graph 5: Confusion matrix (normalized)")
    render_heatmap(
        title="Intent Confusion Matrix (Normalized)",
        labels=label_order,
        matrix=intent_summary["normalized_confusion_matrix"],
        value_format=".2f",
    )

    confusions = get_top_confusions(
        label_order,
        intent_summary["confusion_matrix"],
        limit=8,
    )
    if confusions:
        confusions_df = pd.DataFrame(confusions)
        confusions_df["pair"] = confusions_df["actual"] + " -> " + confusions_df["predicted"]
        st.markdown("Graph 6: Top confusion pairs")
        st.bar_chart(confusions_df.set_index("pair")[["count"]])
        st.dataframe(confusions_df[["actual", "predicted", "count"]], use_container_width=True, hide_index=True)


def render_e2e_section(e2e_summary: dict) -> None:
    st.subheader("End-to-End Evaluation")
    sample_count = int(e2e_summary["sample_count"])
    st.caption(f"Samples: {sample_count}")

    col1, col2, col3 = st.columns(3)
    col1.metric("E2E Success", as_percent(e2e_summary["end_to_end_success_rate"]))
    col2.metric("Intent Accuracy", as_percent(e2e_summary["intent_accuracy"]))
    col3.metric("Entity Match", as_percent(e2e_summary["entity_exact_match_rate"]))

    col4, col5 = st.columns(2)
    col4.metric("Validation Pass", as_percent(e2e_summary["validation_pass_rate"]))
    col5.metric("Execution Success", as_percent(e2e_summary["execution_success_rate"]))

    pipeline_df = pd.DataFrame(
        [
            {"stage": "intent_accuracy", "rate": e2e_summary["intent_accuracy"]},
            {"stage": "entity_match", "rate": e2e_summary["entity_exact_match_rate"]},
            {"stage": "validation_pass", "rate": e2e_summary["validation_pass_rate"]},
            {"stage": "execution_success", "rate": e2e_summary["execution_success_rate"]},
            {"stage": "end_to_end_success", "rate": e2e_summary["end_to_end_success_rate"]},
        ]
    ).set_index("stage")

    st.markdown("Graph 7: End-to-end stage rates")
    st.line_chart(pipeline_df)

    pipeline_counts_df = pipeline_df.rename(columns={"rate": "count"}).copy()
    pipeline_counts_df["count"] = pipeline_counts_df["count"].apply(lambda rate: as_count(rate, sample_count))
    st.markdown("Graph 8: End-to-end stage counts")
    st.area_chart(pipeline_counts_df)

    stage_items = sorted(e2e_summary.get("stage_breakdown", {}).items(), key=lambda item: item[0])
    if stage_items:
        stage_df = pd.DataFrame(stage_items, columns=["stage", "count"]).set_index("stage")
        st.markdown("Graph 9: Stage breakdown")
        st.bar_chart(stage_df)
        render_pie_chart(
            title="Graph 10: Stage breakdown share",
            labels=stage_df.index.tolist(),
            values=stage_df["count"].tolist(),
        )

    e2e_success_count = as_count(e2e_summary["end_to_end_success_rate"], sample_count)
    e2e_failure_count = max(sample_count - e2e_success_count, 0)
    render_pie_chart(
        title="Graph 11: E2E success vs failure",
        labels=["success", "failure"],
        values=[e2e_success_count, e2e_failure_count],
        donut=True,
    )


def render_ner_section(ner_summary: dict) -> None:
    st.subheader("NER Evaluation")
    sample_count = int(ner_summary.get("sample_count", 0))
    entity_type_count = int(ner_summary.get("entity_type_count", 0))
    st.caption(f"Samples: {sample_count} | Entity types: {entity_type_count}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Micro Precision", as_percent(ner_summary.get("micro_precision", 0.0)))
    col2.metric("Micro Recall", as_percent(ner_summary.get("micro_recall", 0.0)))
    col3.metric("Micro F1", as_percent(ner_summary.get("micro_f1", 0.0)))
    col4.metric("Entity Exact Match", as_percent(ner_summary.get("entity_exact_match_rate", 0.0)))

    counts_df = pd.DataFrame(
        [
            {"metric": "expected_entities", "count": int(ner_summary.get("expected_entity_count", 0))},
            {"metric": "predicted_entities", "count": int(ner_summary.get("predicted_entity_count", 0))},
            {"metric": "true_positive", "count": int(ner_summary.get("true_positive_entities", 0))},
            {"metric": "false_positive", "count": int(ner_summary.get("false_positive_entities", 0))},
            {"metric": "false_negative", "count": int(ner_summary.get("false_negative_entities", 0))},
        ]
    ).set_index("metric")
    st.markdown("NER Graph 1: Entity volume and errors")
    st.bar_chart(counts_df)

    render_pie_chart(
        title="NER Graph 2: TP / FP / FN share",
        labels=["TP", "FP", "FN"],
        values=[
            int(ner_summary.get("true_positive_entities", 0)),
            int(ner_summary.get("false_positive_entities", 0)),
            int(ner_summary.get("false_negative_entities", 0)),
        ],
        donut=True,
    )

    per_entity_type_rows = ner_summary.get("per_entity_type", [])
    if per_entity_type_rows:
        per_entity_type_df = pd.DataFrame(per_entity_type_rows).set_index("entity_type")

        st.markdown("NER Graph 3: Per-entity precision / recall / F1")
        st.bar_chart(per_entity_type_df[["precision", "recall", "f1"]])

        st.markdown("NER Graph 4: Per-entity TP / FP / FN")
        st.bar_chart(per_entity_type_df[["tp", "fp", "fn"]])

        st.dataframe(
            per_entity_type_df.reset_index(),
            use_container_width=True,
            hide_index=True,
        )

    per_intent_rows = ner_summary.get("per_intent_entity_exact_match", [])
    if per_intent_rows:
        per_intent_df = pd.DataFrame(per_intent_rows).set_index("intent")

        st.markdown("NER Graph 5: Per-intent entity exact match")
        st.line_chart(per_intent_df[["exact_match_rate"]])

        st.markdown("NER Graph 6: Per-intent sample volume")
        st.bar_chart(per_intent_df[["samples"]])

        st.dataframe(
            per_intent_df.reset_index(),
            use_container_width=True,
            hide_index=True,
        )


def main() -> None:
    st.set_page_config(page_title="Desktop Assistant Evaluation", layout="wide")
    st.title("Desktop Assistant Evaluation Dashboard")
    st.caption("Run a simple evaluation and view key statistics.")

    with st.sidebar:
        st.header("Controls")
        scope = st.selectbox("Scope", options=["both", "intent", "e2e", "ner"], index=0)
        seed = st.number_input("Seed", min_value=0, value=42, step=1)
        save_reports = st.checkbox("Save reports to disk", value=False)
        generate_plot_files = st.checkbox("Generate plot files", value=False, disabled=not save_reports)
        show_raw = st.checkbox("Show raw JSON", value=False)
        run_clicked = st.button("Run Evaluation", type="primary", use_container_width=True)

    if not run_clicked:
        st.info("Choose options in the sidebar and click Run Evaluation.")
        return

    try:
        with st.spinner("Running evaluation..."):
            summary = run_evaluation(
                scope=scope,
                seed=int(seed),
                persist_reports=save_reports,
                no_plots=not generate_plot_files,
            )
    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")
        st.exception(exc)
        return

    st.success("Evaluation completed")

    reports_dir = summary.get("reports_dir")
    if reports_dir:
        st.info(f"Reports saved at: {reports_dir}")

    intent_summary = summary.get("intent")
    e2e_summary = summary.get("e2e")
    ner_summary = summary.get("ner")

    if intent_summary:
        render_intent_section(intent_summary)

    if e2e_summary:
        render_e2e_section(e2e_summary)

    if ner_summary:
        render_ner_section(ner_summary)

    if show_raw:
        st.subheader("Raw summary")
        st.json(summary)


if __name__ == "__main__":
    main()
