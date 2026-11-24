from __future__ import annotations

import json

import streamlit as st

from lab.core.groundtruth import GroundTruthStore
from lab.core.metrics import (
    aggregate_metrics,
    compare_run_regressions,
    evaluate_run,
    finite_metric,
)
from lab.core.models import RunResult
from lab.core.review import ReviewStore
from lab.core.settings import load_settings
from lab.core.workspace import WorkspaceManager


st.set_page_config(page_title="Results — Extraction Lab", layout="wide")
settings = load_settings()
manager = WorkspaceManager(settings)
if "workspace_id" not in st.session_state:
    st.warning("Create a workspace on Home first.")
    st.stop()
workspace_id = st.session_state.workspace_id
gt_store = GroundTruthStore(manager)
review_store = ReviewStore(manager)

st.title("Results and local export")
runs = [
    RunResult.model_validate_json(path.read_text(encoding="utf-8"))
    for path in sorted((manager.path(workspace_id) / "runs").glob("run_*.json"))
]
if not runs:
    st.info("No extraction runs yet.")
    st.stop()
for run in runs:
    ground_truth = gt_store.load(workspace_id, run.doc_id)
    if ground_truth is None:
        continue
    page_metrics = evaluate_run(run, ground_truth)
    if not page_metrics:
        continue
    aggregate = aggregate_metrics(
        page_metrics, run.run_id, run.variant, ground_truth.version
    )
    st.subheader(f"{run.adapter.label} — {run.variant}")
    st.text(
        f"Verified pages: {aggregate.verified_pages}\n"
        f"CER: {aggregate.char_edits}/{aggregate.ref_chars}="
        f"{aggregate.cer:.4f}\n"
        f"WER: {aggregate.word_edits}/{aggregate.ref_words}="
        f"{aggregate.wer:.4f}\n"
        f"CER no diacritics: {aggregate.cer_no_diacritics:.4f}\n"
        f"Bag-of-words F1: {aggregate.bow_f1:.4f}\n"
        f"Numbers item: {aggregate.num_correct}/{aggregate.num_ref_count}="
        f"{finite_metric(aggregate.num_recall)} (hypothesis "
        f"{aggregate.num_hyp_count}); page mean "
        f"{finite_metric(aggregate.num_page_mean_accuracy)} over "
        f"{aggregate.num_pages_with_items} pages (running-text regex)\n"
        f"Curated GT numeric values: running text "
        f"{aggregate.curated_num_text_correct}/{aggregate.curated_num_text_ref}; "
        f"tables {aggregate.curated_num_table_correct}/"
        f"{aggregate.curated_num_table_ref}; unmatched hypothesis numbers "
        f"{aggregate.curated_num_unmatched_hyp}\n"
        f"Dates item: {aggregate.date_correct}/{aggregate.date_ref_count}="
        f"{finite_metric(aggregate.date_accuracy)}; page mean "
        f"{finite_metric(aggregate.date_page_mean_accuracy)} over "
        f"{aggregate.date_pages_with_items} pages\n"
        f"Identifiers item: {aggregate.identifier_correct}/"
        f"{aggregate.identifier_ref_count}="
        f"{finite_metric(aggregate.identifier_accuracy)}; page mean "
        f"{finite_metric(aggregate.identifier_page_mean_accuracy)} over "
        f"{aggregate.identifier_pages_with_items} pages\n"
        f"Operators item: {aggregate.operator_correct}/"
        f"{aggregate.operator_ref_count}="
        f"{finite_metric(aggregate.comparison_operator_accuracy)}; page mean "
        f"{finite_metric(aggregate.operator_page_mean_accuracy)} over "
        f"{aggregate.operator_pages_with_items} pages\n"
        f"Reading-order tau: {finite_metric(aggregate.reading_order_tau)}; "
        f"matched lines {aggregate.reading_order_matched_lines}/"
        f"{aggregate.reading_order_ref_lines}; tau pages "
        f"{aggregate.reading_order_pages_with_tau}/{aggregate.verified_pages}\n"
        f"Unreadable characters: {aggregate.unreadable_chars}"
    )
    if run.raw_run_id:
        raw_run = next((item for item in runs if item.run_id == run.raw_run_id), None)
        if raw_run is not None:
            regressions = compare_run_regressions(raw_run, run, ground_truth)
            st.text(
                "Correct RAW → incorrect CLEANED / newly introduced wrong\n"
                + "\n".join(
                    f"{category}: regressed={values['regressed']}; "
                    f"new_wrong={values['introduced_wrong']}; "
                    f"fixed={values['fixed']}; reference={values['reference']}"
                    for category, values in regressions.items()
                )
            )

reviews = review_store.list(workspace_id)
st.subheader("Human reviews")
review_lines = [
    f"{review.run_id} page={review.page} rating={review.rating} "
    f"issues={','.join(review.issues) or 'none'}"
    for review in reviews
]
st.text("\n".join(review_lines) or "No human reviews saved.")
st.download_button(
    "Download reviews JSON",
    json.dumps(
        [review.model_dump(mode="json") for review in reviews],
        ensure_ascii=False,
        indent=2,
    ),
    file_name="reviews.json",
    mime="application/json",
)
st.caption("Original PDFs and extracted text are excluded from this export.")
