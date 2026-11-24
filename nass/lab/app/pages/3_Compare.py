from __future__ import annotations

import json
from datetime import datetime, timezone

import streamlit as st

from lab.app.components import diff_html, overlay_blocks, runs_for_page_set, text_html
from lab.core.adapters.postprocess import (
    VARIANT_VERSION,
    available_variants,
    correct_text,
    derive_variant,
    detect_sensitive_tokens,
)
from lab.core.groundtruth import GroundTruthStore
from lab.core.ingest import IngestService
from lab.core.metrics import compare_text_regressions, evaluate_page, finite_metric
from lab.core.models import HumanReview, RunResult
from lab.core.review import ReviewStore
from lab.core.settings import load_settings
from lab.core.workspace import WorkspaceManager


st.set_page_config(page_title="Compare — Extraction Lab", layout="wide")
settings = load_settings()
manager = WorkspaceManager(settings)
if "workspace_id" not in st.session_state:
    st.warning("Create a workspace on Home first.")
    st.stop()
workspace_id = st.session_state.workspace_id
ingest = IngestService(manager, settings)
gt_store = GroundTruthStore(manager)
review_store = ReviewStore(manager)

st.title("Compare RAW and corrected extraction")
run_files = sorted((manager.path(workspace_id) / "runs").glob("run_*.json"))
runs = [RunResult.model_validate_json(path.read_text(encoding="utf-8")) for path in run_files]
raw_runs = [run for run in runs if run.variant == "raw" and run.raw_run_id is None]
if not raw_runs:
    st.info("Run text extraction first.")
    st.stop()

doc_ids = sorted({run.doc_id for run in raw_runs})
doc_id = st.selectbox("Document", doc_ids)
document = ingest.get(workspace_id, doc_id)
doc_runs = [run for run in raw_runs if run.doc_id == doc_id]
page_sets = sorted({tuple(page.page for page in run.pages) for run in doc_runs})
selected_page_set = st.selectbox(
    "Run page set", page_sets, format_func=lambda value: ", ".join(map(str, value))
)
if selected_page_set is None:
    st.stop()
show_all_runs = st.checkbox("Show all runs, including older duplicates")
doc_runs = runs_for_page_set(doc_runs, selected_page_set, show_all=show_all_runs)
extractor_labels = [run.adapter.label for run in doc_runs]
selected_extractors = st.multiselect(
    "Extractors", extractor_labels, default=extractor_labels
)
doc_runs = [run for run in doc_runs if run.adapter.label in selected_extractors]
page_number = st.selectbox("Page", list(selected_page_set))
if page_number is None:
    st.stop()
show_overlay = st.checkbox("Show block-order overlay", value=True)
monospace = st.checkbox("Monospace text")
show_invisibles = st.checkbox("Show invisible and presentation-form markers")
show_experimental = st.checkbox(
    "Show experimental unsafe correction variants",
    value=False,
    help=(
        "digit_run_fix and bidi_reorder are hidden by default because benchmark "
        "regression checks found newly introduced sensitive values."
    ),
)

image_path = ingest.render_page(workspace_id, doc_id, page_number)
st.subheader("Original page")
st.image(image_path, width="stretch")
ground_truth = gt_store.load(workspace_id, doc_id)
gt_page = (
    next((item for item in ground_truth.pages if item.page == page_number), None)
    if ground_truth
    else None
)

for run in doc_runs:
    st.divider()
    st.subheader(run.adapter.label)
    st.caption(
        f"Params: {json.dumps(run.adapter.params, ensure_ascii=False, sort_keys=True)} · "
        f"extraction: {run.wall_ms:.2f} ms · startup: {run.startup_ms:.2f} ms · "
        f"overlay: {run.overlay_ms:.2f} ms · cache_hit={run.cache_hit}"
    )
    page_result = next((item for item in run.pages if item.page == page_number), None)
    if page_result is None:
        st.warning("No result for this page")
        continue
    if show_overlay and page_result.blocks:
        st.image(
            overlay_blocks(image_path, document, run, page_number),
            caption="Numbered extraction blocks",
            width="stretch",
        )
    variant = st.selectbox(
        f"Correction variant — {run.adapter.label}",
        available_variants(
            run.adapter.adapter_id, include_experimental=show_experimental
        ),
        key=f"variant_{run.run_id}_{page_number}",
    )
    corrected = correct_text(page_result.text, run.adapter.adapter_id, variant)
    raw_column, cleaned_column = st.columns(2)
    raw_column.markdown("**RAW — exactly as extracted**")
    raw_column.markdown(
        text_html(
            page_result.text,
            monospace=monospace,
            show_invisibles=show_invisibles,
        ),
        unsafe_allow_html=True,
    )
    cleaned_column.markdown(f"**CLEANED — {variant} v{VARIANT_VERSION}**")
    cleaned_column.markdown(
        text_html(
            corrected.text,
            monospace=monospace,
            show_invisibles=show_invisibles,
        ),
        unsafe_allow_html=True,
    )
    st.markdown("**RAW → CLEANED diff**")
    st.markdown(diff_html(page_result.text, corrected.text), unsafe_allow_html=True)
    st.text(f"Changed spans: {len(corrected.changes)}")
    raw_tokens = detect_sensitive_tokens(page_result.text)
    cleaned_tokens = detect_sensitive_tokens(corrected.text)
    st.text(
        "Detected numeric tokens and identifiers\n"
        f"RAW: {[(token.kind, token.value) for token in raw_tokens]}\n"
        f"CLEANED: {[(token.kind, token.value) for token in cleaned_tokens]}"
    )
    for warning in corrected.warnings:
        st.warning(warning)
    if corrected.changes:
        st.text(
            "\n".join(
                f"{change.kind} [{change.confidence}]: "
                f"{change.before!r} -> {change.after!r}"
                for change in corrected.changes[:25]
            )
        )
    if gt_page is not None:
        st.markdown("**Ground Truth**")
        st.markdown(text_html(gt_page.text), unsafe_allow_html=True)
        if gt_page.status == "verified":
            cleaned_page = page_result.model_copy(update={"text": corrected.text})
            raw_metric = evaluate_page(
                run.run_id, "raw", page_result, gt_page, ground_truth.version
            )
            clean_metric = evaluate_page(
                run.run_id, variant, cleaned_page, gt_page, ground_truth.version
            )
            regressions = compare_text_regressions(
                gt_page.text, page_result.text, corrected.text
            )
            st.text(
                "Metric             RAW       CLEANED\n"
                f"CER edits/ref      {raw_metric.char_edits}/{raw_metric.ref_chars}"
                f"={raw_metric.cer:.4f}    {clean_metric.char_edits}/"
                f"{clean_metric.ref_chars}={clean_metric.cer:.4f}\n"
                f"WER edits/ref      {raw_metric.word_edits}/{raw_metric.ref_words}"
                f"={raw_metric.wer:.4f}    {clean_metric.word_edits}/"
                f"{clean_metric.ref_words}={clean_metric.wer:.4f}\n"
                f"Numbers item       {raw_metric.num_correct}/"
                f"{raw_metric.num_ref_count}    {clean_metric.num_correct}/"
                f"{clean_metric.num_ref_count}\n"
                f"Dates item         {raw_metric.date_correct}/"
                f"{raw_metric.date_ref_count}    {clean_metric.date_correct}/"
                f"{clean_metric.date_ref_count}\n"
                f"Identifiers item   {raw_metric.identifier_correct}/"
                f"{raw_metric.identifier_ref_count}    "
                f"{clean_metric.identifier_correct}/"
                f"{clean_metric.identifier_ref_count}\n"
                f"Operators item     {raw_metric.operator_correct}/"
                f"{raw_metric.operator_ref_count}    "
                f"{clean_metric.operator_correct}/"
                f"{clean_metric.operator_ref_count}\n"
                f"Order tau/coverage {finite_metric(raw_metric.reading_order_tau)} "
                f"({raw_metric.reading_order_matched_lines}/"
                f"{raw_metric.reading_order_ref_lines} lines)    "
                f"{finite_metric(clean_metric.reading_order_tau)} "
                f"({clean_metric.reading_order_matched_lines}/"
                f"{clean_metric.reading_order_ref_lines} lines)"
            )
            st.text(
                "Sensitive-value regressions (numbers/dates/IDs/operators)\n"
                + "\n".join(
                    f"{category}: correct RAW {values['correct_raw']}/"
                    f"{values['reference']}; correct CLEANED "
                    f"{values['correct_cleaned']}/{values['reference']}; "
                    f"regressed={values['regressed']}; fixed={values['fixed']}; "
                    f"new_wrong={values['introduced_wrong']}"
                    for category, values in regressions.items()
                )
            )
        else:
            st.info("Verify ground truth to compute accuracy.")
    if variant != "raw" and st.button(
        f"Save derived {variant} run",
        key=f"save_{run.run_id}_{page_number}_{variant}",
    ):
        derived = derive_variant(run, variant)
        WorkspaceManager.atomic_write_json(
            manager.path(workspace_id) / "runs" / f"{derived.run_id}.json",
            derived.model_dump(mode="json"),
        )
        st.success(f"Saved {derived.run_id}; RAW {run.run_id} was not modified.")
    with st.expander(f"Human review — {run.adapter.label}"):
        rating = st.selectbox(
            "Rating",
            ["correct", "minor_issues", "major_issues", "unusable"],
            key=f"rating_{run.run_id}",
        )
        issues = st.multiselect(
            "Issues",
            [
                "reversed_order",
                "wrong_column_order",
                "broken_letters",
                "missing_text",
                "extra_text",
                "numbers_changed",
                "diacritics_lost",
                "header_footer_noise",
                "merged_words",
                "split_words",
                "other",
            ],
            key=f"issues_{run.run_id}",
        )
        notes = st.text_area("Notes", key=f"notes_{run.run_id}")
        reviewer = st.text_input("Reviewer", key=f"reviewer_{run.run_id}")
        if st.button("Save review", key=f"review_{run.run_id}"):
            review_store.save(
                workspace_id,
                HumanReview(
                    run_id=run.run_id,
                    page=page_number,
                    rating=rating,
                    issues=issues,
                    notes=notes or None,
                    reviewer=reviewer or None,
                    created_at=datetime.now(timezone.utc),
                ),
            )
            st.success("Review saved locally.")

st.warning(
    "Corrections are derived artifacts. RAW is immutable. "
    "Low/medium-confidence changes require visual review."
)
