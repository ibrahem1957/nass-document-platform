from __future__ import annotations

import streamlit as st

from lab.app.components import text_html
from lab.core.groundtruth import GroundTruthError, GroundTruthStore
from lab.core.ingest import IngestService
from lab.core.models import RunResult
from lab.core.settings import load_settings
from lab.core.workspace import WorkspaceManager


st.set_page_config(page_title="Ground Truth — Extraction Lab", layout="wide")
settings = load_settings()
manager = WorkspaceManager(settings)
if "workspace_id" not in st.session_state:
    st.warning("Create a workspace on Home first.")
    st.stop()
workspace_id = st.session_state.workspace_id
ingest = IngestService(manager, settings)
store = GroundTruthStore(manager)
documents = ingest.list_documents(workspace_id)

st.title("Ground Truth")
if not documents:
    st.info("Upload a document first.")
    st.stop()
labels = {f"{doc.display_name} ({doc.doc_id})": doc for doc in documents}
document = labels[st.selectbox("Document", list(labels))]
uploaded = st.file_uploader(
    "Import benchmark JSON or form-feed TXT", type=["json", "txt"]
)
verify_import = False
import_reviewer = ""
if uploaded is not None:
    # Import-only options: shown only while a file is being imported, so they cannot be
    # mistaken for the per-page "Mark this page verified" control below.
    verify_import = st.checkbox(
        "Import: mark ALL imported pages verified (only after visual/source review)",
        key=f"gt-import-verify-{document.doc_id}",
    )
    import_reviewer = st.text_input("Import reviewer", key=f"gt-import-reviewer-{document.doc_id}")
if uploaded is not None and st.button("Import Ground Truth"):
    try:
        if uploaded.name.lower().endswith(".json"):
            store.import_json(
                workspace_id,
                document,
                uploaded.getvalue(),
                verified=verify_import,
                reviewer=import_reviewer or None,
            )
        else:
            store.import_text(
                workspace_id, document, uploaded.getvalue().decode("utf-8")
            )
        st.session_state["gt_flash"] = "Ground Truth imported locally."
        st.rerun()
    except (GroundTruthError, UnicodeDecodeError) as exc:
        st.error(str(exc))

ground_truth = store.load(workspace_id, document.doc_id)
if ground_truth is None:
    run_files = sorted((manager.path(workspace_id) / "runs").glob("run_*.json"))
    runs = []
    for path in run_files:
        run = RunResult.model_validate_json(path.read_text(encoding="utf-8"))
        if run.doc_id == document.doc_id:
            runs.append(run)
    if runs:
        by_label = {f"{run.adapter.label} — {run.run_id}": run for run in runs}
        selected = by_label[st.selectbox("Seed draft from run", list(by_label))]
        if st.button("Seed draft"):
            store.seed_from_run(workspace_id, document, selected)
            st.rerun()
    st.info("Import or seed Ground Truth to begin review.")
    st.stop()

flash = st.session_state.pop("gt_flash", None)
if flash:
    st.success(flash)

verified_count = sum(page.status == "verified" for page in ground_truth.pages)
history = store.versions(workspace_id, document.doc_id)
st.caption(
    f"Version {ground_truth.version} · verified "
    f"{verified_count}/{len(ground_truth.pages)} pages"
    + (f" · {len(history)} versions kept in history" if history else "")
)
page_number = st.selectbox("Page", [page.page for page in ground_truth.pages])
page = next(item for item in ground_truth.pages if item.page == page_number)
left, right = st.columns(2)
left.image(
    ingest.render_page(workspace_id, document.doc_id, page_number), width="stretch"
)
right.markdown("**Current reference**")
right.markdown(text_html(page.text), unsafe_allow_html=True)

if page.status == "verified":
    when = page.verified_at.strftime("%Y-%m-%d %H:%M UTC") if page.verified_at else "unknown time"
    st.info(f"Page {page_number} is VERIFIED (by {page.verified_by or 'unnamed reviewer'}, {when}).")
else:
    st.warning(f"Page {page_number} is a DRAFT: metrics will not use it until it is verified.")

# One form per document/page/version: the text, the checkbox and Save are submitted
# together, and widget state can never carry over to another page or a newer version.
form_key = f"gt-{document.doc_id}-p{page_number}-v{ground_truth.version}"
with st.form(form_key):
    edited = st.text_area(
        "Visible text in logical reading order", page.text, height=420, key=f"{form_key}-text"
    )
    verified = st.checkbox(
        f"Mark page {page_number} verified",
        value=page.status == "verified",
        key=f"{form_key}-verified",
    )
    reviewer = st.text_input("Reviewer", value=page.verified_by or "", key=f"{form_key}-reviewer")
    notes = st.text_area("Notes", value=page.notes or "", key=f"{form_key}-notes")
    submitted = st.form_submit_button("Save page")
if submitted:
    saved = store.update_page(
        workspace_id,
        document.doc_id,
        page_number,
        edited,
        verify=verified,
        reviewer=reviewer or None,
        notes=notes or None,
    )
    saved_page = next(item for item in saved.pages if item.page == page_number)
    st.session_state["gt_flash"] = (
        f"Saved version {saved.version}: page {page_number} is now {saved_page.status.upper()} "
        f"(verified {sum(item.status == 'verified' for item in saved.pages)}/{len(saved.pages)} pages)."
    )
    st.rerun()
