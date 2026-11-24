"""Regression tests: marking a Ground Truth page verified is saved, counted and shown."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fitz
import pytest
from streamlit.testing.v1 import AppTest

from lab.core.groundtruth import GroundTruthError, GroundTruthStore
from lab.core.ingest import IngestService
from lab.core.workspace import WorkspaceManager

PAGE = Path(__file__).resolve().parents[2] / "lab" / "app" / "pages" / "4_Ground_Truth.py"


def _eight_page_workspace(settings, tmp_path):
    manager = WorkspaceManager(settings)
    workspace = manager.create()
    pdf = tmp_path / "eight.pdf"
    document = fitz.open()
    for number in range(1, 9):
        document.new_page().insert_text((72, 72), f"page {number}")
    document.save(pdf)
    with pdf.open("rb") as handle:
        doc = IngestService(manager, settings).ingest(workspace.workspace_id, handle, pdf.name)
    payload = {
        "pages": [
            {"page_number": number, "full_text_in_reading_order": f"page {number}"}
            for number in range(1, 9)
        ]
    }
    store = GroundTruthStore(manager)
    store.import_json(workspace.workspace_id, doc, json.dumps(payload))
    return manager, store, workspace.workspace_id, doc


def _app(workspace_id: str) -> AppTest:
    app = AppTest.from_file(str(PAGE), default_timeout=60)
    app.session_state["workspace_id"] = workspace_id
    return app.run()


def _caption(app: AppTest) -> str:
    return next(item.value for item in app.caption if item.value.startswith("Version"))


def _page_checkbox(app: AppTest, number: int):
    return next(item for item in app.checkbox if item.label == f"Mark page {number} verified")


def test_marking_page_one_verified_shows_one_of_eight_immediately(
    settings, tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("LAB_TMP_DIR", settings.tmp_dir)
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    _manager, store, workspace_id, doc = _eight_page_workspace(settings, tmp_path)
    app = _app(workspace_id)
    assert "verified 0/8 pages" in _caption(app)
    # The import-only option is hidden when no file is being imported.
    assert not any("imported pages" in item.label for item in app.checkbox)

    _page_checkbox(app, 1).check()
    next(item for item in app.text_input if item.label == "Reviewer").input("Reviewer A")
    next(button for button in app.button if button.label == "Save page").click().run()

    # The very next screen already reflects the save: no stale count.
    assert "Version 2 · verified 1/8 pages" in _caption(app)
    assert any("page 1 is now VERIFIED" in item.value for item in app.success)
    assert any("Page 1 is VERIFIED" in item.value for item in app.info)
    stored = store.load(workspace_id, doc.doc_id)
    assert stored.version == 2
    assert stored.pages[0].status == "verified" and stored.pages[0].verified_by == "Reviewer A"
    assert all(page.status == "draft" for page in stored.pages[1:])


def test_verification_state_never_leaks_to_another_page(settings, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LAB_TMP_DIR", settings.tmp_dir)
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    _manager, store, workspace_id, doc = _eight_page_workspace(settings, tmp_path)
    app = _app(workspace_id)
    _page_checkbox(app, 1).check()
    next(button for button in app.button if button.label == "Save page").click().run()
    page_select = next(item for item in app.selectbox if item.label == "Page")
    page_select.select(2).run()
    assert _page_checkbox(app, 2).value is False
    assert any("Page 2 is a DRAFT" in item.value for item in app.warning)
    # Saving page 2 unchanged keeps it a draft and keeps page 1 verified.
    next(button for button in app.button if button.label == "Save page").click().run()
    stored = store.load(workspace_id, doc.doc_id)
    assert [page.status for page in stored.pages[:2]] == ["verified", "draft"]
    assert "verified 1/8 pages" in _caption(app)


def test_every_version_is_kept_and_history_is_never_overwritten(settings, tmp_path) -> None:
    _manager, store, workspace_id, doc = _eight_page_workspace(settings, tmp_path)
    store.update_page(workspace_id, doc.doc_id, 1, "page 1", verify=True, reviewer="R", notes=None)
    store.update_page(workspace_id, doc.doc_id, 2, "page 2", verify=False, reviewer=None, notes="x")
    history = store.versions(workspace_id, doc.doc_id)
    assert [item.version for item in history] == [1, 2, 3]
    assert history[0].pages[0].status == "draft"
    assert history[1].pages[0].status == "verified"
    current = store.load(workspace_id, doc.doc_id)
    with pytest.raises(GroundTruthError, match="must increase"):
        store.save(workspace_id, current)
    with pytest.raises(GroundTruthError, match="refusing to overwrite history"):
        store._archive(
            workspace_id, current.model_copy(update={"version": 1, "source": "manual"})
        )


def test_version_saved_before_history_existed_is_archived_on_first_save(
    settings, tmp_path
) -> None:
    manager, store, workspace_id, doc = _eight_page_workspace(settings, tmp_path)
    # Simulate a pre-fix workspace: a current file with no history directory.
    history_dir = manager.path(workspace_id) / "gt" / "history" / doc.doc_id
    for path in history_dir.glob("*.json"):
        path.unlink()
    store.update_page(workspace_id, doc.doc_id, 1, "page 1", verify=True, reviewer="R", notes=None)
    assert [item.version for item in store.versions(workspace_id, doc.doc_id)] == [1, 2]
