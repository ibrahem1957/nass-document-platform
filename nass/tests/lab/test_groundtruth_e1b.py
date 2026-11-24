from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from lab.core.groundtruth import GroundTruthError, GroundTruthStore
from lab.core.models import LabDocument
from lab.core.workspace import WorkspaceManager


def _document() -> LabDocument:
    return LabDocument(
        doc_id="doc_test",
        display_name="benchmark.pdf",
        kind="pdf",
        sha256="a" * 64,
        size_bytes=1,
        page_count=2,
        page_sizes=[(10, 10), (10, 10)],
        has_text_layer=[True, True],
        pdf_producer="Synthetic",
        pdf_creator="Synthetic",
        uploaded_at=datetime.now(timezone.utc),
    )


def test_import_benchmark_json_and_version_verified_edits(settings) -> None:
    manager = WorkspaceManager(settings)
    workspace = manager.create()
    store = GroundTruthStore(manager)
    payload = json.dumps(
        {
            "document": {"page_count": 2},
            "pages": [
                {"page_number": 1, "full_text_in_reading_order": "نص ١"},
                {"page_number": 2, "full_text_in_reading_order": "Text 2"},
            ],
        },
        ensure_ascii=False,
    )
    ground_truth = store.import_json(
        workspace.workspace_id,
        _document(),
        payload,
        verified=True,
        reviewer="source-review",
    )
    assert ground_truth.version == 1
    assert all(page.status == "verified" for page in ground_truth.pages)
    updated = store.update_page(
        workspace.workspace_id,
        "doc_test",
        1,
        "نص مصحح",
        verify=False,
        reviewer=None,
        notes="edited",
    )
    assert updated.version == 2
    assert updated.pages[0].status == "draft"
    assert updated.pages[0].verified_by is None


def test_import_keeps_curated_numeric_values_with_sources_and_duplicates(settings) -> None:
    manager = WorkspaceManager(settings)
    workspace = manager.create()
    store = GroundTruthStore(manager)
    payload = json.dumps(
        {
            "document": {"page_count": 2},
            "pages": [
                {
                    "page_number": 1,
                    "full_text_in_reading_order": "التقرير ٢٠٢٥ و ٢٠٢٥",
                    "numeric_values": [
                        {"block_id": "p1.h", "text": "٢٠٢٥"},
                        {"block_id": "p1.h", "text": "٢٠٢٥"},
                        {"block_id": "t1", "row": 0, "column": "الربع", "text": "١٤٢٫٦"},
                    ],
                },
                {"page_number": 2, "full_text_in_reading_order": "Text 2"},
            ],
        },
        ensure_ascii=False,
    )
    ground_truth = store.import_json(
        workspace.workspace_id, _document(), payload, verified=True, reviewer="source"
    )
    values = ground_truth.pages[0].numeric_values
    assert [(value.text, value.source) for value in values] == [
        ("٢٠٢٥", "text"),
        ("٢٠٢٥", "text"),
        ("١٤٢٫٦", "table"),
    ]
    assert values[2].row == 0 and values[2].column == "الربع"
    assert ground_truth.pages[1].numeric_values == []
    # Editing page text keeps the curated list (it is never inferred from text).
    edited = store.update_page(
        workspace.workspace_id, "doc_test", 1, "نص", verify=False, reviewer=None, notes=None
    )
    assert len(edited.pages[0].numeric_values) == 3


def test_import_rejects_numeric_value_without_exact_text(settings) -> None:
    manager = WorkspaceManager(settings)
    workspace = manager.create()
    store = GroundTruthStore(manager)
    payload = json.dumps(
        {
            "document": {"page_count": 2},
            "pages": [
                {
                    "page_number": 1,
                    "full_text_in_reading_order": "x",
                    "numeric_values": [{"block_id": "p1", "numeric_value": 5}],
                }
            ],
        }
    )
    with pytest.raises(GroundTruthError, match="exact text"):
        store.import_json(workspace.workspace_id, _document(), payload)


def test_import_rejects_wrong_page_count(settings) -> None:
    manager = WorkspaceManager(settings)
    workspace = manager.create()
    store = GroundTruthStore(manager)
    payload = json.dumps({"document": {"page_count": 8}, "pages": []})
    with pytest.raises(GroundTruthError, match="page count"):
        store.import_json(workspace.workspace_id, _document(), payload)
