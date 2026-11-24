from __future__ import annotations

import hashlib
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from docprep.inspector import InspectorError, create_server, load_registry


def _prepared(text: str = "نص عربي <script>alert(1)</script>", *, geometry: str = "page_level_only") -> dict:
    return {
        "schema_version": "1.0.0",
        "kind": "docprep_prepared_document",
        "document_id": "document-run_test",
        "source_run_id": "run_test",
        "source_document": {},
        "language": None,
        "page_count": 2,
        "metadata": {
            "title": "عنوان عربي", "title_source_section_id": "s1", "author": None,
            "author_status": "unavailable", "source_filename": "fixture.pdf", "source_run_id": "run_test",
            "language": None, "language_status": "unavailable", "page_count": 2, "annotations": [],
        },
        "pages": [{"page": 1}, {"page": 2}],
        "sections": [{
            "section_id": "s1", "title": "القسم", "level": 0, "parent": None, "children": [],
            "heading_element_ids": ["e1"], "page_start": 1, "page_end": 2, "confidence": {},
            "element_ids_in_reading_order": ["e1"], "unit_ids_in_reading_order": ["e1"],
            "assignment_confidences": {"e1": "high"}, "quality_finding_ids": ["q1"],
        }],
        "elements": [{
            "element_id": "e1", "original_type": "page_text", "text": text, "page": 1,
            "reading_order": 0, "document_order": 0, "bbox_pt": [0, 0, 100, 100],
            "geometry_reliability": geometry, "validation_state": "needs_review",
            "source_extraction": {"source": "fixture"}, "structure_assignments": [],
            "section_ids": ["s1"], "roles": ["heading"], "relationship_ids": ["r1"],
            "quality_finding_ids": ["q1"], "quality_inventory": {},
        }],
        "derived_spans": {"structure": [], "relationships": []},
        "lists": [],
        "visuals": [{
            "visual_id": "e1", "visual_type": "figure", "page": 1,
            "validation_state": "needs_review", "original_content": text,
            "relationship_ids": ["r1"], "quality_finding_ids": ["q1"], "layout_provenance": None,
        }],
        "logical_tables": [{
            "logical_table_id": "lt1", "source_part_ids": ["e1"], "pages": [1],
            "repeated_header_between_parts": [], "pair_evidence": [], "relationship_ids": [],
            "caption_relationship_ids": [], "quality_finding_ids": ["q1"], "provenance": {},
        }],
        "nodes": [{
            "node_id": "e1", "node_type": "text", "page": 1, "text": text, "source_id": "e1",
            "start": None, "end": None, "provenance": {},
        }],
        "relationships": [{
            "rel_id": "r1", "type": "describes", "source_id": "e1", "target_id": "e1",
            "evidence": {"reason": "fixture"}, "confidence": "low", "method": "fixture",
            "conflicts": [], "review_recommended": True, "limitations": ["synthetic"],
        }],
        "quality": {
            "findings": [{
                "finding_id": "q1", "category": "ocr_corruption", "subtype": "possible_error",
                "severity": "medium", "confidence": "low", "source_ids": ["e1"],
                "page_numbers": [1], "evidence": {"text": text}, "explanation": "Unverified fixture",
                "suggested_review_action": "Inspect", "status": "needs_review",
            }],
            "inventory": [], "summary": {}, "validation": {},
        },
        "indexes": {
            "element_by_id": {"e1": 0}, "section_by_id": {"s1": 0}, "list_by_id": {},
            "logical_table_by_id": {"lt1": 0}, "elements_by_section": {"s1": [0]},
            "visuals_by_describing_source": {}, "quality_findings_by_element": {"e1": ["q1"]},
            "relationships_by_node": {"e1": ["r1"]},
        },
        "provenance": {
            "component": "Unified Prepared Document Builder", "docprep_version": "test",
            "output_schema_version": "1.0.0", "processing_configuration": {},
            "processing_configuration_sha256": "0" * 64, "input_directories": {},
            "input_sha256": {}, "upstream_manifests": {}, "content_policy": "verbatim",
        },
    }


def _write_fixture(root: Path, document_id: str = "one", *, pdf: bool = True,
                   text: str = "نص عربي <script>alert(1)</script>", geometry: str = "page_level_only") -> tuple[Path, Path]:
    prepared_dir = root / f"prepared-{document_id}"
    nass_dir = root / f"nass-{document_id}"
    prepared_dir.mkdir()
    nass_dir.mkdir()
    prepared = _prepared(text, geometry=geometry)
    (prepared_dir / "prepared_document.json").write_text(
        json.dumps(prepared, ensure_ascii=False), encoding="utf-8",
    )
    validation = {
        "schema_version": 1, "kind": "docprep_prepared_document_validation",
        "source_run_id": "run_test", "summary": {"ok": 15, "error": 0}, "checks": [],
    }
    (prepared_dir / "prepared_document_validation.json").write_text(
        json.dumps(validation), encoding="utf-8",
    )
    pdf_bytes = b"%PDF-1.4\n% synthetic inspector fixture\n%%EOF\n"
    manifest = {
        "run_id": "run_test",
        "document": {"display_name": "fixture.pdf", "bytes": len(pdf_bytes),
                     "sha256": hashlib.sha256(pdf_bytes).hexdigest()},
    }
    (nass_dir / "execution_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if pdf:
        (nass_dir / "verified-source.pdf").write_bytes(pdf_bytes)
    return prepared_dir / "prepared_document.json", nass_dir


def _write_config(root: Path, specifications: list[tuple[str, str, Path, Path]]) -> Path:
    path = root / "inspector.json"
    value = {
        "schema_version": 1,
        "kind": "docprep_inspector_documents",
        "documents": [
            {"id": item_id, "label": label, "prepared_path": str(prepared), "nass_dir": str(nass)}
            for item_id, label, prepared, nass in specifications
        ],
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@contextmanager
def _running_server(registry):
    server = create_server(registry, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _get(url: str) -> tuple[int, dict[str, str], bytes]:
    with urlopen(url, timeout=5) as response:
        return response.status, dict(response.headers), response.read()


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def test_valid_prepared_document_loading_and_pdf_identity(tmp_path):
    prepared, nass = _write_fixture(tmp_path)
    registry = load_registry(_write_config(tmp_path, [("one", "One", prepared, nass)]))
    record = registry["one"]
    assert record.summary()["element_count"] == 1
    assert record.summary()["pdf_available"] is True
    assert record.pdf_path == (nass / "verified-source.pdf").resolve()


def test_incompatible_prepared_schema_fails_clearly(tmp_path):
    prepared, nass = _write_fixture(tmp_path)
    value = json.loads(prepared.read_text(encoding="utf-8"))
    value["schema_version"] = "2.0.0"
    prepared.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(InspectorError, match="incompatible prepared schema_version"):
        load_registry(_write_config(tmp_path, [("one", "One", prepared, nass)]))


def test_broken_relationship_endpoint_is_rejected(tmp_path):
    prepared, nass = _write_fixture(tmp_path)
    value = json.loads(prepared.read_text(encoding="utf-8"))
    value["relationships"][0]["target_id"] = "missing"
    prepared.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(InspectorError, match="unresolved endpoints"):
        load_registry(_write_config(tmp_path, [("one", "One", prepared, nass)]))


def test_document_selection_and_multiple_document_isolation(tmp_path):
    first, first_nass = _write_fixture(tmp_path, "first", text="first")
    second, second_nass = _write_fixture(tmp_path, "second", text="second")
    registry = load_registry(_write_config(tmp_path, [
        ("first", "First", first, first_nass), ("second", "Second", second, second_nass),
    ]))
    with _running_server(registry) as base:
        _, _, body = _get(f"{base}/api/documents")
        assert [item["id"] for item in json.loads(body)["documents"]] == ["first", "second"]
        _, _, first_body = _get(f"{base}/api/documents/first")
        _, _, second_body = _get(f"{base}/api/documents/second")
        assert json.loads(first_body)["document"]["elements"][0]["text"] == "first"
        assert json.loads(second_body)["document"]["elements"][0]["text"] == "second"
        with pytest.raises(HTTPError) as error:
            _get(f"{base}/api/documents/unknown")
        assert error.value.code == 404


def test_page_section_element_and_quality_data_are_available(tmp_path):
    prepared, nass = _write_fixture(tmp_path)
    registry = load_registry(_write_config(tmp_path, [("one", "One", prepared, nass)]))
    with _running_server(registry) as base:
        _, _, body = _get(f"{base}/api/documents/one")
    document = json.loads(body)["document"]
    assert document["page_count"] == 2
    assert document["sections"][0]["element_ids_in_reading_order"] == ["e1"]
    assert document["elements"][0]["section_ids"] == ["s1"]
    assert document["quality"]["findings"][0]["status"] == "needs_review"


def test_logical_table_navigation_data_is_preserved(tmp_path):
    prepared, nass = _write_fixture(tmp_path)
    registry = load_registry(_write_config(tmp_path, [("one", "One", prepared, nass)]))
    table = registry["one"].prepared["logical_tables"][0]
    assert table["source_part_ids"] == ["e1"]
    assert table["pages"] == [1]


def test_missing_pdf_has_structured_data_fallback(tmp_path):
    prepared, nass = _write_fixture(tmp_path, pdf=False)
    registry = load_registry(_write_config(tmp_path, [("one", "One", prepared, nass)]))
    assert registry["one"].pdf_path is None
    with _running_server(registry) as base:
        _, _, body = _get(f"{base}/api/documents/one")
        assert json.loads(body)["summary"]["pdf_available"] is False
        with pytest.raises(HTTPError) as error:
            _get(f"{base}/api/documents/one/pdf")
        assert error.value.code == 404


def test_pdf_range_requests_support_native_browser_renderer(tmp_path):
    prepared, nass = _write_fixture(tmp_path)
    registry = load_registry(_write_config(tmp_path, [("one", "One", prepared, nass)]))
    with _running_server(registry) as base:
        request = Request(f"{base}/api/documents/one/pdf", headers={"Range": "bytes=0-7"})
        with urlopen(request, timeout=5) as response:
            assert response.status == 206
            assert response.headers["Content-Range"].startswith("bytes 0-7/")
            assert response.read() == b"%PDF-1.4"


def test_missing_geometry_is_page_navigation_only(tmp_path):
    prepared, nass = _write_fixture(tmp_path, geometry="page_level_only")
    registry = load_registry(_write_config(tmp_path, [("one", "One", prepared, nass)]))
    assert registry["one"].prepared["elements"][0]["geometry_reliability"] == "page_level_only"
    script = Path("docprep/inspector_static/app.js").read_text(encoding="utf-8")
    assert "Page synchronized without a bounding box" in script
    assert "does not draw overlays" in script


def test_arabic_rtl_and_xss_safe_dom_rendering(tmp_path):
    original = "نص عربي <script>alert(1)</script>"
    prepared, nass = _write_fixture(tmp_path, text=original)
    registry = load_registry(_write_config(tmp_path, [("one", "One", prepared, nass)]))
    assert registry["one"].prepared["elements"][0]["text"] == original
    script = Path("docprep/inspector_static/app.js").read_text(encoding="utf-8")
    style = Path("docprep/inspector_static/app.css").read_text(encoding="utf-8")
    assert ".innerHTML" not in script
    assert 'text.dir = "auto"' in script
    assert "unicode-bidi: plaintext" in style


def test_csp_and_json_content_type_prevent_html_execution(tmp_path):
    prepared, nass = _write_fixture(tmp_path)
    registry = load_registry(_write_config(tmp_path, [("one", "One", prepared, nass)]))
    with _running_server(registry) as base:
        _, headers, body = _get(f"{base}/api/documents/one")
    assert headers["Content-Type"].startswith("application/json")
    assert "script-src 'self'" in headers["Content-Security-Policy"]
    assert json.loads(body)["document"]["elements"][0]["text"].startswith("نص عربي")


@pytest.mark.parametrize("path", [
    "/../config/inspector_documents.json",
    "/%2e%2e/config/inspector_documents.json",
    "/api/documents/one/%5c..%5cprepared_document.json",
])
def test_path_traversal_is_rejected(tmp_path, path):
    prepared, nass = _write_fixture(tmp_path)
    registry = load_registry(_write_config(tmp_path, [("one", "One", prepared, nass)]))
    with _running_server(registry) as base:
        with pytest.raises(HTTPError) as error:
            _get(base + path)
        assert error.value.code == 404


def test_server_requests_are_read_only(tmp_path):
    prepared, nass = _write_fixture(tmp_path)
    config = _write_config(tmp_path, [("one", "One", prepared, nass)])
    registry = load_registry(config)
    before = _hash_tree(tmp_path)
    with _running_server(registry) as base:
        _get(f"{base}/api/documents")
        _get(f"{base}/api/documents/one")
        _get(f"{base}/api/documents/one/pdf")
        request = Request(f"{base}/api/documents/one", data=b"{}", method="POST")
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=5)
        assert error.value.code == 405
    assert _hash_tree(tmp_path) == before


def test_ui_contains_navigation_and_filter_controls():
    html = Path("docprep/inspector_static/index.html").read_text(encoding="utf-8")
    script = Path("docprep/inspector_static/app.js").read_text(encoding="utf-8")
    for control in ("previousPage", "nextPage", "pageInput", "sectionTree", "severityFilter", "reviewFilter"):
        assert f'id="{control}"' in html
    assert "selectSection" in script
    assert "selectElement" in script
    assert "finding.status === \"needs_review\"" in script


def test_static_and_api_security_headers(tmp_path):
    prepared, nass = _write_fixture(tmp_path)
    registry = load_registry(_write_config(tmp_path, [("one", "One", prepared, nass)]))
    with _running_server(registry) as base:
        _, headers, body = _get(base + "/")
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Cache-Control"] == "no-store"
    assert b"DocPrep Document Inspector" in body


def test_configuration_rejects_unsafe_and_duplicate_ids(tmp_path):
    prepared, nass = _write_fixture(tmp_path)
    with pytest.raises(InspectorError, match="unsafe document id"):
        load_registry(_write_config(tmp_path, [("../bad", "Bad", prepared, nass)]))
    with pytest.raises(InspectorError, match="duplicate configured document id"):
        load_registry(_write_config(tmp_path, [
            ("same", "First", prepared, nass), ("same", "Second", prepared, nass),
        ]))
