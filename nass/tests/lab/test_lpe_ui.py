"""UI smoke test for the Local PDF Extraction Benchmark page (temporary results root)."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from lab.local_pdf_extraction import runner, storage
from tests.lab import lpe_pdf_factory as factory

PAGE = Path(__file__).resolve().parents[2] / "lab" / "app" / "pages" / "9_Local_PDF_Extraction_Benchmark.py"


def test_page_renders_saved_run_in_both_view_modes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENTICDOC_RESULTS_DIR", str(tmp_path))
    root = tmp_path / "local_pdf_extraction"
    data = factory.small_pdf(tmp_path / "s.pdf", pages=2).read_bytes()
    run_path = storage.create_run(root, [("تقرير.pdf", data)],
                                  runner.method_entries(["pymupdf", "pypdf", "pymupdf_ar_reordered"], {}), {"reorder": {}})
    runner.run(run_path)
    files_before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file())

    app = AppTest.from_file(str(PAGE), default_timeout=60).run()
    assert not app.exception, app.exception
    markdown = " ".join(item.value for item in app.markdown)
    assert "COMPLETED" in markdown
    assert "Processed — PyMuPDF + Arabic logical-order reconstruction (not raw library output)" in markdown
    assert any(button.label == "Run Extraction" for button in app.button)
    assert "Reversed-word signs (tokens ending لا / starting ال)" in markdown

    app.radio(key="lpe-view-mode").set_value("All Pages").run()
    assert not app.exception, app.exception
    assert any("Pages 1-2 of 2" in item.value for item in app.caption)

    app.checkbox(key="lpe-view-ocr").check().run()
    assert not app.exception, app.exception
    assert any("No verified earlier OCR/VLM page results" in item.value for item in app.info)

    next(b for b in app.button if b.label == "Verify saved pages").click().run()
    assert not app.exception, app.exception
    assert any(b.label.startswith("Resume (0 page results") for b in app.button)
    next(b for b in app.button if b.label == "Prepare HTML").click().run()
    next(b for b in app.button if b.label == "Prepare ZIP").click().run()
    assert not app.exception, app.exception
    labels = {element.proto.label for element in app.get("download_button")}
    assert {"Download TXT", "Download JSON", "Download CSV", "Download HTML Comparison",
            "Download All Results ZIP"} <= labels
    assert run_path.name in " ".join(item.value for item in app.markdown)  # Saved runs table
    exported = {p.name for p in (run_path / "exports").iterdir()}
    assert exported == {"comparison.html", "summary.csv", "all_results.json", "all_results.zip"}

    # Viewing only adds display-cache images and exports; no result file was changed.
    for path in files_before:
        assert (tmp_path / path).exists()
    files_after = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file()
                         and "page_images" not in p.parts and "exports" not in p.parts)
    assert set(files_before) <= set(files_after)
    assert not [p for p in files_after if p not in files_before]
