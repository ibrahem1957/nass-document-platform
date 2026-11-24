"""Test helper: register a test PDF's exact hash in a temporary synthetic-document
registry and authorize it, exactly as a user would (never via a folder)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from lab.core.e2.api import documents


def register_test_documents(monkeypatch, tmp_path: Path, *pdfs: Path) -> Path:
    lines = []
    for index, pdf in enumerate(pdfs):
        data = Path(pdf).read_bytes()
        lines += [
            "[[documents]]",
            f'id = "test-document-{index}"',
            f'filename = "{Path(pdf).name}"',
            f'sha256 = "{hashlib.sha256(data).hexdigest()}"',
            f"bytes = {len(data)}",
            "page_count = 1",
            'title = "test fixture"',
            'provenance = "generated test data"',
            'verified_on = "2026-11-17"',
            "",
        ]
    registry = tmp_path / "test_synthetic_documents.toml"
    registry.write_text("\n".join(lines), encoding="utf-8")
    monkeypatch.setattr(documents, "DOCUMENT_REGISTRY", registry)
    return registry


def authorize(pdf: Path, root: Path, *, days: int = 1) -> Path:
    sha = hashlib.sha256(Path(pdf).read_bytes()).hexdigest()
    entry = documents.load_registry()[sha]
    return documents.authorize_document(
        entry, file_path=Path(pdf), reviewer="Tester", declaration=documents.SYNTHETIC_DECLARATION,
        days_valid=days, root=root,
    )
