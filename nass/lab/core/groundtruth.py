from __future__ import annotations

import json
from datetime import datetime, timezone

from lab.core.models import (
    GroundTruth,
    GroundTruthNumericValue,
    GroundTruthPage,
    GroundTruthTable,
    LabDocument,
    RunResult,
)
from lab.core.workspace import WorkspaceManager


class GroundTruthError(ValueError):
    pass


def _numeric_values(raw_page: dict, page_number: int) -> list[GroundTruthNumericValue]:
    """Copy curated numeric values exactly as supplied; never infer missing ones.

    An entry with `row`/`column` belongs to a table cell; everything else is
    running text. Duplicate entries are kept to preserve occurrence counts.
    """
    raw_values = raw_page.get("numeric_values", [])
    if not isinstance(raw_values, list):
        raise GroundTruthError(f"Ground-truth page {page_number} numeric_values must be a list")
    values: list[GroundTruthNumericValue] = []
    for item in raw_values:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
            raise GroundTruthError(
                f"Ground-truth page {page_number} has a numeric value without exact text"
            )
        is_table = "row" in item or "column" in item
        row = item.get("row")
        column = item.get("column")
        values.append(
            GroundTruthNumericValue(
                text=item["text"],
                source="table" if is_table else "text",
                block_id=str(item["block_id"]) if item.get("block_id") is not None else None,
                row=row if isinstance(row, int) else None,
                column=str(column) if column is not None else None,
            )
        )
    return values


def _tables(raw_page: dict, page_number: int) -> list[GroundTruthTable]:
    """Copy source tables (logical column order) exactly; reject ragged rows."""
    raw_tables = raw_page.get("tables", [])
    if not isinstance(raw_tables, list):
        raise GroundTruthError(f"Ground-truth page {page_number} tables must be a list")
    tables: list[GroundTruthTable] = []
    for index, item in enumerate(raw_tables):
        columns = item.get("columns") if isinstance(item, dict) else None
        rows = item.get("rows") if isinstance(item, dict) else None
        if (
            not isinstance(columns, list)
            or not isinstance(rows, list)
            or not all(isinstance(column, str) for column in columns)
            or not all(
                isinstance(row, list)
                and len(row) == len(columns)
                and all(isinstance(cell, str) for cell in row)
                for row in rows
            )
        ):
            raise GroundTruthError(
                f"Ground-truth page {page_number} table {index + 1} needs string "
                "columns and rows with one cell per column"
            )
        tables.append(
            GroundTruthTable(
                table_id=str(item.get("id") or f"p{page_number}_t{index + 1}"),
                columns=columns,
                rows=rows,
            )
        )
    return tables


class GroundTruthStore:
    def __init__(self, manager: WorkspaceManager):
        self.manager = manager

    def _path(self, workspace_id: str, doc_id: str):
        return self.manager.path(workspace_id) / "gt" / f"{doc_id}.json"

    def load(self, workspace_id: str, doc_id: str) -> GroundTruth | None:
        path = self._path(workspace_id, doc_id)
        if not path.is_file():
            return None
        return GroundTruth.model_validate_json(path.read_text(encoding="utf-8"))

    def _history_dir(self, workspace_id: str, doc_id: str):
        return self.manager.path(workspace_id) / "gt" / "history" / doc_id

    def _archive(self, workspace_id: str, ground_truth: GroundTruth) -> None:
        """Keep every version write-once; an existing identical archive is accepted."""
        directory = self._history_dir(workspace_id, ground_truth.doc_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"v{ground_truth.version:04d}.json"
        text = json.dumps(ground_truth.model_dump(mode="json"), ensure_ascii=False, indent=2)
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != text:
                raise GroundTruthError(
                    f"Ground-truth version {ground_truth.version} already exists with "
                    "different content; refusing to overwrite history"
                ) from None

    def save(self, workspace_id: str, ground_truth: GroundTruth) -> GroundTruth:
        current = self.load(workspace_id, ground_truth.doc_id)
        if current is not None:
            if ground_truth.version <= current.version:
                raise GroundTruthError(
                    f"Ground-truth version must increase (current {current.version}, "
                    f"new {ground_truth.version})"
                )
            # Versions saved before history existed are archived on first touch.
            self._archive(workspace_id, current)
        self._archive(workspace_id, ground_truth)
        self.manager.atomic_write_json(
            self._path(workspace_id, ground_truth.doc_id),
            ground_truth.model_dump(mode="json"),
        )
        return ground_truth

    def versions(self, workspace_id: str, doc_id: str) -> list[GroundTruth]:
        """Every archived version, oldest first (the current one included)."""
        directory = self._history_dir(workspace_id, doc_id)
        if not directory.is_dir():
            return []
        return [
            GroundTruth.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("v*.json"))
        ]

    def import_json(
        self,
        workspace_id: str,
        document: LabDocument,
        payload: bytes | str,
        *,
        verified: bool = False,
        reviewer: str | None = None,
    ) -> GroundTruth:
        try:
            data = json.loads(payload.decode("utf-8") if isinstance(payload, bytes) else payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GroundTruthError("Ground-truth JSON is invalid") from exc
        declared = data.get("document", {})
        declared_file = declared.get("file")
        if declared_file and declared_file != document.display_name:
            raise GroundTruthError(
                "Ground-truth filename does not match the selected document"
            )
        if declared.get("page_count") not in {None, document.page_count}:
            raise GroundTruthError("Ground-truth page count does not match the document")
        raw_pages = data.get("pages")
        if not isinstance(raw_pages, list):
            raise GroundTruthError("Ground-truth JSON must contain a pages list")
        pages: list[GroundTruthPage] = []
        now = datetime.now(timezone.utc)
        for raw in raw_pages:
            page_number = raw.get("page_number", raw.get("page"))
            text = raw.get("full_text_in_reading_order", raw.get("text"))
            if not isinstance(page_number, int) or not 1 <= page_number <= document.page_count:
                raise GroundTruthError("Ground-truth contains an invalid page number")
            if not isinstance(text, str):
                raise GroundTruthError(f"Ground-truth page {page_number} has no reference text")
            pages.append(
                GroundTruthPage(
                    page=page_number,
                    text=text,
                    status="verified" if verified else "draft",
                    seeded_from=None,
                    verified_by=reviewer if verified else None,
                    verified_at=now if verified else None,
                    notes="Imported from benchmark JSON",
                    numeric_values=_numeric_values(raw, page_number),
                    tables=_tables(raw, page_number),
                )
            )
        if len({page.page for page in pages}) != len(pages):
            raise GroundTruthError("Ground-truth contains duplicate pages")
        current = self.load(workspace_id, document.doc_id)
        ground_truth = GroundTruth(
            doc_id=document.doc_id,
            doc_sha256=document.sha256,
            task="text",
            pages=sorted(pages, key=lambda page: page.page),
            version=(current.version + 1) if current else 1,
            source="imported",
        )
        return self.save(workspace_id, ground_truth)

    def import_text(self, workspace_id: str, document: LabDocument, text: str) -> GroundTruth:
        pieces = text.split("\f")
        if len(pieces) != document.page_count:
            raise GroundTruthError("TXT must contain one form-feed-separated section per page")
        current = self.load(workspace_id, document.doc_id)
        ground_truth = GroundTruth(
            doc_id=document.doc_id,
            doc_sha256=document.sha256,
            task="text",
            pages=[
                GroundTruthPage(
                    page=index,
                    text=value,
                    status="draft",
                    seeded_from=None,
                    verified_by=None,
                    verified_at=None,
                    notes="Imported from form-feed TXT",
                )
                for index, value in enumerate(pieces, start=1)
            ],
            version=(current.version + 1) if current else 1,
            source="imported",
        )
        return self.save(workspace_id, ground_truth)

    def seed_from_run(self, workspace_id: str, document: LabDocument, run: RunResult) -> GroundTruth:
        current = self.load(workspace_id, document.doc_id)
        ground_truth = GroundTruth(
            doc_id=document.doc_id,
            doc_sha256=document.sha256,
            task="text",
            pages=[
                GroundTruthPage(
                    page=page.page,
                    text=page.text,
                    status="draft",
                    seeded_from=run.run_id,
                    verified_by=None,
                    verified_at=None,
                    notes="Seeded from extractor output; review required",
                )
                for page in run.pages
            ],
            version=(current.version + 1) if current else 1,
            source="manual",
        )
        return self.save(workspace_id, ground_truth)

    def update_page(
        self,
        workspace_id: str,
        doc_id: str,
        page_number: int,
        text: str,
        *,
        verify: bool,
        reviewer: str | None,
        notes: str | None,
    ) -> GroundTruth:
        ground_truth = self.load(workspace_id, doc_id)
        if ground_truth is None:
            raise GroundTruthError("Import or seed ground truth first")
        now = datetime.now(timezone.utc)
        pages = []
        found = False
        for page in ground_truth.pages:
            if page.page == page_number:
                found = True
                changed = page.text != text
                pages.append(
                    page.model_copy(
                        update={
                            "text": text,
                            "status": "verified" if verify else "draft",
                            "verified_by": reviewer if verify else None,
                            "verified_at": now if verify else None,
                            "notes": notes,
                            "seeded_from": None if changed else page.seeded_from,
                        }
                    )
                )
            else:
                pages.append(page)
        if not found:
            raise GroundTruthError("Ground-truth page does not exist")
        return self.save(
            workspace_id,
            ground_truth.model_copy(update={"pages": pages, "version": ground_truth.version + 1}),
        )
