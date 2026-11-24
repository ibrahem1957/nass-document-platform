"""Write-once human review templates and evaluation for Component 3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .output import OutputExistsError, safe_name, timestamp

FIELDS = (
    "finding_id", "category", "subtype", "page", "verbatim_excerpt", "severity", "tool_confidence",
    "evidence_summary", "human_verdict", "reviewer_note",
)
VERDICTS = {"", "confirmed_issue", "false_positive", "uncertain", "not_reviewed"}


class QualityEvaluationError(ValueError):
    """Invalid quality output or human review."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualityEvaluationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise QualityEvaluationError(f"{path} must contain an object")
    return value


def _pages(value: str) -> set[int]:
    pages = set()
    for part in value.split(","):
        try:
            page = int(part.strip())
        except ValueError as exc:
            raise QualityEvaluationError(f"invalid page: {part!r}") from exc
        if page < 1:
            raise QualityEvaluationError(f"invalid page: {page}")
        pages.add(page)
    return pages


def _root(quality_dir: Path) -> Path:
    return quality_dir.parent / "quality_reviews"


def _write_directory(target: Path, contents: dict[str, str], manifest: dict[str, Any]) -> Path:
    if target.exists():
        raise OutputExistsError(f"review output already exists (never overwritten): {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.partial-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    try:
        hashes = {}
        for name, text in contents.items():
            path = staging / name
            path.write_text(text, encoding="utf-8-sig" if name.endswith(".csv") else "utf-8", newline="")
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["output_hashes"] = hashes
        (staging / "review_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                                       encoding="utf-8", newline="\n")
        os.rename(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def make_template(quality_dir: Path | str, pages: set[int], *, stamp: str | None = None) -> Path:
    quality_dir = Path(quality_dir)
    data = _read_json(quality_dir / "quality_findings.json")
    sources = data.get("review_sources") or {}
    rows = []
    for finding in data.get("findings") or []:
        finding_pages = {int(value) for value in finding.get("page_numbers") or [] if isinstance(value, int)}
        if pages and not pages & finding_pages:
            continue
        source_ids = [str(value) for value in finding.get("source_ids") or []]
        excerpt = next((str((sources.get(value) or {}).get("text") or "") for value in source_ids
                        if (sources.get(value) or {}).get("text")), "")
        rows.append({
            "finding_id": finding.get("finding_id"), "category": finding.get("category"),
            "subtype": finding.get("subtype"), "page": min(finding_pages) if finding_pages else "",
            "verbatim_excerpt": " ".join(excerpt.split())[:700], "severity": finding.get("severity"),
            "tool_confidence": finding.get("confidence"),
            "evidence_summary": " ".join(json.dumps(finding.get("evidence"), ensure_ascii=False).split())[:1000],
            "human_verdict": "", "reviewer_note": "",
        })
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    target = _root(quality_dir) / f"{safe_name(quality_dir.name)}__template__{stamp or timestamp()}"
    manifest = {"schema_version": 1, "kind": "docprep_quality_review_template_manifest",
                "quality_output_dir": str(quality_dir), "pages": sorted(pages), "finding_count": len(rows),
                "human_fields_initially_blank": True, "written_at": datetime.now(timezone.utc).isoformat()}
    return _write_directory(target, {"quality_review_template.csv": buffer.getvalue()}, manifest)


def evaluate_review(quality_dir: Path | str, review_csv: Path | str, *, stamp: str | None = None) -> Path:
    quality_dir, review_csv = Path(quality_dir), Path(review_csv)
    data = _read_json(quality_dir / "quality_findings.json")
    known = {str(item.get("finding_id")) for item in data.get("findings") or []}
    with review_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or any(field not in reader.fieldnames for field in FIELDS):
            raise QualityEvaluationError("review CSV lacks required columns")
        rows = list(reader)
    unknown = sorted({str(item.get("finding_id")) for item in rows} - known)
    if unknown:
        raise QualityEvaluationError(f"unknown finding IDs: {', '.join(unknown[:10])}")
    normalized = [str(item.get("human_verdict") or "").strip().casefold() for item in rows]
    invalid = sorted(set(normalized) - VERDICTS)
    if invalid:
        raise QualityEvaluationError(f"invalid verdicts: {', '.join(invalid)}")
    counts = Counter(normalized)
    summary = {
        "schema_version": 1, "kind": "docprep_quality_human_evaluation",
        "quality_output_dir": str(quality_dir), "review_csv": str(review_csv),
        "counts": {"template_rows": len(rows), "reviewed": len(rows) - counts[""] - counts["not_reviewed"],
                   "confirmed_issue": counts["confirmed_issue"], "false_positive": counts["false_positive"],
                   "uncertain": counts["uncertain"], "not_reviewed": counts["not_reviewed"] + counts[""]},
        "coverage_denominator": len(rows), "created_at": datetime.now(timezone.utc).isoformat(),
    }
    markdown = "# Quality Human Review Summary\n\n" + "\n".join(
        f"- {key}: {value}" for key, value in summary["counts"].items()) + "\n"
    target = _root(quality_dir) / f"{safe_name(quality_dir.name)}__evaluation__{stamp or timestamp()}"
    manifest = {"schema_version": 1, "kind": "docprep_quality_review_evaluation_manifest",
                "quality_output_dir": str(quality_dir), "review_csv_sha256":
                hashlib.sha256(review_csv.read_bytes()).hexdigest(), "written_at": datetime.now(timezone.utc).isoformat()}
    return _write_directory(target, {
        "quality_review_evaluation.json": json.dumps(summary, ensure_ascii=False, indent=2),
        "quality_review_evaluation.md": markdown,
    }, manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m docprep.evaluate_quality")
    parser.add_argument("--make-template", type=Path)
    parser.add_argument("--pages", default="")
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--quality", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.make_template:
            target = make_template(args.make_template, _pages(args.pages) if args.pages else set())
        elif args.evaluate and args.quality:
            target = evaluate_review(args.quality, args.evaluate)
        else:
            parser.error("use --make-template DIR [--pages ...] or --evaluate CSV --quality DIR")
            return 2
    except (QualityEvaluationError, OSError, OutputExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"written: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

