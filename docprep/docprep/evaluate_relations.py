"""Create and score write-once human review templates for Component 2."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .output import OutputExistsError, timestamp

FIELDS = (
    "relationship_id", "relationship_type", "source_excerpt", "target_excerpt", "page",
    "tool_confidence", "human_verdict", "human_note",
)
VALID_VERDICTS = {"correct", "incorrect", "uncertain", ""}


class RelationEvaluationError(ValueError):
    """Invalid relationship output or human review."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RelationEvaluationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RelationEvaluationError(f"{path} must contain a JSON object")
    return value


def _pages(value: str) -> set[int]:
    pages = set()
    for item in value.split(","):
        item = item.strip()
        if not re.fullmatch(r"[1-9][0-9]*", item):
            raise RelationEvaluationError(f"invalid page number: {item!r}")
        pages.add(int(item))
    return pages


def _review_root(relationship_dir: Path) -> Path:
    return relationship_dir.parent / "reviews"


def make_template(relationship_dir: Path | str, pages: set[int], *, stamp: str | None = None) -> Path:
    relationship_dir = Path(relationship_dir)
    data = _load(relationship_dir / "relationships.json")
    nodes = {str(item.get("node_id")): item for item in data.get("nodes") or []}
    root = _review_root(relationship_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{relationship_dir.name}__review_template__{stamp or timestamp()}.csv"
    if target.exists():
        raise OutputExistsError(f"review template already exists (never overwritten): {target}")
    rows = []
    for relation in data.get("relationships") or []:
        source = nodes.get(str(relation.get("source_id")), {})
        target_node = nodes.get(str(relation.get("target_id")), {})
        page = relation.get("source_page") or relation.get("target_page")
        relevant_pages = {value for value in (relation.get("source_page"), relation.get("target_page"))
                          if isinstance(value, int)}
        if pages and not (pages & relevant_pages):
            continue
        rows.append({
            "relationship_id": relation.get("rel_id"),
            "relationship_type": relation.get("type"),
            "source_excerpt": " ".join(str(source.get("text") or "").split())[:500],
            "target_excerpt": " ".join(str(target_node.get("text") or "").split())[:500],
            "page": page or "",
            "tool_confidence": relation.get("confidence"),
            "human_verdict": "",
            "human_note": "",
        })
    with target.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return target


def evaluate_review(relationship_dir: Path | str, review_csv: Path | str, *, stamp: str | None = None) -> Path:
    relationship_dir, review_csv = Path(relationship_dir), Path(review_csv)
    data = _load(relationship_dir / "relationships.json")
    known = {str(item.get("rel_id")) for item in data.get("relationships") or []}
    with review_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or any(field not in reader.fieldnames for field in FIELDS):
            raise RelationEvaluationError("review CSV does not have the required template columns")
        rows = list(reader)
    unknown = sorted({str(row.get("relationship_id")) for row in rows} - known)
    if unknown:
        raise RelationEvaluationError(f"review contains unknown relationship IDs: {', '.join(unknown[:10])}")
    invalid = sorted({str(row.get("human_verdict") or "").strip().casefold() for row in rows} - VALID_VERDICTS)
    if invalid:
        raise RelationEvaluationError(f"invalid human verdict(s): {', '.join(invalid)}")
    verdicts = Counter(str(row.get("human_verdict") or "").strip().casefold() for row in rows)
    summary = {
        "schema_version": 1,
        "kind": "docprep_relationship_human_evaluation",
        "relationship_output_dir": str(relationship_dir),
        "review_file": str(review_csv),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "total_rows": len(rows),
            "reviewed": len(rows) - verdicts[""],
            "correct": verdicts["correct"],
            "incorrect": verdicts["incorrect"],
            "uncertain": verdicts["uncertain"],
            "unreviewed": verdicts[""],
        },
    }
    root = _review_root(relationship_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{relationship_dir.name}__review_evaluation__{stamp or timestamp()}.json"
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m docprep.evaluate_relations")
    parser.add_argument("--make-template", type=Path, help="relationship output directory")
    parser.add_argument("--pages", default="", help="comma-separated pages for the template")
    parser.add_argument("--evaluate", type=Path, help="completed human review CSV")
    parser.add_argument("--relationships", type=Path, help="relationship output directory for --evaluate")
    args = parser.parse_args(argv)
    try:
        if args.make_template:
            target = make_template(args.make_template, _pages(args.pages) if args.pages else set())
        elif args.evaluate and args.relationships:
            target = evaluate_review(args.relationships, args.evaluate)
        else:
            parser.error("use --make-template DIR [--pages ...] or --evaluate CSV --relationships DIR")
            return 2
    except (RelationEvaluationError, OSError, OutputExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"written: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

