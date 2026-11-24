"""Compare two Automatic Extraction runs (READ ONLY: both runs are re-assembled in memory, nothing is
written). Typical use: a new run against the previous baseline run of the same PDF.

    python -m lab.core.auto.compare <baseline run dir> <new run dir> [--json]

Reported per run: visuals processed / pending review / flagged, ambiguous titles, masked regions,
cross-source duplicate candidates (the order-insensitive value-and-label check, so it is computed for
older runs too), ordered duplicates removed by the safety net, reading-order methods, validation
summary and OCR job counts. Accuracy against Ground Truth is NOT computed here.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from lab.core.auto import assemble, pipeline, regions


def run_summary(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    document = assemble.assemble(run_dir)
    visuals = [e for e in document["elements"] if e["type"] in {"figure", "table"} and e.get("region_id")]
    selections = []
    for page in document["plan"]["layout"]["pages"]:
        if page.get("selection"):
            selections.append(page["selection"])
        selections += [e["selection"] for e in page.get("embedded_scans") or [] if e.get("selection")]
    decisions = Counter(r["decision"] for s in selections for r in s["regions"])
    report = document["validation"]
    figure_jobs = [j for j in document["jobs"].values() if j.get("purpose") == "digital_figure"]
    empty = []
    for job in document["jobs"].values():
        result = document["results"].get(job["job_id"]) or {}
        if result.get("state") == "completed" and not ((result.get("response") or {}).get("final_text") or "").strip():
            empty.append(job["job_id"])
    return {
        "digital_visual_crops": {"jobs": len(figure_jobs),
                                 "methods": dict(Counter((j.get("image") or {}).get("method") or "unknown"
                                                         for j in figure_jobs)),
                                 "render_dpi": dict(Counter(str((j.get("image") or {}).get("render_dpi"))
                                                            for j in figure_jobs))},
        "empty_ocr_jobs": empty,
        "run_id": run_dir.name, "finalized": pipeline.is_finalized(run_dir),
        "document_sha256": document["plan"]["run"]["document"]["sha256"],
        "visuals": {"processed": sum(1 for v in visuals if v.get("content")),
                    "pending_review": sum(1 for v in visuals if v.get("validation_state") == "pending_review"),
                    "decisions": dict(decisions),
                    "flagged_processed": sum(decisions[d] for d in regions.FLAGGED)},
        "ambiguous_titles_processed": sum(1 for v in visuals if (v.get("provenance") or {}).get("title_ambiguity")),
        "masked_regions": sum(len((p.get("duplicates") or {}).get("masked_ids") or []) for p in document["pages"]),
        "duplicate_check": {k: report["duplicate_check"][k] for k in ("raw_page_ocr", "final_document")},
        "ordered_spans_removed": sum(len(p["dedup"]) for p in document["pages"]),
        "reading_order": dict(Counter(p["reading_order_method"] for p in document["pages"])),
        "validation": report["summary"] | {"status": report["status"]},
        "jobs": pipeline.status(run_dir)["counts"],
    }


def compare(baseline: Path, new: Path) -> dict:
    old, current = run_summary(baseline), run_summary(new)
    notes = []
    if old["document_sha256"] != current["document_sha256"]:
        notes.append("the two runs are for DIFFERENT PDFs (SHA-256 differs)")
    return {"baseline": old, "new": current, "notes": notes,
            "delta": {"empty_ocr_jobs": len(current["empty_ocr_jobs"]) - len(old["empty_ocr_jobs"]),
                      "visuals_processed": current["visuals"]["processed"] - old["visuals"]["processed"],
                      "pending_review": current["visuals"]["pending_review"] - old["visuals"]["pending_review"],
                      "duplicate_candidates_final": (current["duplicate_check"]["final_document"]["total"]
                                                     - old["duplicate_check"]["final_document"]["total"])}}


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 2:
        print(__doc__)
        return 2
    result = compare(Path(args[0]), Path(args[1]))
    if "--json" in argv:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    for side in ("baseline", "new"):
        s = result[side]
        print(f"{side}: {s['run_id']} | visuals processed {s['visuals']['processed']}, pending review "
              f"{s['visuals']['pending_review']}, flagged {s['visuals']['flagged_processed']} | masked {s['masked_regions']} | "
              f"duplicate candidates final {s['duplicate_check']['final_document']['total']} "
              f"(raw page OCR {s['duplicate_check']['raw_page_ocr']['total']}) | reading order {s['reading_order']} | "
              f"validation {s['validation']} | jobs {s['jobs']} | digital crops {s['digital_visual_crops']} | "
              f"empty OCR {len(s['empty_ocr_jobs'])} {s['empty_ocr_jobs'][:12]}")
    print("delta:", result["delta"], *result["notes"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
