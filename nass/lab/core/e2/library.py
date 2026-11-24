"""Extensible comparison table built from the persistent result store.

Every registered candidate appears for every language, whatever its state:
`completed`, `partial`, `failed`, `incomplete` (run directory without final.json),
`unsupported on current hardware` or `not yet tested`. No score is shown for a
candidate without a stored run. No candidate is ranked or rejected here.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from lab.core.e2.specs import ModelSpec
from lab.core.metrics import finite_metric

LANGUAGES = ("Arabic", "English")
STATUS_ORDER = ("completed", "partial", "failed", "incomplete")


def _ratio(pair: tuple[int, int]) -> str:
    correct, total = pair
    return f"{correct}/{total}"


def _metric_cells(summary: dict) -> list[str]:
    metric = summary["metrics"]
    tables = summary["tables"]
    mixed = summary["mixed_direction"]
    words = summary["words"]
    timing = summary["timing"]
    peak = summary["memory"].get("peak_rss_mb")
    mean = timing.get("page_ms_mean")
    return [
        f"{metric['char_edits']}/{metric['ref_chars']}={metric['cer']:.4f}",
        f"{metric['word_edits']}/{metric['ref_words']}={metric['wer']:.4f}",
        _ratio((metric["num_correct"], metric["num_ref_count"])),
        f"{metric['curated_num_text_correct'] + metric['curated_num_table_correct']}/"
        f"{metric['curated_num_text_ref'] + metric['curated_num_table_ref']}",
        _ratio((metric["date_correct"], metric["date_ref_count"])),
        _ratio((metric["identifier_correct"], metric["identifier_ref_count"])),
        _ratio((metric["operator_correct"], metric["operator_ref_count"])),
        _ratio((tables["cells_exact"], tables["cells_reference"])),
        _ratio((tables["data_cells_associated"], tables["data_cells_reference"])),
        _ratio((mixed["exact"], mixed["reference"])),
        f"{words['missing']}/{words['reference']}; +{words['extra']}",
        f"{finite_metric(metric['reading_order_tau'])} "
        f"({metric['reading_order_matched_lines']}/{metric['reading_order_ref_lines']})",
        "—" if mean is None else f"{mean / 1000:.1f}",
        "—" if peak is None else f"{peak:.0f}",
    ]


EMPTY = ["—"] * 14


def truncated_pages(directory: str | Path) -> list[int] | None:
    """Pages whose stored RAW response stopped at max_new_tokens.

    Read from the write-once page records, so runs stored before this column existed
    are covered too. Returns None when no page record carries the flag (text layers).
    """
    pages_dir = Path(directory) / "pages"
    found: list[int] = []
    flagged = False
    for path in sorted(pages_dir.glob("p*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        raw = record.get("raw") if isinstance(record, dict) else None
        if (
            isinstance(raw, dict)
            and isinstance(raw.get("truncated_at_max_new_tokens"), bool)
        ):
            flagged = True
            if raw["truncated_at_max_new_tokens"]:
                found.append(int(record.get("page", 0)))
    return found if flagged else None


def unknown_truncation_pages(directory: str | Path) -> list[int]:
    """Pages from a generating backend that did not report token termination."""
    found: list[int] = []
    for path in sorted((Path(directory) / "pages").glob("p*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        raw = record.get("raw") if isinstance(record, dict) else None
        if (
            isinstance(raw, dict)
            and "truncated_at_max_new_tokens" in raw
            and raw["truncated_at_max_new_tokens"] is None
        ):
            found.append(int(record.get("page", 0)))
    return found


def _truncation_cell(directory: str) -> str:
    pages = truncated_pages(directory)
    unknown = unknown_truncation_pages(directory)
    if pages is None:
        return f"unknown ({len(unknown)} pages)" if unknown else "n/a"
    cell = f"{len(pages)} ({', '.join(map(str, pages))})" if pages else "0"
    return cell + (f"; unknown {len(unknown)}" if unknown else "")


def build_comparison(store, specs: list[ModelSpec], runner=None) -> str:
    runs = store.runs()
    candidates: list[tuple[str, str, ModelSpec | None]] = []
    seen: set[str] = set()
    for item in runs:
        adapter = item["final"].get("adapter_id") or item["started"].get("adapter_id")
        if adapter and adapter.startswith("text.") and adapter not in seen:
            seen.add(adapter)
            candidates.append((adapter, item["final"].get("label") or adapter, None))
    for spec in specs:
        candidates.append((f"e2.{spec.id}", spec.label, spec))
    lines = [
        "# E2 Model Evaluation Library — comparison table",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()} from `{store.root}`.",
        "Every candidate is listed for every language. Statuses: completed, partial,",
        "failed, incomplete, unsupported on current hardware, not yet tested. No score is",
        "shown without a stored run; no candidate is ranked, selected or rejected here.",
        "Compare rows only when their page sets match.",
        "",
        "| Candidate | Language | Status | Pages ok/total | Truncated pages | Variant / kind | CER | WER | Num | "
        "Curated num | Date | ID | Op | Table cells | Table assoc | Mixed lines | Words missing; extra | "
        "Reading order | s/page | Peak RAM MB | Stored runs |",
        "|" + "---|" * 21,
    ]
    for adapter_id, label, spec in candidates:
        for language in LANGUAGES:
            history = [
                item
                for item in runs
                if (item["final"].get("adapter_id") or item["started"].get("adapter_id")) == adapter_id
                and (item["final"].get("language") or item["started"].get("language")) == language
            ]
            history.sort(key=lambda item: item["final"].get("finalized_at") or "")
            chosen = None
            for status in STATUS_ORDER:
                matching = [item for item in history if item["final"].get("status") == status]
                if matching:
                    chosen = matching[-1]
                    break
            if chosen is None:
                state = "not yet tested"
                if spec is not None and runner is not None:
                    readiness = runner.readiness(spec)
                    state = (
                        "unsupported on current hardware (not rejected)"
                        if readiness.status == "insufficient_hardware"
                        else f"not yet tested ({readiness.status})"
                    )
                lines.append(
                    f"| {label} | {language} | {state} | — | — | — | " + " | ".join(EMPTY) + f" | {len(history)} |"
                )
                continue
            status = chosen["final"]["status"]
            summaries = chosen["summaries"]
            truncation = _truncation_cell(chosen["directory"])
            if not summaries:
                pages = chosen["final"].get("pages_saved") or chosen["final"].get("pages") or []
                lines.append(
                    f"| {label} | {language} | {status} | {len(pages)} saved | {truncation} | — | "
                    + " | ".join(EMPTY)
                    + f" | {len(history)} |"
                )
                continue
            for summary in summaries:
                lines.append(
                    f"| {label} | {language} | {status} | "
                    f"{summary['pages_ok']}/{summary['pages_total']} | {truncation} | "
                    f"{summary['variant']} / {summary.get('kind', '')} | "
                    + " | ".join(_metric_cells(summary))
                    + f" | {len(history)} |"
                )
    lines += [
        "",
        "Notes: `Num` is the regex over running text; `Curated num` scores every verified",
        "ground-truth numeric value including table cells. `Table cells` = exact cell text;",
        "`Table assoc` = value under the correct row and column in a parsed output table",
        "(text layers have no structure, so 0). `Words … +N` lists extra words, which include",
        "headers/footers absent from the reference: a review signal, not proof of hallucination.",
        "`Truncated pages` = pages whose RAW response hit max_new_tokens (usually a repetition",
        "loop); `completed` means every page returned, not that every page finished cleanly.",
        "`n/a` = the source is a text layer with no generation step. `unknown` = the backend",
        "(e.g. the official PaddleOCR pipeline) does not report token termination; it is never",
        "counted as 0.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Rebuild the comparison table from stored runs. Runs no model."""
    import argparse

    from lab.core.e2.runner import E2Runner
    from lab.core.e2.specs import load_model_specs
    from lab.core.e2.store import ResultStore, results_root
    from lab.core.settings import load_settings
    from lab.core.workspace import WorkspaceManager

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--results-root", type=Path, default=None)
    args = parser.parse_args(argv)
    store = ResultStore(args.results_root or results_root())
    settings = load_settings()
    runner = E2Runner(WorkspaceManager(settings), settings)
    path = store.write_report("comparison", build_comparison(store, load_model_specs(), runner))
    print(f"Wrote comparison library: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
