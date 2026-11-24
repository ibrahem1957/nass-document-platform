"""CLI: python -m docprep.structure <run_dir> [--out <root>] [--config <toml>] [--no-pdf] [--no-semantic]

Reads one saved Nass Automatic Extraction run (read-only) and writes a new, write-once output folder
<root>/<run_id>__<UTC timestamp>/ with section_tree.json, assignments.json, structure_report.md,
structure_report.html and manifest.json.

Exit codes: 0 success, 2 input or configuration error, 3 output folder already exists, 4 validation errors
(outputs are still written so the findings can be reviewed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigError, load_config
from .inputs import InputError
from .output import OutputExistsError, build_manifest, write_outputs
from .pipeline import assignments_json, build_structure, section_tree_json
from .report_html import render_html
from .report_md import render_markdown


def run(run_dir: Path, *, out_root: Path | None = None, config_path: Path | None = None, use_pdf: bool = True,
        semantic: bool | None = None, stamp: str | None = None) -> tuple[Path, list[dict]]:
    overrides = {}
    if semantic is not None:
        overrides["semantic"] = {"enabled": semantic}
    if out_root is not None:
        overrides["output"] = {"root": str(out_root)}
    config = load_config(config_path, overrides)
    result = build_structure(run_dir, config, use_pdf=use_pdf)
    contents = {
        "section_tree.json": json.dumps(section_tree_json(result), ensure_ascii=False, indent=2),
        "assignments.json": json.dumps(assignments_json(result), ensure_ascii=False, indent=2),
        "structure_report.md": render_markdown(result),
        "structure_report.html": render_html(result),
    }

    def manifest(output_hashes: dict[str, str]) -> dict:
        return build_manifest(run_id=result.inputs.run_id, run_dir=str(result.inputs.run_dir),
                              input_hashes_before=result.inputs.input_hashes, input_hashes_after=result.hashes_after,
                              config=config.to_dict(), started_at=result.started_at, finished_at=result.finished_at,
                              output_hashes=output_hashes, findings=result.findings, warnings=result.warnings)

    target = write_outputs(config.output.root, result.inputs.run_id, contents, manifest, stamp=stamp,
                           read_only=config.output.read_only_files)
    return target, result.findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m docprep.structure",
                                     description="Build the document structure of a saved Nass run (read-only).")
    parser.add_argument("run_dir", type=Path, help="saved Automatic Extraction run folder")
    parser.add_argument("--out", type=Path, default=None, help="output root (default: configuration / "
                                                               "DOCPREP_OUTPUT_ROOT)")
    parser.add_argument("--config", type=Path, default=None, help="TOML configuration file")
    parser.add_argument("--no-pdf", action="store_true", help="do not read source.pdf (no typography)")
    parser.add_argument("--no-semantic", action="store_true", help="disable the semantic verification layer")
    args = parser.parse_args(argv)
    try:
        target, findings = run(args.run_dir, out_root=args.out, config_path=args.config, use_pdf=not args.no_pdf,
                               semantic=False if args.no_semantic else None)
    except (InputError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OutputExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    errors = [f for f in findings if f["severity"] == "error"]
    print(f"written: {target}")
    for finding in findings:
        print(f"  {finding['severity']:7} {finding['code']}: {finding['message']}")
    return 4 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
