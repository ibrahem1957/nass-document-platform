"""CLI: python -m docprep.quality RUN --structure DIR --relations DIR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .inputs import InputError
from .output import OutputExistsError
from .quality_builder import build_quality
from .quality_config import DEFAULT_QUALITY_OUTPUT_ROOT, QualityConfig
from .quality_inputs import load_quality_inputs
from .quality_output import write_quality_outputs


def run(run_dir: Path, structure_dir: Path, relations_dir: Path, *,
        out_root: Path = DEFAULT_QUALITY_OUTPUT_ROOT, stamp: str | None = None) -> tuple[Path, list[dict]]:
    inputs = load_quality_inputs(run_dir, structure_dir, relations_dir)
    config = QualityConfig(output_root=str(out_root))
    result = build_quality(inputs, config)
    return write_quality_outputs(result, inputs, out_root=out_root, stamp=stamp, config=config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m docprep.quality",
        description="Inspect saved Nass, Structure and Relationship outputs without modifying them.",
    )
    parser.add_argument("run_dir", type=Path, help="saved Nass Automatic Extraction run")
    parser.add_argument("--structure", type=Path, required=True, help="matching Structure output")
    parser.add_argument("--relations", type=Path, required=True, help="matching Relationship output")
    parser.add_argument("--out", type=Path, default=DEFAULT_QUALITY_OUTPUT_ROOT, help="write-once output root")
    args = parser.parse_args(argv)
    try:
        target, validation = run(args.run_dir, args.structure, args.relations, out_root=args.out)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OutputExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    errors = [item for item in validation if item.get("severity") == "error"]
    print(f"written: {target}")
    return 4 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

