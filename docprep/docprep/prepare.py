"""CLI: python -m docprep.prepare RUN --structure DIR --relations DIR --quality DIR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .inputs import InputError
from .output import OutputExistsError
from .prepared_builder import build_prepared_document
from .prepared_config import DEFAULT_PREPARED_OUTPUT_ROOT, PreparedConfig
from .prepared_inputs import load_prepared_inputs
from .prepared_output import write_prepared_outputs


def run(run_dir: Path, structure_dir: Path, relations_dir: Path, quality_dir: Path, *,
        out_root: Path = DEFAULT_PREPARED_OUTPUT_ROOT, stamp: str | None = None) -> tuple[Path, list[dict]]:
    inputs = load_prepared_inputs(run_dir, structure_dir, relations_dir, quality_dir)
    config = PreparedConfig(output_root=str(out_root))
    result = build_prepared_document(inputs, config)
    return write_prepared_outputs(result, inputs, out_root=out_root, stamp=stamp, config=config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m docprep.prepare",
        description="Build a unified, self-contained document representation from saved DocPrep layers.",
    )
    parser.add_argument("run_dir", type=Path, help="saved Nass Automatic Extraction run")
    parser.add_argument("--structure", type=Path, required=True, help="matching Structure output")
    parser.add_argument("--relations", type=Path, required=True, help="matching Relationships output")
    parser.add_argument("--quality", type=Path, required=True, help="matching Quality output")
    parser.add_argument("--out", type=Path, default=DEFAULT_PREPARED_OUTPUT_ROOT, help="write-once output root")
    args = parser.parse_args(argv)
    try:
        target, validation = run(args.run_dir, args.structure, args.relations, args.quality, out_root=args.out)
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
