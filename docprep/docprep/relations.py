"""CLI for Component 2: Content Relationship Builder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .inputs import InputError
from .output import OutputExistsError
from .relation_builder import build_relationships
from .relation_inputs import load_relationship_inputs
from .relation_output import DEFAULT_RELATION_OUTPUT_ROOT, write_relation_outputs


def run(run_dir: Path, structure_dir: Path, *, out_root: Path = DEFAULT_RELATION_OUTPUT_ROOT,
        stamp: str | None = None) -> tuple[Path, list[dict]]:
    inputs = load_relationship_inputs(run_dir, structure_dir)
    result = build_relationships(inputs)
    return write_relation_outputs(result, out_root=out_root, stamp=stamp)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m docprep.relations",
        description="Build auditable content relationships from saved Nass and DocPrep structure outputs.",
    )
    parser.add_argument("run_dir", type=Path, help="saved Nass Automatic Extraction run")
    parser.add_argument("--structure", type=Path, required=True, help="matching DocPrep structure output folder")
    parser.add_argument("--out", type=Path, default=DEFAULT_RELATION_OUTPUT_ROOT, help="write-once output root")
    args = parser.parse_args(argv)
    try:
        target, findings = run(args.run_dir, args.structure, out_root=args.out)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OutputExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    errors = [item for item in findings if item.get("severity") == "error"]
    print(f"written: {target}")
    return 4 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

