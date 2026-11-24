"""Local setup check for nass-document-platform. No network request, no model load, no download.

Usage (from the repository root, with the root virtual environment active):
    python scripts/check_setup.py

Checks: Python version, required packages, Studio configuration, DocPrep schema, optional
PP-DocLayout packages, and the Nass readiness adapter (configuration inspection only, exactly what the
Studio readiness banner runs). Secret values are never printed. Exit code 0 when the core components
are importable and configured; readiness problems for new extractions are reported as warnings."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"fitz": "PyMuPDF", "jsonschema": "jsonschema", "httpx": "httpx", "pydantic": "pydantic",
            "PIL": "pillow", "pytest": "pytest"}
OPTIONAL = {"paddle": "paddlepaddle (PP-DocLayout)", "paddleocr": "paddleocr (PP-DocLayout)",
            "streamlit": "streamlit (Nass Extraction Lab only)"}


def line(status: str, message: str) -> None:
    print(f"[{status:<4}] {message}")


def main() -> int:
    failures = 0
    if sys.version_info[:2] == (3, 12):
        line("OK", f"Python {sys.version.split()[0]}")
    else:
        line("WARN", f"Python {sys.version.split()[0]}: the verified version is 3.12 x64")
    for module, name in REQUIRED.items():
        if importlib.util.find_spec(module):
            line("OK", f"package {name}")
        else:
            line("FAIL", f"package {name} missing (pip install -r requirements.txt)")
            failures += 1
    for module, name in OPTIONAL.items():
        line("OK" if importlib.util.find_spec(module) else "WARN",
             f"optional {name}" + ("" if importlib.util.find_spec(module) else " not installed"))

    schema = ROOT / "docprep" / "schemas" / "prepared_document.schema.json"
    try:
        json.loads(schema.read_text(encoding="utf-8"))
        line("OK", "DocPrep schema " + schema.relative_to(ROOT).as_posix())
    except (OSError, ValueError) as exc:
        line("FAIL", f"DocPrep schema unreadable: {exc}")
        failures += 1

    sys.path.insert(0, str(ROOT / "studio"))
    try:
        from studio.config import load_config
        config = load_config(ROOT / "studio" / "config" / "studio.json")
        line("OK", f"Studio config (nass_root={config.nass_root.name}, docprep_root={config.docprep_root.name}, "
                   f"{len(config.existing_documents)} library document(s))")
    except Exception as exc:  # noqa: BLE001
        line("FAIL", f"Studio config: {type(exc).__name__}: {exc}")
        return failures + 1

    env_file = ROOT / "nass" / ".env"
    line("OK" if env_file.is_file() else "WARN",
         "nass/.env present" if env_file.is_file() else "nass/.env not found (copy .env.example to nass/.env)")

    completed = subprocess.run(
        [sys.executable, str(ROOT / "studio" / "studio" / "nass_runner.py"), "--nass-root", str(config.nass_root),
         "readiness"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
        check=False)
    record = next((json.loads(item[len("STUDIO_RESULT="):]) for item in reversed(completed.stdout.splitlines())
                   if item.startswith("STUDIO_RESULT=")), None)
    if record is None:
        line("FAIL", "Nass readiness adapter produced no result: " + (completed.stderr.strip()[-400:] or "?"))
        failures += 1
    elif record.get("ready"):
        line("OK", f"Nass ready for new extractions (layout {record['layout']['model_name']}, "
                   f"Qari {record.get('service_display')}, mode {record.get('deployment_mode')}, "
                   f"token {'present' if record.get('token_present') else 'ABSENT'})")
    else:
        detail = record.get("error") or "; ".join((record.get("layout") or {}).get("problems") or [])
        line("WARN", f"Nass not ready for new extractions: {detail} "
                     "(Studio still opens saved documents; see docs/RUNPOD_SETUP.md)")
    print("FAILED" if failures else "Core components OK.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
