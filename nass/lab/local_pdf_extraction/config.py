"""Where this benchmark stores its results.

The project resolves its results root through `AGENTICDOC_RESULTS_DIR`
(default `D:/agenticdoc_e2/results`, see `lab.core.e2.store.results_root`).
This feature appends `local_pdf_extraction` and never writes anywhere else.
"""

from __future__ import annotations

from pathlib import Path

from lab.core.e2.store import results_root

RESULTS_SUBDIR = "local_pdf_extraction"
MAX_FILE_MB = 50
MAX_PAGES_PER_FILE = 500
VIEWER_DPI = 110


def benchmark_root() -> Path:
    return results_root() / RESULTS_SUBDIR
