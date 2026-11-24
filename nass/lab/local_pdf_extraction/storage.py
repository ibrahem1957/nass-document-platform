"""Permanent, crash-safe storage for benchmark runs.

    <root>/run_YYYYMMDD_HHMMSS_xxxxxx/
        manifest.json  state.json  summary.json  .lock (while a worker runs)
        documents/<doc_key>/original.pdf
        documents/<doc_key>/<method_id>/page_NNN.txt|json, full_document.txt
        documents/<doc_key>/<method_id>/_quarantine/   (invalid files moved aside after approval)
        page_images/<doc_key>/page_NNN_<dpi>dpi.png     (display cache only)
        ground_truth/<doc_key>/ground_truth.txt + ground_truth.json
        resume_requests/request_NNNN.json   logs/worker_NNNN.log   exports/

Every file is written atomically (temp file in the same folder + os.replace),
UTF-8 without BOM. A page counts as completed only when `verify_page` passes.
A valid completed page is never overwritten. Nothing outside `<root>` is touched.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import socket
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
RUN_ID_PATTERN = re.compile(r"^run_\d{8}_\d{6}_[0-9a-f]{6}$")
DOC_KEY_PATTERN = re.compile(r"^d\d{2,3}_([a-z0-9_]{1,24}_)?[0-9a-f]{8}$")
LOCK_NAME = ".lock"
CANCEL_NAME = "cancel_request.json"
HEARTBEAT_STALE_S = 30.0


class StorageError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ------------------------------------------------------------ atomic writes
def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        with open(temp, "xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def encode_text(text: str) -> bytes:
    # surrogatepass keeps lone surrogates that some libraries emit (byte-exact round trip).
    return text.encode("utf-8", errors="surrogatepass")


def decode_text(data: bytes) -> str:
    return data.decode("utf-8", errors="surrogatepass")


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, encode_text(text))


def atomic_write_json(path: Path, value) -> None:
    atomic_write_bytes(path, json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# -------------------------------------------------------------- run layout
def new_run_id(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"run_{now:%Y%m%d_%H%M%S}_{secrets.token_hex(3)}"


def run_dir(root: Path, run_id: str) -> Path:
    if not RUN_ID_PATTERN.match(run_id):
        raise StorageError(f"Invalid run id: {run_id!r}")
    return Path(root) / run_id


def doc_key(index: int, original_name: str, sha256: str) -> str:
    """Short, ASCII-only, unique folder name. The original name lives in the manifest only."""
    stem = Path(unicodedata.normalize("NFKC", original_name)).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", stem.encode("ascii", "ignore").decode()).strip("_")[:24].strip("_")
    return f"d{index:02d}_{slug}_{sha256[:8]}" if slug else f"d{index:02d}_{sha256[:8]}"


def _checked_key(key: str) -> str:
    if not DOC_KEY_PATTERN.match(key):
        raise StorageError(f"Invalid document key: {key!r}")
    return key


def _checked_method(method_id: str) -> str:
    if not re.fullmatch(r"[a-z0-9_]{1,40}", method_id):
        raise StorageError(f"Invalid method id: {method_id!r}")
    return method_id


def document_dir(run_path: Path, key: str) -> Path:
    return run_path / "documents" / _checked_key(key)


def method_dir(run_path: Path, key: str, method_id: str) -> Path:
    return document_dir(run_path, key) / _checked_method(method_id)


def page_paths(run_path: Path, key: str, method_id: str, page: int) -> tuple[Path, Path]:
    folder = method_dir(run_path, key, method_id)
    return folder / f"page_{page:03d}.txt", folder / f"page_{page:03d}.json"


def original_pdf(run_path: Path, key: str) -> Path:
    return document_dir(run_path, key) / "original.pdf"


def pdf_page_count(data: bytes) -> int:
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as document:
        if document.needs_pass:
            raise StorageError("Encrypted PDF (password required)")
        return len(document)


def inspect_upload(name: str, data: bytes, max_mb: int, max_pages: int) -> dict:
    """Validate one uploaded PDF in memory. The name is never used as a path."""
    if not data.startswith(b"%PDF"):
        raise StorageError(f"{name}: not a PDF file")
    if len(data) > max_mb * 1024 * 1024:
        raise StorageError(f"{name}: larger than {max_mb} MB")
    pages = pdf_page_count(data)
    if pages < 1 or pages > max_pages:
        raise StorageError(f"{name}: {pages} pages (allowed 1-{max_pages})")
    return {"original_name": name, "size_bytes": len(data), "page_count": pages, "sha256": sha256_bytes(data)}


def create_run(root: Path, uploads: list[tuple[str, bytes]], methods: list[dict], settings: dict,
               *, max_mb: int = 50, max_pages: int = 500,
               ground_truth: list[tuple[str, list[str]] | None] | None = None) -> Path:
    """Create a run folder with original PDFs, manifest.json and a PENDING state.json.

    `ground_truth` (optional, aligned with `uploads`): (text, source file names) per PDF. It is
    saved into the run BEFORE the worker starts, so the worker's summary.json includes CER/WER.
    """
    if not uploads:
        raise StorageError("No PDF files")
    if not methods:
        raise StorageError("No libraries selected")
    if ground_truth is not None and len(ground_truth) != len(uploads):
        raise StorageError("Ground truth list must have one entry (or None) per uploaded PDF")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    run_id = new_run_id()
    path = run_dir(root, run_id)
    path.mkdir(exist_ok=False)
    documents = []
    for index, (name, data) in enumerate(uploads, start=1):
        info = inspect_upload(name, data, max_mb, max_pages)
        key = doc_key(index, name, info["sha256"])
        atomic_write_bytes(original_pdf(path, key), data)
        entry = {"doc_key": key, **info, "ground_truth_attached": False}
        gt = ground_truth[index - 1] if ground_truth else None
        if gt is not None and gt[0].strip():
            save_ground_truth(path, key, gt[0], list(gt[1]))
            entry.update(ground_truth_attached=True, ground_truth_sources=list(gt[1]))
        documents.append(entry)
    manifest = {
        "schema": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": utc_now(),
        "finished_at": None,
        "status": "PENDING",
        "methods": methods,
        "documents": documents,
        "settings": settings,
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "note": "Local text-layer extraction only: no OCR, no LLM, no network.",
    }
    atomic_write_json(path / "manifest.json", manifest)
    atomic_write_json(path / "state.json", initial_state(manifest))
    return path


def initial_state(manifest: dict) -> dict:
    total = sum(document["page_count"] for document in manifest["documents"])
    return {
        "schema": SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "status": "PENDING",
        "updated_at": utc_now(),
        "attempt": 0,
        "worker": None,
        "message": None,
        "methods": {
            method["method_id"]: {"status": "PENDING", "done": 0, "failed": 0, "total": total,
                                  "elapsed_ms": 0.0, "started_at": None, "finished_at": None, "error": None}
            for method in manifest["methods"]
        },
        "pages": {method["method_id"]: {} for method in manifest["methods"]},
    }


def load_manifest(run_path: Path) -> dict:
    return read_json(run_path / "manifest.json")


def load_state(run_path: Path) -> dict | None:
    try:
        return read_json(run_path / "state.json")
    except (OSError, ValueError):
        return None


# ----------------------------------------------------- page write / verify
def verify_page(run_path: Path, key: str, method_id: str, page: int) -> tuple[str, str | None]:
    """Return (state, reason): valid | failed | missing | invalid.

    valid  : .txt and .json exist, JSON parses, status COMPLETED, and the stored
             character count and SHA-256 match the .txt on disk.
    failed : a well-formed FAILED record (the extraction raised); no .txt expected.
    """
    text_path, json_path = page_paths(run_path, key, method_id, page)
    if not json_path.exists():
        if text_path.exists():
            return "invalid", "text file without JSON record (incomplete write)"
        if side_file_paths(run_path, key, method_id, page):
            return "invalid", "side file (e.g. tables) without JSON record (incomplete write)"
        return "missing", None
    try:
        record = read_json(json_path)
    except (OSError, ValueError) as error:
        return "invalid", f"JSON does not parse: {type(error).__name__}"
    if not isinstance(record, dict) or record.get("page") != page or record.get("method_id") != method_id:
        return "invalid", "JSON record does not match this page/method"
    if record.get("status") == "FAILED":
        return "failed", str(record.get("error") or "extraction failed")
    if record.get("status") != "COMPLETED":
        return "invalid", f"status is {record.get('status')!r}"
    if not text_path.exists():
        return "invalid", "text file missing"
    data = text_path.read_bytes()
    if sha256_bytes(data) != record.get("text_sha256"):
        return "invalid", "text SHA-256 does not match the JSON record"
    try:
        if len(decode_text(data)) != record.get("char_count"):
            return "invalid", "character count does not match the JSON record"
    except UnicodeDecodeError:
        return "invalid", "text file is not valid UTF-8"
    side_files = record.get("side_files") or {}
    if not isinstance(side_files, dict):
        return "invalid", "side_files is not a mapping"
    folder = method_dir(run_path, key, method_id)
    for name, digest in side_files.items():
        if not side_file_name_ok(name, page):
            return "invalid", f"unexpected side file name {name!r}"
        path = folder / name
        if not path.is_file():
            return "invalid", f"side file {name} missing"
        if sha256_bytes(path.read_bytes()) != digest:
            return "invalid", f"side file {name} does not match its SHA-256"
    return "valid", None


def side_file_name_ok(name: str, page: int) -> bool:
    """Side files belong to one page: page_NNN_<kind>.json|md (e.g. page_016_tables.json)."""
    return bool(re.fullmatch(rf"page_{page:03d}_[a-z]{{1,20}}\.(json|md)", name))


def side_file_paths(run_path: Path, key: str, method_id: str, page: int) -> list[Path]:
    folder = method_dir(run_path, key, method_id)
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.glob(f"page_{page:03d}_*") if side_file_name_ok(path.name, page))


def read_side_file(run_path: Path, key: str, method_id: str, page: int, name: str) -> bytes | None:
    if not side_file_name_ok(name, page):
        raise StorageError(f"Unexpected side file name {name!r}")
    path = method_dir(run_path, key, method_id) / name
    return path.read_bytes() if path.is_file() else None


def _quarantine(run_path: Path, key: str, method_id: str, page: int) -> list[str]:
    moved = []
    folder = method_dir(run_path, key, method_id) / "_quarantine"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for path in [*page_paths(run_path, key, method_id, page), *side_file_paths(run_path, key, method_id, page)]:
        if path.exists():
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / f"{path.stem}.{stamp}{path.suffix}"
            os.replace(path, target)
            moved.append(target.name)
    return moved


def write_page_result(run_path: Path, key: str, method_id: str, page: int, record: dict,
                      text: str | None, *, replace_approved: bool = False,
                      side_files: dict[str, bytes] | None = None) -> dict:
    """Write the page result: side files and .txt first, then the .json record (the commit marker).

    `side_files` (e.g. page_NNN_tables.json/.md of the processed method) are hashed into
    the record and re-verified by `verify_page`; they never replace or alter the .txt.

    Refuses to touch a valid completed page. An invalid/failed page is replaced only
    with `replace_approved=True`; its old files are moved to `_quarantine/` first.
    """
    state, _ = verify_page(run_path, key, method_id, page)
    if state == "valid":
        raise StorageError(f"Refusing to overwrite a valid completed page: {key}/{method_id}/page {page}")
    quarantined: list[str] = []
    if state in ("invalid", "failed"):
        if not replace_approved:
            raise StorageError(f"Page needs explicit approval to be re-extracted: {key}/{method_id}/page {page}")
        quarantined = _quarantine(run_path, key, method_id, page)
    text_path, json_path = page_paths(run_path, key, method_id, page)
    record = dict(record, page=page, method_id=method_id, document_key=key, schema=SCHEMA_VERSION)
    if quarantined:
        record["replaced_after_approval"] = quarantined
    if side_files:
        folder = method_dir(run_path, key, method_id)
        hashes = {}
        for name, data in side_files.items():
            if not side_file_name_ok(name, page):
                raise StorageError(f"Unexpected side file name {name!r}")
            atomic_write_bytes(folder / name, data)
            hashes[name] = sha256_bytes(data)
        record["side_files"] = hashes
    if text is not None:
        data = encode_text(text)
        record.update(text_file=text_path.name, text_sha256=sha256_bytes(data), char_count=len(text),
                      word_count=len(text.split()), is_empty=not text.strip())
        atomic_write_bytes(text_path, data)
    atomic_write_json(json_path, record)
    return record


def read_page(run_path: Path, key: str, method_id: str, page: int) -> tuple[dict | None, str | None]:
    text_path, json_path = page_paths(run_path, key, method_id, page)
    try:
        record = read_json(json_path)
    except (OSError, ValueError):
        record = None
    try:
        text = decode_text(text_path.read_bytes())
    except (OSError, UnicodeDecodeError):
        text = None
    return record, text


def scan_run(run_path: Path, manifest: dict | None = None) -> dict:
    """Re-verify every expected page on disk (authoritative status source)."""
    manifest = manifest or load_manifest(run_path)
    result = {"pages": {}, "counts": {"valid": 0, "failed": 0, "missing": 0, "invalid": 0, "total": 0},
              "methods": {}, "issues": []}
    for method in manifest["methods"]:
        method_id = method["method_id"]
        counts = {"valid": 0, "failed": 0, "missing": 0, "invalid": 0, "total": 0}
        for document in manifest["documents"]:
            key = document["doc_key"]
            for page in range(1, document["page_count"] + 1):
                state, reason = verify_page(run_path, key, method_id, page)
                result["pages"][(key, method_id, page)] = state
                counts[state] += 1
                counts["total"] += 1
                if state in ("invalid", "failed"):
                    result["issues"].append({"doc_key": key, "document": document["original_name"],
                                             "method_id": method_id, "page": page, "state": state, "reason": reason})
        result["methods"][method_id] = counts
        for name, value in counts.items():
            result["counts"][name] += value
    return result


def write_full_document(run_path: Path, key: str, method_id: str, page_count: int) -> Path:
    """Concatenate the saved page texts (`=== page N ===` separators, matching the GT format)."""
    parts = []
    for page in range(1, page_count + 1):
        state, reason = verify_page(run_path, key, method_id, page)
        _, text = read_page(run_path, key, method_id, page)
        body = text if state == "valid" and text is not None else f"[NOT AVAILABLE — {state}{': ' + reason if reason else ''}]"
        parts.append(f"=== page {page} ===\n{body}")
    path = method_dir(run_path, key, method_id) / "full_document.txt"
    atomic_write_text(path, "\n".join(parts) + "\n")
    return path


# -------------------------------------------------------------------- lock
def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock(run_path: Path) -> str:
    token = secrets.token_hex(8)
    payload = {"pid": os.getpid(), "host": socket.gethostname(), "token": token,
               "created_at": utc_now(), "heartbeat_at": utc_now(), "heartbeat_epoch": time.time()}
    try:
        with open(run_path / LOCK_NAME, "x", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except FileExistsError as error:
        raise StorageError("This run is locked by another worker (or a stale lock needs to be cleared)") from error
    return token


def heartbeat(run_path: Path, token: str) -> None:
    info = read_lock(run_path)
    if info and info.get("token") == token:
        info.update(heartbeat_at=utc_now(), heartbeat_epoch=time.time())
        atomic_write_json(run_path / LOCK_NAME, info)


def release_lock(run_path: Path, token: str) -> None:
    info = read_lock(run_path)
    if info and info.get("token") == token:
        (run_path / LOCK_NAME).unlink()


def read_lock(run_path: Path) -> dict | None:
    try:
        return read_json(run_path / LOCK_NAME)
    except (OSError, ValueError):
        return {"corrupt": True} if (run_path / LOCK_NAME).exists() else None


def lock_status(run_path: Path) -> dict:
    """{"locked": bool, "alive": bool, "stale": bool, ...}. Alive = worker process still running."""
    info = read_lock(run_path)
    if info is None:
        return {"locked": False, "alive": False, "stale": False}
    if info.get("corrupt"):
        return {"locked": True, "alive": False, "stale": True, "reason": "lock file unreadable"}
    same_host = info.get("host") == socket.gethostname()
    alive = same_host and _pid_alive(int(info.get("pid") or 0))
    age = time.time() - float(info.get("heartbeat_epoch") or 0)
    return {"locked": True, "alive": alive, "stale": not alive, "pid": info.get("pid"),
            "heartbeat_age_s": round(age, 1), "heartbeat_old": age > HEARTBEAT_STALE_S,
            "reason": None if alive else "worker process is not running"}


def clear_stale_lock(run_path: Path) -> None:
    """Explicit operator action only. Refuses while the worker process is alive."""
    status = lock_status(run_path)
    if status["alive"]:
        raise StorageError("The worker is still running; the lock is not stale")
    if status["locked"]:
        logs = run_path / "logs"
        logs.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        os.replace(run_path / LOCK_NAME, logs / f"stale_lock_cleared_{stamp}.json")  # kept as evidence


def request_cancel(run_path: Path) -> None:
    atomic_write_json(run_path / CANCEL_NAME, {"requested_at": utc_now()})


# --------------------------------------------------------- run status
def derive_status(run_path: Path, manifest: dict | None = None, scan: dict | None = None,
                  *, worker_finished: bool = False) -> dict:
    """Status from the files on disk; never claims COMPLETED unless every page verifies.

    `worker_finished=True` is used only by the worker itself after its last page, while it
    still holds the lock, so that its own lock is not reported as RUNNING."""
    manifest = manifest or load_manifest(run_path)
    scan = scan or scan_run(run_path, manifest)
    state = load_state(run_path) or {}
    lock = lock_status(run_path)
    counts = scan["counts"]
    done = counts["valid"] + counts["failed"]
    if lock["alive"] and not worker_finished:
        status = "RUNNING"
    elif state.get("attempt", 0) == 0 and done == 0:
        status = "PENDING"
    elif counts["missing"] or counts["invalid"]:
        status = "INTERRUPTED"
    elif counts["failed"] == counts["total"]:
        status = "FAILED"
    elif counts["failed"]:
        status = "PARTIAL"
    else:
        status = "COMPLETED"
    label = status if status in ("COMPLETED", "PENDING") else f"{status} — {done}/{counts['total']} pages"
    return {"status": status, "label": label, "counts": counts, "lock": lock, "scan": scan, "state": state}


def list_runs(root: Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted((path for path in root.iterdir() if path.is_dir() and RUN_ID_PATTERN.match(path.name)
                   and (path / "manifest.json").exists()), key=lambda path: path.name, reverse=True)


def delete_run(root: Path, run_id: str, confirmation: str) -> None:
    """Delete exactly one run folder, after typed confirmation. Never anything else."""
    if confirmation != run_id:
        raise StorageError("Confirmation text does not match the run id")
    root = Path(root).resolve()
    path = run_dir(root, run_id).resolve()
    if path.parent != root or not (path / "manifest.json").exists():
        raise StorageError("Not a benchmark run folder")
    if lock_status(path)["alive"]:
        raise StorageError("The run is still executing")
    shutil.rmtree(path)


# --------------------------------------------------------- display cache
def page_image(run_path: Path, key: str, page: int, dpi: int = 110) -> Path:
    """Render one page for DISPLAY only (cached in the run folder). Never used for text."""
    target = run_path / "page_images" / _checked_key(key) / f"page_{page:03d}_{int(dpi)}dpi.png"
    if target.exists():
        return target
    import pymupdf

    with pymupdf.open(original_pdf(run_path, key)) as document:
        pixmap = document[page - 1].get_pixmap(dpi=int(dpi))
        atomic_write_bytes(target, pixmap.tobytes("png"))
    return target


# ---------------------------------------------------------- ground truth
def save_ground_truth(run_path: Path, key: str, text: str, source_names: list[str], *, replace: bool = False) -> Path:
    folder = run_path / "ground_truth" / _checked_key(key)
    path = folder / "ground_truth.txt"
    if path.exists() and not replace:
        raise StorageError("Ground truth already attached; choose 'replace' explicitly")
    if path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        os.replace(path, folder / f"ground_truth.replaced_{stamp}.txt")
    atomic_write_text(path, text)
    atomic_write_json(folder / "ground_truth.json", {"saved_at": utc_now(), "sources": source_names,
                                                     "sha256": sha256_bytes(encode_text(text))})
    return path


def load_ground_truth(run_path: Path, key: str) -> str | None:
    path = run_path / "ground_truth" / _checked_key(key) / "ground_truth.txt"
    return decode_text(path.read_bytes()) if path.exists() else None
