"""Extraction worker. Runs as its own process so a page refresh or a Streamlit
restart does not lose or stop the job:

    python -m lab.local_pdf_extraction.runner --run-dir <run folder> [--resume-request <file>]

Guarantees:
* one worker per run (`.lock`, created exclusively; stale locks are cleared only by
  an explicit operator action);
* each library x page call is isolated: an exception is recorded as a FAILED page
  and the next page/library continues;
* after every page the result files are written first, then `state.json` (atomic);
* only pages that are MISSING on disk are processed. Invalid (corrupt/incomplete)
  or FAILED pages are re-extracted only when listed in an approved resume request,
  and valid completed pages are never touched.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from lab.local_pdf_extraction import diagnostics, exports, extractors, storage

PROJECT_DIR = Path(__file__).resolve().parents[2]


class Cancelled(Exception):
    pass


def method_entries(methods: list[str], reorder_settings: dict) -> list[dict]:
    entries = []
    for method_id in methods:
        info = extractors.METHOD_BY_ID[method_id]
        status = extractors.availability(info)
        entries.append({
            "method_id": method_id, "label": info.label, "library": info.library,
            "library_version": status["version"], "available": status["available"],
            "availability_error": status["error"], "output_type": info.output_type, "api": info.api,
        })
    return entries


def create_resume_request(run_path: Path, methods: list[str] | None, approved: list[dict]) -> Path:
    folder = run_path / "resume_requests"
    folder.mkdir(exist_ok=True)
    number = len(list(folder.glob("request_*.json"))) + 1
    path = folder / f"request_{number:04d}.json"
    storage.atomic_write_json(path, {"created_at": storage.utc_now(), "methods": methods,
                                     "approved_reextract": [
                                         {"doc_key": item["doc_key"], "method_id": item["method_id"],
                                          "page": int(item["page"])} for item in approved]})
    return path


def plan_work(run_path: Path, manifest: dict, methods: list[str], approved: set[tuple]) -> dict:
    """{method_id: {doc_key: [(page, replace_approved, state_before)]}}: missing + approved invalid/failed."""
    work: dict = {}
    for method_id in methods:
        for document in manifest["documents"]:
            key = document["doc_key"]
            for page in range(1, document["page_count"] + 1):
                state, _ = storage.verify_page(run_path, key, method_id, page)
                if state == "missing":
                    work.setdefault(method_id, {}).setdefault(key, []).append((page, False, state))
                elif state in ("invalid", "failed") and (key, method_id, page) in approved:
                    work.setdefault(method_id, {}).setdefault(key, []).append((page, True, state))
    return work


class _State:
    def __init__(self, run_path: Path, manifest: dict):
        self.path = run_path
        self.data = storage.load_state(run_path) or storage.initial_state(manifest)
        for method in manifest["methods"]:
            self.data["methods"].setdefault(method["method_id"], {})
            self.data["pages"].setdefault(method["method_id"], {})

    def save(self) -> None:
        self.data["updated_at"] = storage.utc_now()
        storage.atomic_write_json(self.path / "state.json", self.data)

    def method(self, method_id: str) -> dict:
        return self.data["methods"][method_id]

    def page(self, method_id: str, key: str, page: int, status: str) -> None:
        self.data["pages"][method_id].setdefault(key, {})[str(page)] = status


_COUNT_KEY = {"valid": "done", "failed": "failed", "invalid": "invalid", "missing": "missing"}


def _set_counts(state: _State, method_id: str, counts: dict) -> None:
    state.method(method_id).update(done=counts["valid"], failed=counts["failed"], total=counts["total"],
                                   invalid=counts["invalid"], missing=counts["missing"])


def _bump(method_state: dict, before: str, after: str) -> None:
    """Incremental count update (avoids re-hashing every page after each page)."""
    method_state[_COUNT_KEY[before]] = max(0, int(method_state.get(_COUNT_KEY[before], 0)) - 1)
    method_state[_COUNT_KEY[after]] = int(method_state.get(_COUNT_KEY[after], 0)) + 1


def _record(manifest: dict, document: dict, entry: dict, attempt: int) -> dict:
    return {
        "run_id": manifest["run_id"], "document_name": document["original_name"],
        "document_sha256": document["sha256"], "method_label": entry["label"], "library": entry["library"],
        "library_version": entry["library_version"], "output_type": entry["output_type"], "api": entry["api"],
        "attempt": attempt, "created_at": storage.utc_now(),
    }


def run(run_path: Path, *, resume_request: Path | None = None, registry: dict | None = None) -> dict:
    """Execute (or resume) a run in this process. Returns the derived final status."""
    run_path = Path(run_path)
    manifest = storage.load_manifest(run_path)
    token = storage.acquire_lock(run_path)
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(5.0):
            try:
                storage.heartbeat(run_path, token)
            except OSError:
                pass

    threading.Thread(target=beat, daemon=True).start()
    state = _State(run_path, manifest)
    try:
        cancel = run_path / storage.CANCEL_NAME
        if cancel.exists():  # an old cancel request must not stop this new attempt
            (run_path / "logs").mkdir(exist_ok=True)
            os.replace(cancel, run_path / "logs" / f"cancel_handled_{int(time.time())}.json")
        request = storage.read_json(resume_request) if resume_request else {}
        all_methods = [method["method_id"] for method in manifest["methods"]]
        methods = [m for m in (request.get("methods") or all_methods) if m in all_methods]
        approved = {(item["doc_key"], item["method_id"], int(item["page"])) for item in request.get("approved_reextract", [])}
        work = plan_work(run_path, manifest, methods, approved)
        state.data.update(status="RUNNING", attempt=int(state.data.get("attempt", 0)) + 1, message=None,
                          worker={"pid": os.getpid(), "started_at": storage.utc_now(), "finished_at": None,
                                  "resume_request": resume_request.name if resume_request else None})
        attempt = state.data["attempt"]
        scan = storage.scan_run(run_path, manifest)
        for method_id in all_methods:
            _set_counts(state, method_id, scan["methods"][method_id])
        state.save()
        entries = {entry["method_id"]: entry for entry in manifest["methods"]}
        settings = manifest.get("settings", {})
        for method_id in methods:
            _run_method(run_path, manifest, state, entries[method_id], work.get(method_id, {}), settings,
                        attempt, registry)
        # This worker still holds the lock here; it is finished, so the lock must not count as RUNNING.
        status = storage.derive_status(run_path, manifest, worker_finished=True)
        state.data.update(status=status["status"], message=None)
    except Cancelled:
        state.data.update(status="INTERRUPTED", message="Stopped by the operator (cancel request)")
        for method in state.data["methods"].values():
            if method.get("status") == "RUNNING":
                method["status"] = "INTERRUPTED"
    except BaseException as error:  # noqa: BLE001 - record, then re-raise
        state.data.update(status="INTERRUPTED", message=f"Worker stopped: {type(error).__name__}: {error}")
        for method in state.data["methods"].values():
            if method.get("status") == "RUNNING":
                method["status"] = "INTERRUPTED"
        raise
    finally:
        stop.set()
        if state.data.get("worker"):
            state.data["worker"]["finished_at"] = storage.utc_now()
        state.save()
        try:
            exports.write_summary(run_path, worker_finished=True)
        except Exception:  # noqa: BLE001 - summary is derived; never mask the run outcome
            traceback.print_exc()
        storage.release_lock(run_path, token)
        final = storage.derive_status(run_path, manifest)
        manifest = storage.load_manifest(run_path)
        manifest.update(status=final["status"], finished_at=storage.utc_now(),
                        last_counts=final["counts"])
        storage.atomic_write_json(run_path / "manifest.json", manifest)
    return storage.derive_status(run_path)


def _run_method(run_path: Path, manifest: dict, state: _State, entry: dict, work: dict, settings: dict,
                attempt: int, registry: dict | None) -> None:
    method_id = entry["method_id"]
    method_state = state.method(method_id)
    if not work:
        if method_state.get("status") in (None, "PENDING", "RUNNING", "INTERRUPTED"):
            method_state["status"] = "COMPLETED" if method_state["done"] else "FAILED"
        state.save()
        return
    method_state.update(status="RUNNING", started_at=storage.utc_now(), error=None)
    state.save()
    started = time.perf_counter()
    ocr_rule = diagnostics.needs_ocr_settings(settings)  # None for runs created before the rule existed
    available = extractors.availability(extractors.METHOD_BY_ID[method_id]) if registry is None else {"available": True}
    for document in manifest["documents"]:
        key = document["doc_key"]
        pages = work.get(key, [])
        if not pages:
            continue
        base = _record(manifest, document, entry, attempt)
        session = None
        open_error = None
        if not available["available"]:
            open_error = f"library unavailable: {available['error']}"
        else:
            try:
                session = extractors.open_session(method_id, storage.original_pdf(run_path, key), settings, registry)
            except Exception as error:  # noqa: BLE001
                open_error = f"could not open document: {type(error).__name__}: {error}"
        try:
            for page, replace, before in pages:
                if (run_path / storage.CANCEL_NAME).exists():
                    raise Cancelled()
                method_state["current"] = f"{document['original_name']} p{page}"
                state.page(method_id, key, page, "RUNNING")
                state.save()
                record = dict(base)
                if open_error:
                    record.update(status="FAILED", error=open_error, error_type="open", extraction_ms=0.0)
                    storage.write_page_result(run_path, key, method_id, page, record, None, replace_approved=replace)
                else:
                    t0 = time.perf_counter()
                    try:
                        output = session.extract(page)
                        elapsed = (time.perf_counter() - t0) * 1000
                        text = output.text
                        if not isinstance(text, str):
                            raise TypeError(f"library returned {type(text).__name__}, not str")
                    except Exception as error:  # noqa: BLE001 - isolate every library x page call
                        record.update(status="FAILED", error=f"{type(error).__name__}: {error}",
                                      error_type="extract", extraction_ms=(time.perf_counter() - t0) * 1000)
                        storage.write_page_result(run_path, key, method_id, page, record, None, replace_approved=replace)
                    else:
                        rule = ocr_rule if entry["output_type"] == "processed" else None
                        if (output.extra.get("reorder") or {}).get("decision") == "mainly_ltr":
                            rule = None  # mainly-LTR page: the single-letter ratio is an indicator only
                        record.update(status="COMPLETED", error=None, extraction_ms=round(elapsed, 3),
                                      diagnostics=diagnostics.page_indicators(text, rule), **output.extra)
                        storage.write_page_result(run_path, key, method_id, page, record, text, replace_approved=replace,
                                                  side_files=getattr(output, "files", None) or None)
                verified, _ = storage.verify_page(run_path, key, method_id, page)
                state.page(method_id, key, page, {"valid": "COMPLETED", "failed": "FAILED"}.get(verified, "INVALID"))
                _bump(method_state, before, verified)
                method_state["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
                state.save()
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception:  # noqa: BLE001
                    traceback.print_exc()
            storage.write_full_document(run_path, key, method_id, document["page_count"])
    method_state.pop("current", None)
    method_state.update(status="COMPLETED" if method_state["done"] else "FAILED", finished_at=storage.utc_now())
    if not method_state["done"]:
        method_state["error"] = "no page completed"
    state.save()


def launch_worker(run_path: Path, resume_request: Path | None = None) -> int:
    """Start the worker as a separate, detached local process. Returns its PID."""
    run_path = Path(run_path)
    if storage.lock_status(run_path)["locked"]:
        raise storage.StorageError("The run is locked (running, or a stale lock must be cleared first)")
    logs = run_path / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / f"worker_{len(list(logs.glob('worker_*.log'))) + 1:04d}.log"
    command = [sys.executable, "-m", "lab.local_pdf_extraction.runner", "--run-dir", str(run_path)]
    if resume_request:
        command += ["--resume-request", str(resume_request)]
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    env = dict(os.environ, PYTHONPATH=str(PROJECT_DIR), PYTHONIOENCODING="utf-8")
    with open(log_path, "ab") as log:
        process = subprocess.Popen(command, cwd=PROJECT_DIR, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, env=env, **kwargs)
    return process.pid


def terminate_worker(run_path: Path) -> bool:
    """Explicit operator action: stop a worker that does not react to a cancel request."""
    status = storage.lock_status(run_path)
    if not status.get("alive") or not status.get("pid"):
        return False
    pid = int(status["pid"])
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
    else:
        os.kill(pid, 15)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume-request")
    args = parser.parse_args(argv)
    run_path = Path(args.run_dir).resolve()
    if not storage.RUN_ID_PATTERN.match(run_path.name) or not (run_path / "manifest.json").exists():
        print(f"Not a benchmark run folder: {run_path}", file=sys.stderr)
        return 2
    resume = Path(args.resume_request).resolve() if args.resume_request else None
    if resume and resume.parent != run_path / "resume_requests":
        print("Resume request must be inside the run's resume_requests folder", file=sys.stderr)
        return 2
    print(f"{storage.utc_now()} worker {os.getpid()} starting {run_path.name}", flush=True)
    final = run(run_path, resume_request=resume)
    print(f"{storage.utc_now()} finished: {final['label']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
