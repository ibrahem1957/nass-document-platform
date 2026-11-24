"""Offline Qari batch for an exported Automatic Extraction job folder (same recipe as qari_service.py).

    python qari_offline_batch.py <unzipped export folder>

Reads jobs.jsonl + images/, appends one line per job to results.jsonl (resumable: jobs already in
results.jsonl are skipped, nothing is overwritten). Each line has the same response schema as the
service, so the app can import it (hash-checked). Requires the GPU host packages of qari_service.py.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

from qari_service import RECIPE, SERVICE_VERSION, TransformersEngine, proven_from_env, transcribe


def main(folder: Path) -> int:
    jobs = [json.loads(line) for line in (folder / "jobs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    results_path = folder / "results.jsonl"
    done = set()
    if results_path.exists():
        done = {json.loads(line)["job_id"] for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [job for job in jobs if job["job_id"] not in done]
    print(f"{len(jobs)} jobs, {len(done)} done already, {len(todo)} to run", flush=True)
    if not todo:
        return 0
    from PIL import Image  # noqa: PLC0415

    proven, overrides = proven_from_env()
    engine = TransformersEngine(proven)
    for index, job in enumerate(todo, start=1):
        data = (folder / job["image"]).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != job["image_sha256"]:
            print(f"[{index}/{len(todo)}] {job['job_id']}: image hash mismatch, skipped", flush=True)
            continue
        with Image.open(folder / job["image"]) as source:
            image_format = source.format
            image = source.convert("RGB")
        started = time.time()
        outcome = transcribe(engine, image, job["kind"], proven)
        response = {"request_id": f"offline:{job['job_id']}", "kind": job["kind"], "service_version": SERVICE_VERSION,
                    "recipe": RECIPE + " (offline batch)",
                    "model": {"base": proven.base_model, "adapter": proven.adapter}, "proven_overrides": overrides,
                    "image": {"sha256": digest, "bytes": len(data), "format": image_format, "width": image.width,
                              "height": image.height, "resized": False},
                    "elapsed_s": round(time.time() - started, 2), **outcome}
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"job_id": job["job_id"], "image_sha256": digest, "response": response},
                                    ensure_ascii=False) + "\n")
        print(f"[{index}/{len(todo)}] {job['job_id']}: attempts={len(outcome['attempts'])} "
              f"flags={outcome['final_flags']}", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python qari_offline_batch.py <export folder>")
    sys.exit(main(Path(sys.argv[1])))
