# Processing flow: from an uploaded PDF to `prepared_document.json`

Traced from `studio/studio/server.py`, `jobs.py`, `adapters.py`, `nass_runner.py`, Nass
`lab/core/auto/pipeline.py` and the DocPrep CLIs.

## 1. Upload (Studio)

The browser sends `POST /api/jobs` (multipart, field `pdf`) to the local Studio server
(`127.0.0.1:8766`). Studio:

1. **Validates the file:**
   - size, at most `max_upload_mb` (200)
   - the `%PDF` signature and an EOF marker
   - the filename
2. **Creates the job folder:** `studio/data/jobs/job_<random id>/`, with
   `original/source.pdf` and `processing_status.json`.
3. **Starts a background worker thread.** Each stage runs as a subprocess with an argument array,
   `shell=False`, a timeout of `stage_timeout_s` (7200 s), and a cancellation hook.

## 2. Readiness (no network, no model load)

`GET /api/readiness` runs `nass_runner.py --nass-root <nass> readiness`, which:
- loads the Nass configuration (`auto_extraction.toml` → `nass/.env` → environment);
- checks the Qari endpoint **configuration**, without contacting it;
- checks that the PP-DocLayout model folder exists (`LAYOUT_MODEL_DIR`).

Readiness fails if:
- `QARI_SERVICE_URL` is missing;
- the deployment mode does not trust the configured host;
- the layout model folder is missing.

The token is never displayed, only whether one is present.

## 3. Nass extraction (`nass_runner.py run`)

Executed with the same Python interpreter (`"nass_python": "current"`):

1. `pipeline.check_service(config)`: refuses to start without a trusted, configured Qari endpoint.
2. **PP-DocLayout readiness:** refuses to start if the layout model is not ready.
3. **`pipeline.create_run(pdf, …)`:** writes
   `studio/data/jobs/job_<id>/upstream/nass_results/automatic_extraction/run_<...>/`. It runs:
   - the text-layer analysis and page routing;
   - layout detection on scanned and mixed pages;
   - region selection and title association;
   - crops and the masked page images;
   - the OCR job list (`plan/jobs.json`).
4. **Pending OCR jobs:** if any exist, a `QariClient` is opened with the URL, token, timeout, retry and
   deployment-trust settings, and `pipeline.execute` sends each job (`POST /v1/ocr`). Every attempt is
   recorded under `raw/`. The report-only digit second read follows; if it fails, it is skipped.
5. **Completeness:** if any job remains pending, unconfirmed or failed, the stage fails with
   "Nass is incomplete".
6. **`assemble.finalize`:** writes `document.json`, `document.md`, `validation_report.json`,
   `rag_chunks.jsonl` and, last, `execution_manifest.json`.
7. **Result line:** the adapter prints
   `STUDIO_RESULT={"ok":true,"run_dir":...,"counts":...}`. Studio checks that the run folder is inside
   the job and contains `document.json`.

**Consequence:** a PDF whose pages produce no OCR jobs (a purely textual digital PDF without visuals or
category-B Arabic text) completes without contacting Qari. However, a Qari URL and the layout model
must still be **configured**. Any PDF with visuals, scanned pages or category-B Arabic text needs a
**reachable** Qari service.

## 4. DocPrep (four subprocesses, `cwd = docprep/`)

```text
python -m docprep.structure <run> --out <job>/upstream/docprep/structure
python -m docprep.relations <run> --structure <s> --out <job>/upstream/docprep/relationships
python -m docprep.quality   <run> --structure <s> --relations <r> --out <job>/upstream/docprep/quality
python -m docprep.prepare   <run> --structure <s> --relations <r> --quality <q> --out <job>/upstream/docprep/prepared
```

- **Output path:** each stage must print exactly one `written: <path>` inside its job-specific root.
- **Single run:** the exact Nass run returned in step 3 is passed to every stage; Studio never searches
  for a "latest" folder.
- **On a non-zero exit:** a `diagnostics/<stage>.json` file is written, and the completed Nass results
  stay available.

## 5. Snapshots and views

Studio reads `prepared_document_validation.json` and refuses the job ("DocPrep prepared document has
validation errors") unless the summary has zero errors. Only then does it write:

```text
job_<id>/nass/document.json                (byte copy)  extracted_content.md  source_run_reference.json
job_<id>/docprep/prepared_document.json    (byte copy)  organized_document.md source_run_references.json
```

**Routes:**
- `GET /api/jobs/<id>`: status
- `GET /api/items/job/<id>/nass`: Nass reading view
- `GET /api/items/job/<id>/docprep`: organized DocPrep view
- `/pdf`: inline source PDF
- `/download/{nass-json,nass-markdown,prepared-json,prepared-markdown}`: JSON downloads are
  byte-identical to the saved artifacts
- `POST /api/jobs/<id>/cancel`, `POST /api/jobs/<id>/retry`: retry always creates a new job folder

**Previously processed documents:** listed from `existing_documents` in `config/studio.json`
(`GET /api/library`). They are served through `/api/items/library/<id>/…` without running any pipeline.

## 6. What reaches the future RAG layer

**Only** `prepared_document.json` files whose `prepared_document_validation.json` reports
`"error": 0`. That includes `json_schema_valid`.

These must not be used as RAG inputs:
- the Nass `rag_chunks.jsonl`
- the Markdown views

They do not carry DocPrep's safety contract.

## Verified in this repository (see docs/TESTING.md)

- **Studio:** launches from `studio/`. The UI assets, readiness, library listing, reading views, PDF and
  byte-identical JSON download routes were exercised against a saved run.
- **Readiness:** with a loopback Qari URL and the local layout model configured, readiness reports
  ready without any network request.
- **Not executed:** a complete new upload → Qari → DocPrep run, because the Qari/RunPod service is
  stopped and must not be started without authorization.
