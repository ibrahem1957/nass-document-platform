# nass-document-platform

A local platform that turns PDFs (digital, scanned or mixed; Arabic, English or both) into auditable,
retrieval-ready evidence:

**Upload PDF → Extract (Nass) → Prepare (DocPrep) → view results in Studio**, with a planned
**→ Index → Ask → Answers with sources (RAG)** extension.

| Folder | Component | Role |
|---|---|---|
| `studio/` | Document Processing Studio | **The application's user interface.** Upload, progress, results, downloads. |
| `nass/` | Nass (package `lab`) | PDF extraction: routing, text layers, PP-DocLayout regions, crops, Qari OCR, assembly, validation |
| `nass/runpod/` | Qari OCR service | GPU-side HTTP service (FastAPI); not run on the Studio machine |
| `docprep/` | DocPrep (package `docprep`) | Structure, relationships, quality findings, `prepared_document.json` and its safety contract |
| `rag/` | Future RAG | Placeholder. Not implemented. See `docs/RAG_HANDOFF.md`. |

**Start here:**
- `docs/ARCHITECTURE.md`, then `docs/PROCESSING_FLOW.md`;
- for AI coding agents: `CLAUDE.md` / `AGENTS.md`.

## Status

- **Working and tested:** Studio, Nass extraction code, DocPrep, and the prepared-document safety
  contract. See `docs/TESTING.md` for exact results, including **19 pre-existing Nass test failures**
  outside the extraction path.
- **Qari OCR service: not running.** The previous RunPod deployment is stopped. New extractions need a
  Qari service that you deploy or provide (`docs/RUNPOD_SETUP.md`).
- **RAG: not implemented.**
- **Licence:** none has been chosen for this project yet; the owner must add one before any wider
  distribution. A bundled test font (Amiri) keeps its own OFL licence
  (`nass/lab/fixtures/fonts/OFL.txt`).

## Requirements

- Windows 10/11 x64 (the verified platform), **Python 3.12 x64**
- Disk for the PP-DocLayout model (≈ 100s of MB) if you will extract scanned pages
- A Qari OCR service on a CUDA GPU host, for new extractions with scanned pages, visuals or Arabic
  spacing repair (`docs/RUNPOD_SETUP.md`)

## Install (Windows PowerShell)

These commands follow the original projects' documented setup. A full clean-machine install was **not**
re-executed during the handoff; `requirements-lock.txt` records the exact package set of the
environment in which all tests passed.

```powershell
git clone <repository-url> nass-document-platform
cd nass-document-platform

# 1. One shared virtual environment at the repository root
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 2. Studio + Nass + DocPrep dependencies
python -m pip install -r requirements.txt

# 3. Local layout detection for scanned pages (PP-DocLayout): PaddlePaddle FIRST, CPU or GPU (not both)
python -m pip install paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -r nass\lab\requirements-doclayout.txt
```

**4. Download the layout model once** (the application never downloads it). Unpack
`https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout_plus-L_infer.tar`
to a folder of your choice. It must contain `inference.yml`, `inference.pdiparams` and
`inference.json`.

**5. Configure:**

```powershell
copy .env.example nass\.env
notepad nass\.env
```

At least set:
- `LAYOUT_MODEL_DIR=C:\path\to\PP-DocLayout_plus-L_infer`
- `QARI_SERVICE_URL`, `QARI_DEPLOYMENT_MODE` (plus `QARI_TRUSTED_DEV_HOSTS` for RunPod), and
  `QARI_SERVICE_TOKEN`

`nass/.env` is git-ignored. Never commit it.

**6. Verify** (no network request, no model load):

```powershell
python scripts\check_setup.py
```

## Start the application

```powershell
cd studio
python -m studio.server --config config\studio.json
```

Open **http://127.0.0.1:8766**. Studio binds only to localhost and does not open a browser. Use
`--port <n>` to change the port.

`studio/config/studio.json` uses project-relative paths:
- `nass_root: ../nass` and `docprep_root: ../docprep`;
- `"current"` as the Python for both components, meaning the interpreter running Studio.

Processing data is written to `studio/data/jobs/` (git-ignored).

**To list already processed documents** under "Open Previously Processed Document", add entries to
`existing_documents`. See `studio/config/studio.example-with-library.json`.

## What works without RunPod / Qari

| Capability | Without a Qari service |
|---|---|
| Launch Studio, view and download previously processed documents | Yes |
| Run DocPrep on any saved Nass run folder (`python -m docprep.structure …`, see `docs/DOCPREP.md`) | Yes (no network) |
| All test suites | Yes |
| New extraction in Studio | **Refused** until `QARI_SERVICE_URL` and `LAYOUT_MODEL_DIR` are configured. With a configured but unreachable service, only PDFs producing no OCR jobs can finish (inferred from code; not re-verified). |
| Offline OCR (export jobs → GPU host batch → import) | Available in the Nass lab (`cd nass; python -m lab.launch`), not in Studio |

## Tests

```powershell
.\scripts\run_tests.ps1
```

Expected results at handoff:
- DocPrep: 439 passed, 4 skipped (private reference)
- Studio: 25 passed
- Nass: 731 passed, 19 pre-existing failures, 5 skipped

Details are in `docs/TESTING.md`.

## Documentation

| Document | Content |
|---|---|
| `docs/ARCHITECTURE.md` | Components, boundaries, data flow, inference versus deterministic steps |
| `docs/PROCESSING_FLOW.md` | Exact trace: upload → Nass → DocPrep → views |
| `docs/NASS.md` | Routing, layout, crops (60 px margin), Qari, reading order, titles, dedup, margin leakage, output contract |
| `docs/DOCPREP.md` | Stages, `prepared_document.json`, relationships, reliability, validation |
| `docprep/docs/RAG_CONSUMER_CONTRACT.md` | **Authoritative meaning of the retrieval-safety fields** |
| `docs/RAG_HANDOFF.md`, `docs/RAG_EVIDENCE_POLICY.md` | Brief and mandatory rules for the RAG implementer |
| `docs/RUNPOD_SETUP.md` | Qari service, RunPod, configuration, errors, costs, handoff dependencies |
| `docs/KNOWN_LIMITATIONS.md` | Verified limitations and the recommended RAG MVP scope |
| `docs/TESTING.md` | How to run tests; exact results; pre-existing failures |
| `studio/README.md`, `docprep/README.md`, `nass/lab/README.md` | Original component READMEs. Nass's describes the original machine layout. |

## Security

- **Secrets:** no secrets are committed. `.env` and runtime data, uploaded PDFs, crops, raw OCR,
  models, caches and logs are git-ignored (`.gitignore`).
- **Private data:** benchmark PDFs and outputs derived from them are not included.
- **Studio:** local, single-user and localhost only. It never displays or stores the Qari token.
