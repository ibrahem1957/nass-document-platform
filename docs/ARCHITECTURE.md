# Architecture

`nass-document-platform` is a **document extraction and preparation platform**. It turns a PDF into
two auditable artifacts:

1. **Nass `document.json`**: extracted observations. These are page elements with verbatim text,
   geometry, OCR provenance and validation state.
2. **DocPrep `prepared_document.json`**: the same source elements, organized into a hierarchy, with
   relationships, quality findings and retrieval-safety metadata.

A retrieval-augmented question-answering layer (RAG) is a **planned extension** in `rag/`. It does not
exist yet and is not a source of truth.

## Components

| Folder | Component | Owns | Entry points (verified in source) |
|---|---|---|---|
| `studio/` | **Document Processing Studio**: the official user interface | Upload, orchestration of Nass then DocPrep, progress and status, result views, downloads, cancel and retry | `python -m studio.server --config config/studio.json` (serves on `127.0.0.1:8766`) |
| `nass/` | **Nass**: PDF extraction (Python package `lab`) | Page routing, text-layer extraction, PP-DocLayout regions, crops, Qari OCR requests, assembly, de-duplication, validation, raw OCR evidence | `lab.core.auto.pipeline` (used by Studio via `studio/studio/nass_runner.py`); the separate Streamlit lab: `python -m lab.launch` |
| `nass/runpod/` | **Qari OCR service** (runs on a GPU host, not on the Studio machine) | Qwen3-VL-4B + Qari adapter behind a FastAPI HTTP API | `uvicorn qari_service:app_from_environment --factory ...` |
| `docprep/` | **DocPrep**: document understanding and preparation (Python package `docprep`) | Structure, relationships, quality findings, prepared evidence, schema validation | `python -m docprep.structure`, `docprep.relations`, `docprep.quality`, `docprep.prepare` (and `docprep.evaluate*`, `docprep.inspector`) |
| `rag/` | **Future RAG module** (empty placeholder) | Indexing, retrieval, grounded answer generation, answer-time evidence enforcement | Not implemented |

### Boundaries

- **Studio:** contains no extraction, OCR, layout, relationship or quality logic. It calls Nass through
  a small adapter script, run as a subprocess, which imports `lab.core.auto`. It calls DocPrep through
  its command-line modules (subprocesses, `shell=False`).
- **DocPrep:** imports nothing from Nass. It only reads a finished Nass run folder, and it calls no
  model and no network service.
- **Nass:** calls exactly one external service, the configured Qari OCR endpoint, and only for images
  it has decided need OCR. PP-DocLayout layout detection runs locally.
- **Future RAG:** must consume validated `prepared_document.json` files. It must never consume raw OCR
  alone. Read `docs/RAG_HANDOFF.md`.

## Data flow

```text
                 ┌────────────────────────── studio (127.0.0.1:8766) ──────────────────────────┐
 PDF upload ───▶ │ job folder studio/data/jobs/job_<id>/                                         │
                 │   1  nass_runner.py run  ──▶  Nass run folder (document.json, layout.json, …) │──▶ Qari OCR service
                 │   2  python -m docprep.structure  ┐                                           │    (GPU host, e.g. RunPod)
                 │   3  python -m docprep.relations  │  each reads the same Nass run folder      │
                 │   4  python -m docprep.quality    │  and the previous stage's output folder    │
                 │   5  python -m docprep.prepare    ┘  ──▶ prepared_document.json (+ validation)  │
                 │   6  immutable JSON snapshots + Markdown reading views shown in the UI        │
                 └──────────────────────────────────────────────────────────────────────────────┘
                                                   │
                                                   ▼
                               future rag/: index → retrieve → grounded answer
```

The step-by-step trace, with real file names, is in `docs/PROCESSING_FLOW.md`.

## Runtime

- **Shared environment:** one Python 3.12 x64 virtual environment at the repository root
  (`.venv`). Studio's `config/studio.json` uses `"nass_python": "current"` and
  `"docprep_python": "current"`: the subprocesses run with the interpreter that runs Studio.
- **Studio:** binds only to `127.0.0.1`. It is a local, single-user development application, not a
  multi-tenant server.
- **Nass:** reads its settings from `nass/lab/config/auto_extraction.toml`, then `nass/.env`, then the
  real environment (later sources win).
- **DocPrep:** reads `docprep/config/docprep.toml`. Studio always passes explicit `--out` folders.

## Model inference versus deterministic code

| Operation | Kind | Where |
|---|---|---|
| Page routing (DIGITAL / SCANNED / MIXED) | Deterministic (PyMuPDF text-layer and image analysis) | Nass, local |
| Text-layer extraction of digital pages | Deterministic | Nass, local |
| Layout boxes (chart, image, table, figure_title, text, …) | **Model inference** (PP-DocLayout_plus-L, PaddlePaddle) | Nass, local CPU or GPU |
| Region selection, title attachment, crops with a 60 px margin | Deterministic rules on the detected boxes | Nass, local |
| OCR of crops and scanned pages | **Model inference** (Qwen3-VL-4B-Instruct + Qari-OCR-0.4.0 adapter) | Qari service, remote GPU |
| Assembly, de-duplication, reading-order placement, validation flags | Deterministic | Nass, local |
| Headings, hierarchy, relationships, quality findings, prepared document | Deterministic (lexical, typographic and geometric rules; no embeddings, no LLM) | DocPrep, local |
| Future retrieval and answer generation | Model inference (to be chosen) | rag/, not implemented |

## Source of truth

1. The **original PDF** is the ultimate source.
2. **Nass outputs** are extracted observations, not guaranteed facts. OCR can be wrong, and is never
   silently corrected.
3. **DocPrep** adds structure and confidence assessments. It never invents source text, numbers, cells,
   coordinates or relationships.
4. **RAG** (future) must cite source elements and pages, and must not invent missing information.
