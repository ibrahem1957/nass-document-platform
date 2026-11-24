# Document Processing Studio

A local, single-user integration application for the existing Nass Automatic Extraction pipeline and
the standalone DocPrep project. Upload one PDF, run the real pipelines in dependency order, and read
two deliberately separate results:

1. **Nass — Extraction:** exact extracted source strings, page by page.
2. **DocPrep — Organized Document:** the same source evidence organized by the saved DocPrep
   hierarchy, lists, visuals, and logical tables.

This project contains no extraction, OCR, layout, relationship, or quality business logic. It invokes
the existing supported Nass Python API and DocPrep CLIs.

## Launch

In the `nass-document-platform` monorepo, with the root virtual environment active (see the root
`README.md` for installation):

    cd studio
    python -m studio.server --config config\studio.json

Open http://127.0.0.1:8766. The server is hard-bound to localhost and does not open a browser.

Studio's own runtime code uses only Python 3.12's standard library. Pytest is needed only for tests.

## Nass and DocPrep prerequisites

`config/studio.json` uses paths relative to this `studio/` folder (`nass_root: ../nass`,
`docprep_root: ../docprep`). `"current"` as the Python interpreter means that Nass and DocPrep run with
the interpreter running the Studio, i.e. the shared root virtual environment.

New extractions require a configured Qari endpoint and a local PP-DocLayout model in `nass/.env`
(see the root `.env.example` and `docs/RUNPOD_SETUP.md`). The readiness banner reports what is
missing. Readiness inspection does not contact the service, load the layout model, or download anything.
The upload screen shows a small informational RunPod notice, but processing starts immediately when
the user clicks Process Document. There is no consent checkbox or extra approval step. The Studio
never displays or stores the token.

## Pipeline and paths

Each upload receives a cryptographically random ID and its own folder:

    data/jobs/job_<id>/
        original/source.pdf
        processing_status.json
        diagnostics/<failed-stage>.json
        upstream/
            nass_results/automatic_extraction/run_<...>/
            docprep/
                structure/<write-once-run>/
                relationships/<write-once-run>/
                quality/<write-once-run>/
                prepared/<write-once-run>/
        nass/
            document.json
            extracted_content.md
            source_run_reference.json
        docprep/
            prepared_document.json
            organized_document.md
            source_run_references.json

The exact Nass run returned by pipeline.create_run is carried into every DocPrep command. Each
DocPrep output path is parsed from that command's explicit written result and validated to be under
the current job. The Studio never searches for a latest timestamped folder.

Execution order:

1. Nass pipeline.create_run.
2. Nass pipeline.execute for pending Qari jobs.
3. Nass assemble.finalize.
4. python -m docprep.structure.
5. python -m docprep.relations.
6. python -m docprep.quality.
7. python -m docprep.prepare.
8. Immutable JSON snapshots and deterministic Markdown reading views.

If DocPrep fails, the completed Nass snapshot and download remain available. Retry always creates a
new job folder. Cancellation terminates the current subprocess and preserves completed stages.

## Result fidelity

- Nass content is sorted by saved page and reading order.
- needs_review elements remain visible with a small review mark.
- Only elements explicitly marked included_in_composite are suppressed from the default Nass reading
  view; they remain in the authoritative JSON.
- DocPrep sections, nested lists, figures, and logical multi-page tables come only from
  prepared_document.json.
- Every prepared source element is represented once in the organized view.
- OCR mistakes, numbers, percentages, and currencies are never corrected.
- Uncertain content remains marked, and uncertain relationships are not presented as confirmed.
- JSON downloads are byte-identical to their saved authoritative artifacts.
- Both primary result tabs open in Readable View. Use JSON View inside either tab to lazily load and
  inspect the complete corresponding saved artifact: Nass `document.json` or DocPrep
  `prepared_document.json`. Switching views or primary tabs never starts processing.
- JSON View is safely rendered as text, supports two-axis scrolling and copying, and preserves JSON
  number tokens while adding indentation for inspection. Download JSON continues to return the exact
  original saved bytes.
- Markdown is deterministic and readable, but is explicitly not a replacement for JSON.

## Existing documents

The secondary “Open Previously Processed Document” control exposes the validated documents listed in
`existing_documents` of `config/studio.json` (empty in the repository; see
`config/studio.example-with-library.json`). It performs no pipeline work and provides no filesystem browser.

## Security model

- Local development only; not a production multi-tenant service.
- Binds only to 127.0.0.1.
- Explicit document/job routes; path traversal is rejected.
- Upload size, PDF signature, EOF, and filename are validated.
- Subprocesses use argument arrays with shell disabled.
- Source-derived values are rendered with DOM textContent; no source HTML is executed.
- CSP permits only same-origin local assets.
- Historical Nass and DocPrep outputs are read-only.
- No RAG, embeddings, retrieval, chat, or external API other than the already configured Nass Qari
  service automatically when the configured Nass pipeline needs it.

The upload workflow is: PDF upload → Process Document → Nass (RunPod when needed) → DocPrep →
separate Nass and DocPrep results.

## Tests

    python -m ruff check studio tests
    node --check studio\static\app.js
    python -m pytest -q

Routine tests use mock adapters and never invoke OCR, RunPod, paid inference, or DocPrep processing.
