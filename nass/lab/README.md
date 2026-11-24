# AgenticDoc Extraction Lab — E1-A

> **Monorepo note:** this README was written for the original standalone layout. In
> `nass-document-platform` the Nass root (formerly `Project sdaia`) is `nass/`, the shared virtual
> environment is the repository-root `.venv`, and Nass reads `nass/.env`. `D:\agenticdoc_e2\...` paths
> are examples from the original development machine. The application UI is Studio (`../studio`); this
> Streamlit lab is an auxiliary evaluation tool.

## E2 offline / air-gap preflight (Windows PowerShell)

The E2 model registry never downloads weights. Review
`docs/local/AIRGAP_DEPLOYMENT_MANIFEST.json`, then run:

Run the preflight with the **model environment** (it checks that environment's
installed packages against the lock) and with `HF_HOME` pointing at the verified
snapshots. Inference is refused if the model would load files from any other
directory than the ones the preflight hashed.

```powershell
cd "<repository>\nass"
$env:PYTHONPATH = (Get-Location).Path
$env:HF_HOME = "D:\agenticdoc_e2\hf"
$env:AGENTICDOC_RESULTS_DIR = "D:\agenticdoc_e2\results"
& "D:\agenticdoc_e2\venv-paddleocr-vl\Scripts\python.exe" -m lab.core.e2.airgap paddleocr-vl-1.6
& "D:\agenticdoc_e2\venv-paddleocr-vl\Scripts\python.exe" -m lab.launch
```

Open **E2 OCR/VLM evaluation** and select PDF, page, model, and mode. The page
shows the exact revision, dependencies, blockers, and saved history. Preflight
and manual runs are write-once under `D:\agenticdoc_e2\results`. Exit code `2`
means the deployment is not yet air-gap ready; it is not a successful promotion.

**Which environment:** start the lab from the model venv on D: (as above) to run
models. On 2026-11-17 it received Streamlit 1.67.0 and its 22 dependencies from
the hash-locked `lab/config/airgap/windows-e2-ui-requirements.lock`, with no
existing package changed. Launched from `.venv` (no torch), the lab can only
review stored runs. The official full-page mode needs a separate, approved venv
because PaddleX pins `numpy<2.4` and `PyYAML==6.0.2`; see
`docs/local/E2B2_FULL_PAGE_INSTALL_PLAN.md`.

Always start the lab with `python -m lab.launch` (127.0.0.1 only, usage
statistics off), not `streamlit run` or the `streamlit.exe` shim, which
Application Control may block. Do not weaken WDAC, AppLocker, Smart App Control,
or Defender.

For manual E2 acceptance, choose a saved run and page in **E2 OCR/VLM
evaluation**, inspect the original beside exact RAW output, then use **Manual
user review** to record PASS, FAIL, or NEEDS REVIEW, notes, and error categories.
Reviews are separate from automated metrics and append-only under
`D:\agenticdoc_e2\results\reviews`. Repeat for every model; tests alone do not
accept E2-B2.

The Extraction Lab is a local Streamlit tool for comparing the raw text produced by PyMuPDF, pypdfium2, pdfplumber, and pypdf against the rendered PDF page. It is independent of `src/` and does not change the existing RAG application.

E1-A evaluates native PDF text layers only. It is not OCR and does not select a production extractor. A user must manually upload and inspect a PDF before E1-A can be accepted; automated tests alone are insufficient.

## Windows PowerShell setup

Open PowerShell and run:

```powershell
cd "<repository>\nass"

py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r .\lab\requirements-lab.txt
```

The verified Windows runtime is Python 3.12 x64. If `py -3.12` is unavailable, install Python 3.12 and recreate the environment; Python 3.11 and 3.13 have not yet been verified for this checkpoint.

```powershell
python -m venv .venv
```

## Launch Streamlit on Windows

With the virtual environment activated and PowerShell still inside `Project sdaia`:

```powershell
python -m lab.launch
```

**Always launch through `lab.launch`.** Don't use `streamlit run` directly. Streamlit reads `.streamlit/config.toml` only from the current directory, so started from `Project sdaia` it ignores `lab/.streamlit/config.toml`. It then binds to **all network interfaces** (other machines on the LAN can reach the lab and its uploaded documents), queries an internet service for the machine's external IP, and enables browser usage statistics. `lab.launch` always passes `--server.address 127.0.0.1` and `--browser.gatherUsageStats false`, and refuses to override them. Extra Streamlit options can still be passed, e.g. `python -m lab.launch --server.port 8502`.

Open the local URL shown in PowerShell, normally `http://127.0.0.1:8501`.

### Windows application-control note

The lab's document details and extraction results use Streamlit text/image primitives and do not require Pandas. Pandas remains a transitive Streamlit dependency, but the E1-A pages deliberately avoid dataframe-capable display calls because some managed Windows environments block unsigned NumPy extension modules under WDAC or an equivalent enterprise policy.

If Windows reports `Application Control policy has blocked this file`, inspect the **Microsoft-Windows-CodeIntegrity/Operational** log and record the blocked path and policy ID. Do not disable Defender, Smart App Control, WDAC, AppLocker, or another security control, and do not assume reinstalling NumPy will change an enterprise signing decision.

## Manual extraction comparison

1. Open **Documents** in the Streamlit sidebar.
2. Upload a PDF. The file stays in a temporary workspace and is never committed automatically.
3. Verify that the original page renders. For Arabic, verify shaping, RTL direction, diacritics, and Arabic/Western numerals visually.
4. Open **Run Text Extraction**, select the PDF and pages, and run all four local extractors. Enable **Re-run (bypass cache)** when you need a fresh execution.
5. Open **Compare**. Confirm that the original page is visible beside every extractor's **RAW — exactly as extracted** text.
6. Toggle the block-order overlay and invisible-character markers. Confirm that Arabic text is readable and that each result corresponds to the same selected page.
7. Record any mismatch or startup/rendering problem in the E1-A review; do not mark E1-A complete until this walkthrough succeeds.

## Temporary data and privacy

- Uploaded documents, page PNGs, and extracted text are stored beneath `%TEMP%\agenticdoc-lab\ws_*` by default.
- **Never commit or push user PDFs, images, extracted text, or workspace directories.** Only synthetic fixtures under the project may be committed.
- Ending a workspace deletes its local directory. Abandoned workspaces are deleted after the configured TTL (24 hours by default).
- E1 has no API adapters and makes no external calls. All E1-A extraction runs stay local.
- The application does not log extracted document text.

## Generate synthetic fixtures and run the headless bench

```powershell
python -m lab.fixtures.generate_fixtures
python -m lab.core.bench --fixtures .\lab\fixtures\generated --out .\lab_exports\e1_fixture_bench
```

Generated fixtures and bench output are ignored by Git. They contain synthetic content only.
The bench reports text-extraction time separately from adapter startup and optional block-overlay generation; do not add those columns together or interpret this synthetic bench as an extractor selection.

## Run E1-A checks

```powershell
python -m ruff check .\lab .\tests\lab
python -m pytest .\tests\lab --cov=lab.core --cov-report=term-missing
```

## Adding an adapter

E1-A adapters implement the protocol in `lab/core/adapters/base.py`. Add one module with `create_adapter(info)` and one entry in `lab/config/adapters.toml`. Do not add OCR, vision, table, or chart adapters during E1-A.

## E1-B correction and Ground Truth workflow

E1-B keeps every RAW run immutable and creates a separately identified derived run for a selected, versioned correction variant. Corrections are extractor-specific: they are never enabled globally and are never applied automatically to an upload.

1. Launch with `python -m lab.launch` and upload the PDF on **Documents**.
2. Run all four local extractors on **Run Text Extraction**.
3. On **Ground Truth**, import the matching benchmark JSON or a UTF-8 TXT whose pages are separated with form-feed (`\f`). Check **Mark imported pages verified** only after checking the source visually; draft pages never produce accuracy metrics.
4. On **Compare**, select an extractor-specific variant. The original page, immutable RAW, CLEANED output, escaped diff, warnings, Ground Truth, and before/after metrics remain visible together.
5. Use **Save derived ... run** only after review. This writes a separate local artifact with `raw_run_id` and variant version; it does not overwrite RAW.
6. Record a human rating in the same page. **Results and local export** keeps automatic metrics and human review separate and excludes PDFs and extracted text from its review export.

`digit_run_fix` and `bidi_reorder` are experimental and hidden by default. They can be shown only with the explicit **Show experimental unsafe correction variants** checkbox for audit/research; never enable them automatically. `digit_run_fix` is producer-specific: it assumes the PDF producer stored Arabic-Indic numbers in visual order and the extractor (PyMuPDF/pypdf on the Skia/Chrome benchmark) returns them reversed. On a PDF whose text layer is already logical it corrupts correct values (`٣٣٫٨٤٪ → ٤٨٫٣٣٪`, `١٢٣٬٤٥٦ → ٦٥٤٬٣٢١`), and other producers (Word, LibreOffice, InDesign, scanned+OCR) have not been measured. Every change it makes carries this warning and must be checked against the page image. `bidi_mirror_fix` changes only captured local RTL signatures **and** only on pages with visual-mirroring evidence (more reversed `)…(` pairs than normal `(…)` pairs); otherwise operators/brackets such as logical `≥ ٪٥٠` are left unchanged with a warning. Derived runs carry `variant_version` 2. `unmapped_marker` only exposes unrecoverable NUL/replacement glyphs; it cannot reconstruct them. Such pages are candidates for a later OCR/VLM evaluation, not proof of an OCR winner.

Automatic metrics disclose both the numerator and denominator. Sensitive-value results distinguish pooled item accuracy from the unweighted page mean, and reading-order tau includes matched/reference line coverage. Derived variants also report values that were correct in RAW but wrong after cleaning, plus newly introduced wrong values. English safety requires both the CER threshold and zero sensitive-value regressions.

The regex number metric is computed on `full_text_in_reading_order`, which excludes table cells. When the Ground Truth JSON provides per-page `numeric_values`, the lab also scores those curated values exactly (same digit script, standalone occurrence, duplicates counted per occurrence, overlapping values never double counted) and reports running text and table cells separately, together with RAW→CLEANED fixed/regressed/new-wrong counts. The curated values are used only for scoring, never for extraction or correction. The field `*_page_weighted_accuracy` is a deprecated alias of `*_page_mean_accuracy` (an unweighted page mean).

To reproduce the local Arabic/English benchmark from PowerShell without any API call:

```powershell
python -m lab.core.e1b_bench `
  --arabic-pdf "C:\path\AgenticDoc_Arabic_Benchmark.pdf" `
  --arabic-gt "C:\path\AgenticDoc_Arabic_GroundTruth.json" `
  --english-pdf "C:\path\AgenticDoc_English_Benchmark.pdf" `
  --english-gt "C:\path\AgenticDoc_English_GroundTruth.json" `
  --out "C:\path\E1B_EVALUATION_REPORT.md"
```

## E2-A OCR/VLM evaluation (local only)

`lab/core/e2/` adds a model registry (`config/models_e2.toml`, pinned revisions), readiness checks, local backends (offline `transformers` from pinned Hugging Face snapshots; a loopback-only Ollama server), page rendering, static prompt templates, a child-process worker with load and per-page timeouts, and E2 metrics. A model receives only the rendered page image and a static prompt; ground truth and extracted text are never inputs. Remote inference is refused (`REMOTE_INFERENCE_ENABLED = False`). Nothing downloads weights: a model whose packages or weights are missing shows its exact status and the approval-gated next step.

The **OCR/VLM Evaluation** page shows readiness, the rendered page, the exact RAW response, metrics against verified ground truth, numeric discrepancies, a side-by-side view and a metrics-only JSON export. Mock models exist only in tests and are labelled MOCK wherever they appear.

E2 metrics extend E1-B: table *cell transcription* (exact standalone span) is separate from *row/column association* (value under the right row header and column in a parsed Markdown/HTML table); mixed Arabic/Latin lines; missing and extra words; time and peak RAM. `markup_strip` is a separately identified post-processing run; RAW model output is never changed.

```powershell
python -m lab.core.e2.e2_bench `
  --arabic-pdf "C:\path\AgenticDoc_Arabic_Benchmark.pdf" --arabic-gt "C:\path\AgenticDoc_Arabic_GroundTruth.json" `
  --english-pdf "C:\path\AgenticDoc_English_Benchmark.pdf" --english-gt "C:\path\AgenticDoc_English_GroundTruth.json" `
  --out "C:\path\E2A_COMPARISON_REPORT.md"
```

The evaluator writes metrics only and deletes its temporary workspaces. Never place real/user PDFs, Ground Truth text, extracted text, page images, or workspace data in Git.

## E2-B2 official PaddleOCR-VL 1.6 full-page pipeline

The model registry keeps two separate PaddleOCR-VL modes:

- **Element-level prompt on whole page** is the historical `OCR:` run. Its saved results are
  retained and are not relabelled as official pipeline results.
- **Official full-page pipeline** runs PP-DocLayoutV3 layout detection and reading order,
  crops the detected regions, recognizes each region with PaddleOCR-VL 1.6, and assembles
  structured Markdown/JSON. It has its own adapter id and write-once result directory.

The official mode never falls back to native PDF text or Ground Truth. Both model directories
must resolve to complete pinned local snapshots before its Run button is enabled. In particular,
the PaddleOCR API is always given explicit local `layout_detection_model_dir` and
`vl_rec_model_dir` paths, so a missing layout model is reported instead of being downloaded.

### Start the lab on Windows PowerShell

These commands launch the interface from the model venv, so the element-level mode can run.
The official mode stays visibly disabled until its separately approved layout weights and its
own full-page venv exist (see `docs/local/E2B2_FULL_PAGE_INSTALL_PLAN.md`).

```powershell
Set-Location "<repository>\nass"
$env:PYTHONPATH = (Get-Location).Path
$env:LAB_TMP_DIR = "D:\agenticdoc_e2\tmp\lab"
$env:AGENTICDOC_RESULTS_DIR = "D:\agenticdoc_e2\results"
$env:HF_HOME = "D:\agenticdoc_e2\hf"
& "D:\agenticdoc_e2\venv-paddleocr-vl\Scripts\python.exe" -m lab.launch
```

Use `lab.launch`, not `streamlit run`; the launcher binds only to `127.0.0.1` and disables
Streamlit telemetry. In the browser:

1. Create a workspace on **Home**, then upload a PDF on **Documents**.
2. Import and verify Ground Truth if accuracy metrics are needed.
3. Open **OCR/VLM Evaluation** and select the PDF, page, `PaddleOCR-VL 1.6`, and inference mode.
4. Check the readiness panel. A missing asset or package disables inference and shows the exact
   blocked next step.
5. After the assets are approved and present, choose one page for the first test and select
   **Run local inference**. Watch the progress indicator and any failure/truncation warning.
6. Inspect the original rendered page beside the exact RAW text and structured pipeline JSON.
   Check Ground Truth diffs, CER/WER, numbers, dates, identifiers, tables, row/column association,
   reading order, and detected errors.
7. Select old and new runs in **Runs to compare without overwriting history**. Every manual run is
   saved automatically under `D:\agenticdoc_e2\results` using model, revision, mode, and run id.

### Manual acceptance checklist

- [ ] The selected original page is the page that was sent to the model.
- [ ] Arabic, English, and mixed lines are readable and in the expected order.
- [ ] Layout regions and RAW recognition output are present; no native text was substituted.
- [ ] Tables retain cell text and sensible row/column relationships.
- [ ] Numbers, dates, identifiers, and comparison symbols were checked against the page.
- [ ] Repetition, truncation, empty output, and failures are clearly flagged.
- [ ] A new unique run directory exists on D: and historical runs still compare unchanged.
- [ ] The user records acceptance or rejection after hands-on inspection.

Automated tests and Claude review establish readiness only; they do not complete this checklist or
accept E2-B2 on the user's behalf.

## Local PDF Extraction Benchmark (page 9, local only)

Sidebar → **Local PDF Extraction Benchmark**. Direct text-layer extraction with PyMuPDF,
pdfplumber, pypdf, pypdfium2 (`PdfTextPage`) and optional pdfminer.six — no OCR, no LLM, no
network. Upload PDFs → select libraries → **Run Extraction**. The job runs in a separate local
worker process, so a page refresh or Streamlit restart does not stop it or lose results.

- Results: `%AGENTICDOC_RESULTS_DIR%\local_pdf_extraction\run_<timestamp>_<id>\` (default
  `D:\agenticdoc_e2\results\local_pdf_extraction`). Nothing outside that folder is written.
- Raw library outputs are saved byte-for-byte. `pymupdf_ar_reordered` is a separate PROCESSED
  method (conditional Arabic logical-order reconstruction; heuristic, not ground-truth verified).
  The frozen prototype it was ported from is `reference/ar_pdf_extract.py`.
- Interrupted runs show `INTERRUPTED — n/N pages`. **Resume** re-verifies saved pages, processes
  only missing pages, and re-extracts corrupt/failed pages only after explicit approval.
- Tests: `python -m pytest tests\lab\test_lpe_arabic_reorder.py tests\lab\test_lpe_pipeline.py tests\lab\test_lpe_ocr_links.py tests\lab\test_lpe_ui.py`

### Structured tables (processed method, optional)

When "Extract tables" is enabled for a run, the processed method `pymupdf_ar_reordered` also
saves `page_NNN_tables.json` and `page_NNN_tables.md` (PyMuPDF `find_tables()` + per-cell Arabic
reconstruction; PROCESSED output, not raw library output, not verified cell by cell). Each cell
keeps `raw_cell_text` (content-stream order, no reordering) and `processed_cell_text`; cells whose
numbers, dates, percentages or Latin tokens changed are `uncertain` (⚠). Tables whose geometry is
inconsistent are `uncertain`; nothing is invented and tables are never merged across pages
(`possible_continuation_of` is metadata only). `page_NNN.txt`, `full_document.txt` and all raw
library outputs are unchanged.

**For RAG / downstream use:** the page JSON lists `table_regions` (table bboxes). The page text
still contains the text of those regions, so for each region use EITHER the page text OR the
structured table — never both, or the content is duplicated.

### Single-letter Arabic tokens, "Needs OCR?" and ground truth

- *Single-letter Arabic token ratio* (indicator, every library and the processed method): share of
  Arabic tokens (whitespace-split) that are one Arabic letter. High on PDFs whose text layer puts
  spaces inside words (e.g. Chrome/Skia: "ا خ ت بار"), ~0 on normal Arabic text.
- "Needs OCR?" (recommendation only) also flags a processed page when that ratio exceeds the
  run's threshold (default 0.10, pages with at least 10 Arabic tokens): "words split by spaces in
  the PDF text layer". The threshold is saved in the run manifest (`settings.diagnostics`).
- Ground truth can be attached per PDF in **New run** (saved before extraction starts) or later in
  **Results** (summary.json is then rewritten). The UI shows whether it is attached and whether
  CER/WER were computed.

## OCR/VLM Evaluation: full-PDF hosted run (page 7, default view)

Page 7 opens on **Full PDF run (hosted API)**:
Upload PDF -> Select Model -> API Key -> Authorize -> Run Full PDF -> Download ZIP.
The previous page (single page, local models, manual review) is unchanged under
**View: Advanced**. Nothing runs when the page opens or a setting changes.

- Models: Qwen3-VL-4B-Instruct via Oxen.ai and PaddleOCR-VL 1.6 via Baidu AI Studio,
  with their registry settings and prompt unchanged (`config/providers_e2.toml`).
- Authorize (all existing gates, re-checked by the runner before every page):
  (a) a new PDF is registered from the UI as a *user-owned test document* (write-once
  record under `results/api_document_registry/<sha256>/`; no TOML edit), then
  authorized by exact SHA-256 under the user-owned declaration; (b) the provider is
  enabled with page limits that cover the PDF; (c) "Prepare the page images" renders
  every page locally and shows exactly what is sent and where; (d) ONE approval covers
  every listed page image (SHA-256 each) for that run and expires after 24 hours.
- Pages are sent one at a time. A page whose submission is unconfirmed is never resent
  automatically; the run stops after 2 consecutive failed/unconfirmed pages. Retrying a
  page is an explicit action (Advanced Details). Baidu result hosts are approved per job,
  or automatically for the listed Baidu hosts if you tick that option for the run.
- Each run has its own write-once folder
  `results/<qwen3_vl|paddleocr_vl>/<PDF name>/run_YYYYMMDD_NNN/` with `metadata.json`,
  `plan.json`, `approval.json`, `attempts/`, `page_NNN.txt` (exact model text),
  `page_NNN.json`, and, when finalized, `full_text.txt`, `report.html` and
  `results.json`. The per-page audit trail (`api_jobs/`, `runs/`, `pages/`) is as before.
- The API key stays in the browser session's memory only. Ground Truth and the PDF text
  layer are never sent, and no accuracy is computed: compare the saved text manually.

## Hybrid PDF Extraction (page 10)

Upload PDF -> Select Fallback Model -> API Key / Local Connection -> Analyze PDF -> Review Plan ->
Approve & Run -> Download ZIP. Code: `lab/core/hybrid/` and `lab/app/pages/10_Hybrid_PDF_Extraction.py`.

- **Analyze PDF** is local only: PyMuPDF raw, the processed method and `find_tables()` from the Local PDF
  Extraction Benchmark (unchanged). Each page gets A (library text usable), B (Arabic words split by
  spaces) or C (no/garbled/unreliable text layer), with the reason. Raster images are flagged.
- **Fallback models** (your choice, never switched automatically): Qari-OCR 0.4.0 on your own
  OpenAI-compatible server (local GPU or RunPod; DPI 250, max_tokens 4096, temperature 0, "Free OCR.";
  "Check connection" only lists models), PaddleOCR-VL 1.6 (Baidu) and Qwen3-VL (Oxen) through the
  existing full-PDF machinery restricted to the selected pages, or **Library Only**.
- Only B/C pages are proposed (image pages only if you tick it). One approval covers the listed page
  images; all existing gates (SHA-256 authorization, enablement and limits, no automatic resend) apply.
- B pages: `library_rejoined` may only delete spaces between Arabic fragments, when the joined word is in
  the VLM text of that page; the non-space characters are verified identical. C pages use the VLM output
  as the candidate (never marked Accepted). Numbers are audited (occurrence-counted, never replaced).
- Each run: `results/hybrid_pdf_extraction/run_*/` (write-once). Ground truth (optional) is used only
  after processing for CER/WER of raw, processed, VLM raw, library_rejoined and the final output.

### Page documents (layout-aware Markdown, Phase 1)

`lab/core/hybrid/layout.py` assembles one Markdown document per page (`page_NNN.md`) and `full_document.md`
from saved data only: accepted decisions, `tables.json`, saved model text and the run's PDF copy
(`source.pdf`, same SHA-256). Text blocks, Markdown tables and "🖼️ Image (not read)" / "📊 Figure (not read)"
placeholders are placed by a recursive XY-cut (right-to-left columns on Arabic pages). Characters of all
text elements must equal the accepted page text as a multiset (whole elements may move); otherwise the page
text is placed as one region and the reason recorded. Layout revisions are append-only under `layout/`
(`revision_NNNNNN.json` + `rev_NNNNNN/`), also for finalized runs. Phase 2 (reading figures with the
fallback model) plugs in through `assemble_page(..., transcriptions=...)`.

Layout version 2 (2026-11-19): paragraph merging (same font size/weight, line pitch <= 1.5 x line height,
aligned, no first-line indent, previous line reaching the column edge, nothing in between; only whitespace
changes) with Markdown list items; frame/shadow raster deduplication (containment >= 95%, inner covers >= 50%,
outer pixel density <= 50% of the inner one; other overlaps are kept and flagged); image skip rule "area < 2%
OR longest side < 100 pt" with borderline images flagged; table gridlines excluded from vector charts by
coordinates. Each page reports text extraction status and layout status separately.

Layout version 3 (2026-11-19): vertically overlapping side-by-side elements separated by a narrow gutter
are ordered by direction (right-to-left on Arabic pages; unclear cases are flagged); bullet-prefixed text is
never a heading, formatting-only bullets are recorded and not rendered, trailing bullets on RTL lines are list
markers; a table continued from the previous page reuses that page's header only when column count and
positions match (otherwise a blank header row; no data row is ever promoted to a header). Structural checks
(side-by-side order, bullet headings, empty list items, table header columns) are recorded per page.

### Phase 2 · Image regions (Benchmark Mode, page 10)

`lab/core/hybrid/enrich.py` + `lab/app/hybrid_phase2_panel.py`. Regions come from the approved layout revision
and are rendered once at 250 DPI into a write-once region set (`<run>/enrichment/set_NNNNNN/`); every provider
receives exactly those crop bytes. Hosted providers (Baidu, Oxen) go through `APIEvaluationRunner` with a crop
preview (`image_transmission_preview`): same document authorization, enablement limits, claims, journals and
no-resubmission rules. Qari uses the existing endpoint client ("Free OCR.", unchanged settings) or the offline
export/import workflow (crops ZIP + manifest -> `{region_id}.txt` ZIP, verified, provenance `manual_import`).
Outputs are compared per region, the preferred one is chosen by the user (append-only selection), image Ground
Truth (AgenticDoc JSON `pages[].visual_elements`, `pages[].tables`; no coordinates, so matches are confirmed
manually) is used for evaluation only, and unified page documents are saved as append-only enrichment revisions
(`enrichment/documents/rev_NNNNNN/`). No request is sent by export, import, comparison, evaluation or recompute.

PaddleOCR-VL extraction modes (Phase 2, Benchmark Mode): "Standard document parsing" (registry defaults) or
"Chart recognition" (`optionalPayload` gets `useChartRecognition: true`, the only chart option documented for
the hosted AI Studio jobs API). OCR for image blocks is not documented for the hosted API and is not sent. The
registry entry is unchanged; the exact options are recorded in each run's plan.json, the job journal (provider
identity / transmission) and every exported result.

### Usage limits (hosted API providers)

Internal limits (not provider quotas) are configured per provider in `providers_e2.toml` under
`[providers.limits]`: `max_pages_per_enablement`, `max_pages_per_day_ceiling`, `max_pages_cumulative`,
`default_enablement_days` and optionally `provider_documented_daily_pages`. An enablement's total budget counts
submissions since that enablement was created; the daily limit counts the UTC day across enablements; the
cumulative cap counts the whole usage history, so disabling and re-enabling cannot raise the budget without
limit. Every submitted request counts (also failed or unconfirmed ones). The usage ledger and earlier
enablement records are never changed. All submission paths share `authorization.check_authorization`.

Phase 2 crop options (per region set, `lab/core/hybrid/crops.py`): resolution `auto` | `native` | 150 | 200 |
250 and format `auto` | `png` | `jpeg` (q90). `native` sends the page's single embedded JPEG unchanged when it
fully represents the page (one image covering the page within 2 pt, no text, drawings, annotations, rotation or
skew, no soft mask, CMYK, JPX or non-JPEG filter); sub-regions of such pages are cut from the embedded pixels.
Otherwise the crop is rendered and the fallback reason recorded. Request timeouts per call default to 60 s
(region < 25% of the page) or 180 s, can be overridden before sending, and are recorded with send/response
times. Crop options and timeouts are stored in regions.json, run plans, export manifests and every output.

Region sets from layout boxes (Phase 2, scanned pages; `lab/core/hybrid/layout_regions.py`): a pluggable layout
source returns per-page blocks `{page, label, bbox_px, image_px_size, region_bbox_pt, source_id, block_index}`.
The first source reads boxes from a SAVED PaddleOCR-VL output (a test of the cropping mechanism only; the target
source is local PP-DocLayout, placeholder `pp_doclayout_local`). Boxes are mapped px -> pt -> embedded-image px,
expanded by a margin (default 15 px, clamped) and cut from the scanned page's own pixels as lossless PNG. Only
exact duplicates are removed; small (< 100 pt), near-duplicate (IoU > 0.9) and overlapping boxes are kept but
never preselected. Region ids are `p<page>_L<n>`; provenance (`layout_box`) is stored in regions.json, the export
manifest and every output. The result is an ordinary, write-once region set.

### Local PP-DocLayout (Phase 2 layout source for scanned pages)

Paddle is used for layout detection only; Qari stays the OCR model. The app never downloads weights and never
contacts a hosted service: model load and inference run under the air-gap outbound network guard.

1. Install PaddlePaddle (CPU or GPU, not both) and `paddleocr` (see `lab/requirements-doclayout.txt`).
2. Download the model once, explicitly, e.g.
   `https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout_plus-L_infer.tar`
   and unpack it to a local folder (it must contain `inference.yml`, `inference.pdiparams` and `inference.json`).
3. Phase 2 -> 1b -> "Local PP-DocLayout": choose the same model name, enter the folder, "Check local model setup"
   (reads the files only; shows versions and the model's label list), then "Run local layout detection".

Detection runs on each scanned page's embedded image at native resolution (no unwarping, no orientation
correction, no resize by the app); boxes outside the image are flagged. Figure titles directly above a chart,
table or image (gap <= 30 pt, overlap >= 50% of the title width, unambiguous) are added to its crop and recorded;
captions below are optional (off by default). The detection is saved as `layout_detection.json` in the new set.

Layout-box selection rules (2026-11-19): a box >= 90% inside another is *contained*, not a partial overlap. The
complete outer visual is preferred (inner boxes stay available, not preselected) unless there are >= 2 disjoint,
non-small inner boxes each covering >= 15% of the outer box and >= 50% together; then the inner visuals are
selected and the outer box is marked *container*. Genuine partial overlaps stay uncertain. A `figure_title` is
attached when it sits up to 30 pt above a visual or overlaps its top boundary by up to 30 pt (>= 50% horizontal
overlap); text nearer to the visual above is that visual's caption and is never attached to the next one;
equally close text is left for review. A region whose border cuts another block (10-90% inside) is flagged; a
larger box is proposed but applied only when chosen, and the detected box is always recorded.
