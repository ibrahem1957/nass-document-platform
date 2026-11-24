# Nass: PDF extraction (`nass/`, Python package `lab`)

Nass turns one PDF into a write-once **run folder**, whose central file is `document.json`. It
extracts and records evidence; it never "fixes" text. Everything below is taken from the docstrings
and code in `nass/lab/core/auto/` (Automatic Extraction Mode, the code path Studio uses).

The `lab` package also contains the earlier **Extraction Lab** (Streamlit pages, E1/E2 evaluation of
other extractors and hosted OCR APIs, air-gap tooling). That code is kept unchanged, and its tests run
in the Nass suite. It is **not** part of the Studio workflow: see `nass/lab/README.md` (written for its
original location; paths there refer to the old machine layout).

## Supported PDFs and routing (`router.py`)

Every page is routed independently. The first matching rule wins, and the rule and its evidence are
recorded in `plan/routing.json`:

| Rule | Condition | Route |
|---|---|---|
| S1 | page-sized image (≥ 85 % of the page) and no text layer | SCANNED |
| S2 | page-sized image and an unreliable text layer (category B or C) | SCANNED |
| S3 | page-sized image and an invisible OCR text layer (render mode 3) | SCANNED |
| S4 | no page-sized image, text layer missing or garbled (category C) | SCANNED (rendered) |
| M1 | page-sized image under a reliable visible text layer | MIXED |
| M2 | reliable text layer and raster images covering ≥ 5 % | MIXED |
| D1 | reliable text layer, no large raster image | DIGITAL |

A document can therefore mix DIGITAL, SCANNED and MIXED pages. Arabic, English and mixed-script
pages use the same routing.

## Per-route extraction

**DIGITAL and MIXED pages**
- Text comes from the native PDF text layer (PyMuPDF), not from OCR.
- Arabic text-layer issues (words split by spaces, category B) keep the native text, and use a
  full-page Qari transcription only as spacing guidance (the `library_rejoined` step).
- Visuals are cropped for Qari (`visual_crops.py`). Native embedded image bytes are used only when
  seven safety checks pass. Otherwise the visible region is rendered at 250 DPI.
- Tables found by PyMuPDF `find_tables` get row/column **cells** in `document.json`.

**SCANNED pages, and embedded scans on MIXED pages**
- Layout detection runs locally with **PP-DocLayout_plus-L** (`regions.py`) on the page image's own
  pixels: no deskew, no resize.
- Region selection follows the deterministic rules in `lab/core/hybrid/layout_regions.py`:
  containment, title attachment, a 30 pt title gap, a 5 pt decision margin, and a 2× rule for a title
  between two visuals.
- Each region gets a decision: `auto`, `auto_flagged`, or `review`. A `review` region is processed only
  after a reviewer clicks "Process now" in the Nass lab. Studio does not offer that action.
- **Crops** (`images.py`) are cut from the native pixel grid with the proven **60 px margin** (clamped
  to the page) and saved as lossless PNG. The margin can catch neighbouring text; see "Margin leakage"
  below.
- **Full-page OCR of the body text** uses a **masked** copy of the page (`masking.py`): every
  automatically processed visual is painted white so its content comes only from its crop. The
  original page image is never changed.
- **Reading order** (`order.py`): Qari returns one text per page with **no paragraph coordinates**.
  Nass places that body text among the visuals deterministically, using the layout text blocks. The
  method is recorded per page:
  - `page_text_first`
  - `single_text_segment` (exact)
  - `paragraphs_aligned_to_text_blocks`
  - `lines_aligned_to_text_blocks`
  - `proportional_to_text_block_area`: order-preserving, but a unit near a boundary may land one
    segment early or late. This is flagged as approximate.

## Qari OCR (`qari_client.py`, `nass/runpod/qari_service.py`)

- **Model:** Qari is the only OCR model: `Qwen/Qwen3-VL-4B-Instruct` with the PEFT adapter
  `NAMAA-Space/Qari-OCR-0.4.0-VL-4B-Instruct`.
- **Fixed recipe** (`PROVEN` in `qari_service.py`):
  - prompt "Free OCR.", greedy decoding
  - `max_new_tokens` 1536
  - time limits of 30 s (crop) and 60 s (page)
  - a loop stopper, and one retry with `repetition_penalty` 1.1
- **Client:** Nass sends `POST /v1/ocr` with `{request_id, kind: crop|page, image_png_b64,
  image_sha256}` and a Bearer token. The service returns every attempt (text, tokens, timing,
  `loop_stopped`, `hit_limit`, `timed_out`).
- **Request records:** every request and outcome is saved under `raw/<job_id>/attempt_NN.*.json`. A
  request that may have reached the service without a response is never resent automatically;
  `GET /v1/results/{request_id}` recovers it.
- **Digit second read:** an optional, report-only second read of crops with Arabic-Indic digits
  (`VERIFY_DIGITS_*`). It is recorded, never applied.
- **Offline path:** `pipeline.export_offline` / `import_offline` plus
  `nass/runpod/qari_offline_batch.py` let the same jobs run on a GPU host without the HTTP service.
  The Nass lab exposes this; **Studio does not**.

Setup and operations are in `docs/RUNPOD_SETUP.md`.

## Caption and title association (`lab/core/hybrid/layout_regions.py`)

A `figure_title` box is **attached** to a visual (and included in its crop) only when the geometry is
unambiguous:
- it sits up to 30 pt above the visual, or overlaps its top boundary;
- at least 50 % of the title's width overlaps the visual;
- it is the visual's only title.

Other rules:
- **Captions below visuals:** not attached by default (`captions_below = false`).
- **A title between two visuals:** attached to the lower visual only if the upper one is at least
  twice as far away. Otherwise it is recorded as `ambiguous` and both visuals are flagged.
- **Recorded evidence:** each title block records `title_association`: status, measured candidates,
  reason codes, policy and limits. Each visual records `title_candidates`. This is geometry only: the
  title's text is never OCR-verified at this stage.

## De-duplication and margin leakage (`dedup.py`)

Nothing is corrected, and every removed span is recorded.

1. **Margin lines:** a crop line is removed only if it matches a text block outside the visual whose
   text is independently known. On scanned pages that text is not known, so nothing is removed and the
   reason is recorded.
2. **Figure text in the page OCR:** removed only for runs of at least 3 tokens matching a crop of the
   same page. Ambiguous mixed lines are kept and recorded.
3. **Suspected margin leakage (diagnostic only):** title or text blocks outside the visual's detected
   box but inside its margin-expanded crop are recorded in `dedup.suspected_margin_leakage`. Caption
   labels found in the crop text are listed **unattributed**, because the OCR can reorder text. This
   is never proof that the crop's OCR contains the neighbouring text, and the text is never changed.

Other related modules:
- `duplicates.py`: report-only candidate pairs between page OCR and crop OCR.
- `redundancy.py`: hides a pending placeholder only when coverage is proven.

## Validation (`validate.py`, `digits.py`)

Validation produces flags only:
- **Per OCR job:** empty output, loop stopped, timeout, token limit, retried, service error,
  unconfirmed.
- **Per region:** incomplete-region suspicion (the region border cuts another block; a larger box is
  proposed, never applied), pending review, removed duplicate.
- **Per text:** number formats and units, digit checks, layout review flags.

Results go to `validation_report.json` and to the per-element `validation_state` values:
- `ok`
- `needs_review`: processed but flagged
- `pending_review`: not processed
- `included_in_composite`

## Run folder (output contract)

```text
<results>/automatic_extraction/run_YYYYMMDD_HHMMSS_xxxxxx/   (unique; every file write-once)
  source.pdf                  exact copy of the input PDF (same SHA-256)
  plan/run.json               document, configuration snapshot, proven-setting differences
  plan/analysis.json          text-layer library pass
  plan/routing.json           DIGITAL / SCANNED / MIXED per page, with rule and evidence
  plan/layout.json            PP-DocLayout boxes, region decisions, title associations, review queue
  plan/jobs.json              OCR jobs (crop/page) with image SHA-256
  crops/<job_id>.png          every image sent to Qari (page/body jobs: the MASKED copy)
  pages/<job_id>_original.png unmasked page image of each page/body job
  raw/<job_id>/attempt_NN.intent.json / .result.json   every request and its full outcome (raw OCR)
  service/service_*.json      endpoint, deployment mode, trust basis per execution
  document.json               final elements (see below)
  document.md                 readable view
  validation_report.json      flags
  rag_chunks.jsonl            Nass's own simple chunks (NOT the RAG input; see docs/RAG_HANDOFF.md)
  execution_manifest.json     written last: its presence means the run is finalized
```

The fields of each `document.json` element, as observed in saved runs:
- `id` (`p<page>_e<n>`), `page`, `reading_order`, `type` (`text`, `page_text`, `figure`, `table`, …)
- `content`: verbatim; **raw OCR is preserved, never rewritten**
- `bbox_pt` (a scanned `page_text` has the full-page box: a container, not a sentence position),
  `crop_bbox_pt`, `crop` (`pixel_box`, `margin_px`, image hash)
- `source` (`qari_crop`, `qari_full_page`, `pymupdf_*`, …), `model`
- `validation_state`, `reasons`, `decision`
- `cells` / `cells_note`: cells only when the source gives real structure (native text layer or an OCR
  Markdown/HTML table; a flattened scanned table has `cells: null`)
- `provenance`: job ID, attempt, request ID, layout label, title attachments and candidates,
  decisions
- `dedup`: removed spans, the margin-line rule, and `suspected_margin_leakage`

## Cases that require manual review

Each of these is flagged; none is silently resolved:
- `needs_review` and `pending_review` elements
- `review` regions
- approximate reading order
- ambiguous titles
- incomplete regions
- OCR loop stops, timeouts and token limits
- digit-check reports
- possible margin leakage
- flattened scanned tables, which have no cell structure

## What is not structurally verified

- **Scanned tables:** OCR returns flattened text. Row/column associations are not established, and the
  text order can mislead: a row label may follow its values.
- **Chart values:** not associated with their series or labels.
- **Scanned reading order:** approximate whenever `proportional_to_text_block_area` is used.
- **Caption-to-visual links on scanned pages:** inferred from geometry. The caption text has no
  coordinates.
