# RAG handoff

The RAG component is **not implemented**. This document is the brief for the engineer who will build
it in `rag/` and integrate it into the existing Studio UI.

**Read first:**
1. `docs/ARCHITECTURE.md` and `docs/PROCESSING_FLOW.md`: what produces your input.
2. `docprep/docs/RAG_CONSUMER_CONTRACT.md` and `docprep/schemas/prepared_document.schema.json`: the
   input contract.
3. `docs/RAG_EVIDENCE_POLICY.md`: the rules you must enforce.
4. `docs/KNOWN_LIMITATIONS.md`: what the evidence cannot support.

## Target user journey (inside the same Studio application)

```text
Upload PDF → Extract (Nass) → Prepare (DocPrep) → Index → Ask Questions → View Answers and Sources
             └──────── existing, must keep working without RAG ───────┘ └──────── new ────────┘
```

**Constraints:**
- **Studio stays the only UI.** Do not create a separate chatbot application. Add the RAG features to
  the Studio front end: `studio/studio/static/index.html`, `app.js`, `app.css`, plus server routes in
  `studio/studio/server.py`. Keep Studio's existing security properties:
  - localhost-only binding;
  - `textContent` rendering of source text (no source HTML executed);
  - path-traversal protection;
  - no shell subprocesses.
- **RAG is optional.** Upload, extraction, preparation, viewing and downloads must keep working when the
  RAG package is not installed or is disabled. Detect it at runtime and hide the RAG controls.

## 1. Ingestion

- **Input:** a `prepared_document.json` whose validation summary has `error == 0`. In a Studio job,
  that is `studio/data/jobs/job_<id>/docprep/prepared_document.json`.
  - Studio writes this snapshot **only** when the validation summary reports zero errors
    (`jobs.py::_snapshot_docprep`).
  - `docprep/source_run_references.json` records the validation file's path and SHA-256.
- **Validate again at ingestion** with `jsonschema` against `docprep/schemas/prepared_document.schema.json`.
- **Units:**
  - `elements[]`
  - `derived_spans` for scanned `page_text`
  - `visuals[]` for visual metadata
  - `logical_tables` for multi-page tables
  - `relationships` for links
- **Stable identifiers:** `document_id`, `source_document.sha256`, `element_id`, `page`, `span_id` +
  offsets.
- **Source text:** never index the Markdown views, `rag_chunks.jsonl` from Nass, or raw OCR alone.

## 2. Chunking

- **Element-based:** chunk along element boundaries; use the hierarchy (`sections`) only where it is
  reliable. Digital documents usually have a real hierarchy; for example, the English traffic report
  has 30 sections.
- **Scanned documents:** frequently one section and approximate reading order. Chunk by element and
  derived span, and keep neighbouring text and visuals separate.
- **Keep source references** (element IDs, pages, offsets) and **all safety metadata** on every chunk
  (policy P0).
- **Do not mix** unrelated figures, tables and paragraphs in one chunk. A caption may be attached to its
  visual only through an accepted `caption_of` relationship.

## 3. Retrieval

- **Method:** semantic, keyword (BM25) or hybrid, chosen by measured results on Arabic, English and
  mixed documents. The embedding model is **not predetermined**.
- **Model sourcing:** prefer self-hostable or downloadable models for production. Hosted APIs are
  acceptable for evaluation only. This follows the project's existing model policy for Nass.
- **Arabic:** normalization for retrieval (alef variants, ta marbuta, diacritics, Arabic-Indic digits)
  must not change the stored source text that is quoted.
- **Access control:** preserve document-level access control if documents belong to different users.
  Studio is single-user today.
- **Relevance is not reliability.** Ranking must not hide flags.

## 4. Grounded answer generation

- Every substantive answer is supported by retrieved evidence and cites the **PDF, page and element**.
- Missing, conflicting, incomplete or uncertain evidence is stated explicitly. **Never fabricate.**
- Apply policies P1–P5. The critical ones:
  - **Flattened tables** (`cell_level_evidence_allowed=false`): no cell-specific answers.
  - **`ambiguous_number`:** quote verbatim and never convert; `٦,٤` is not `64`.
  - **Adjacent numbers and labels:** never treat them as one table row without verified structure.
  - **`possible_foreign_text`:** potential contamination only; a caption inside crop OCR is not proof
    of ownership.
  - **Unresolved caption candidates:** never used as relationships.

## 5. Studio integration to build later

- **Indexing status** per document, e.g. not indexed, indexing, indexed, or refused (validation
  errors).
- **Question input** (chat) scoped to one or more processed documents.
- **Answers with citations:** document, page and element. Navigate to the source page through the
  existing `/api/items/<source>/<id>/pdf` and reading views where feasible.
- **Clear messages** when evidence is insufficient.
- **Warnings** for ambiguous OCR numbers, flattened tables (unsupported numeric questions), flagged
  visuals and review-required content.

## 6. Suggested first implementation steps

1. **`rag/` package skeleton:** a loader that reads a prepared document, validates it, and refuses it
   on errors. Add tests (policy P6, items 1 and 6).
2. **Chunker** that preserves IDs, pages and safety metadata. Test it on the regression fixtures (see
   `docs/TESTING.md`).
3. **Answer-time policy engine (P1–P5)**, unit-tested **before** any LLM integration: the table, number
   and caption refusals.
4. **Retrieval baseline and evaluation set:** questions with known answers and page citations,
   including questions that must be refused.
5. **Studio integration** behind a feature check.

## What already exists to help you

- **Synthetic test helpers** for prepared documents: `docprep/tests/docprep/helpers.py`,
  `test_prepare.py`.
- **Safety-contract tests**, showing exactly which elements get which flags:
  `docprep/tests/docprep/test_rag_safety.py`.
- **Real benchmark outputs** exist on the original development machine but are **not** in this
  repository, because they derive from private documents. Ask the owner for access, or generate your
  own by processing non-sensitive PDFs through Studio.
