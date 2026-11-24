# CLAUDE.md: instructions for AI coding agents

This is the single source of agent instructions for this repository. `AGENTS.md` points here; do not
create diverging copies.

## Project identity

**nass-document-platform** is a **document extraction and preparation platform**. It turns PDFs
(digital, scanned or mixed; Arabic, English or both) into auditable evidence:

- **Nass `document.json`:** extracted observations.
- **DocPrep `prepared_document.json`:** organized evidence with relationships, quality findings and a
  retrieval-safety contract.

**RAG is a future extension** (`rag/` is empty). It is not the current source of truth, and nothing
in the existing pipeline depends on it.

## Component responsibilities

| Folder | Owns | Must not |
|---|---|---|
| `studio/` | User interface and orchestration. Upload, run Nass then DocPrep, progress, views, downloads, cancel and retry. **The only application UI.** | Contain extraction, OCR, layout, relationship or quality logic. Be replaced by another UI. |
| `nass/` (package `lab`) | PDF extraction and **original OCR evidence**: routing, text layers, PP-DocLayout, crops (60 px margin), Qari requests, raw attempt records, assembly, dedup, validation flags | Correct, rewrite or "improve" OCR text |
| `nass/runpod/` | GPU-side Qari service code | Be started or called without explicit owner authorization |
| `docprep/` | Structure, relationships, quality metadata, prepared evidence, schema validation | Invent source text, numbers, cells, coordinates or relationships. Import Nass code. |
| `rag/` (future) | Indexing, retrieval, answer generation, **answer-time evidence enforcement** | Bypass DocPrep's safety metadata. Answer without source evidence. |

## Source of truth

1. **The original PDF** is the ultimate source.
2. **Nass outputs** are extracted observations, not guaranteed facts.
3. **DocPrep** adds structure and confidence assessments. It never invents source evidence.
4. **RAG** must cite source elements and pages, and must not invent missing information.

## Non-negotiable engineering rules

- **Read before editing.** Read the relevant implementation and contracts first:
  `docs/ARCHITECTURE.md`, `docs/PROCESSING_FLOW.md`, `docprep/docs/RAG_CONSUMER_CONTRACT.md`, and
  `docprep/schemas/prepared_document.schema.json`.
- **Never assume extracted text is reliable.** Never treat the absence of a warning as proof of
  correctness. `unassessed` and `no_recorded_concern` do **not** mean "verified".
- **Provenance:**
  - Never discard source provenance: element IDs, pages, offsets, hashes, raw attempt records.
  - Never silently rewrite raw OCR. Source text stays byte-identical through every stage.
- **Never fabricate** numbers, table cell associations, source coordinates or captions.
  - A full-page OCR bounding box is a page container, not a sentence position.
  - `ambiguous_number` tokens (e.g. `٦,٤`) have no value.
- **Relationships:** never convert uncertain relations (unresolved caption candidates, ambiguous
  titles, `co_located_in_section`) into confirmed relationships without new independent evidence.
- **Answers:** never invent answers for missing evidence. Never bypass `rag_flags`,
  `retrieval_readiness`, `evidence_reliability`, `table_structure` or `cell_level_evidence_allowed`.
- **No benchmark tuning:** no hardcoded fixes for specific documents, pages, figure numbers or
  benchmark questions. Fixes must be general.
- **Tests before success:** run the relevant regression tests before reporting success (see
  `docs/TESTING.md`). Report failures and skips exactly. The 19 known Nass failures are listed there;
  any other failure is a regression until proven otherwise.
- **Studio must keep working.** Keep the existing interface working, and preserve backwards
  compatibility of `document.json` and `prepared_document.json` where practical. Schema 1.1.0 allows
  only additive fields.
- **Paid and external resources:** never start RunPod, call the Qari service, run OCR, download models,
  or use paid APIs without the owner's explicit authorization.
- **Secrets:** never commit secrets (`nass/.env`, tokens, pod URLs), private PDFs, runtime outputs, or
  model weights. Check `git status` before committing.
- **Proven settings:** keep the proven extraction settings (Qari recipe, 60 px crop margin, title rules,
  `captions_below=false`) unless the owner approves a measured change.

## Layout and commands

```text
studio/   python -m studio.server --config config/studio.json        (run from studio/)
nass/     python -m pytest tests/lab -q                                (run from nass/)
docprep/  python -m pytest tests/docprep -q                            (run from docprep/)
          python -m docprep.{structure,relations,quality,prepare} ...  (always pass --out)
scripts/  python scripts/check_setup.py ; .\scripts\run_tests.ps1    (run from repo root)
```

- **Environment:** one root `.venv` (Python 3.12 x64).
- **Nass configuration:** `nass/lab/config/auto_extraction.toml`, then `nass/.env` (copied from
  `.env.example`), then the real environment.
- **Studio configuration:** `studio/config/studio.json` (relative paths; `"current"` interpreter).
- **Runtime data:** `studio/data/jobs/` (git-ignored).

## Starting checklist for the RAG work

Read in this order:
1. `README.md`
2. `docs/ARCHITECTURE.md`
3. `docs/PROCESSING_FLOW.md`
4. `docs/RAG_HANDOFF.md`
5. `docs/RAG_EVIDENCE_POLICY.md`
6. `docprep/docs/RAG_CONSUMER_CONTRACT.md` and `docprep/schemas/prepared_document.schema.json`
7. `docs/KNOWN_LIMITATIONS.md`

Then:
- **Implement the answer-time policy engine and its refusal tests before** integrating any LLM.
- **Integrate into Studio** behind a runtime check, so Upload, Extract and Prepare keep working without
  RAG.

## Critical facts that are easy to get wrong

- **Scanned tables:** they have `table_structure: "none"`. Their flattened OCR can put a row label
  **after** its values. Never answer cell-specific questions from them.
- **Margin leakage:** `possible_foreign_text` is a possible contamination, not a confirmed one. A
  caption inside crop OCR does not prove which visual it belongs to.
- **Scanned structure:** scanned documents often have one section and an approximate reading order. Do
  not chunk them by section.
- **Raw extraction files:** Nass `rag_chunks.jsonl` and the Markdown views are **not** RAG inputs.
- **Indexing gate:** Studio only snapshots prepared documents whose validation has zero errors. Index
  nothing else.
