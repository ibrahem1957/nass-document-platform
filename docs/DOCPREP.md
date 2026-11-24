# DocPrep: document understanding and preparation (`docprep/`, Python package `docprep`)

DocPrep reads one **finished Nass run folder** and writes `prepared_document.json`, a self-contained
document that keeps every source element verbatim. It adds hierarchy, relationships, quality findings
and retrieval-safety metadata.

- **Inputs:** never modified; they are hashed before and after each stage.
- **External calls:** none. No model, no network.
- **Determinism:** fully deterministic; the rules are lexical, typographic and geometric.

The detailed component documents, which are authoritative for their parts, are in `docprep/docs/`:

| Document | Covers |
|---|---|
| `docprep/README.md` | Component 1, the Structure Builder: headings, levels, assignments, scanned `page_text` segmentation |
| `docprep/docs/CONTENT_RELATIONSHIP_BUILDER.md` | Component 2: relationships, visuals, lists, logical tables |
| `docprep/docs/DOCUMENT_QUALITY_VALIDATION_LAYER.md` | Component 3: quality findings, reliability, inventory |
| `docprep/docs/PREPARED_DOCUMENT_SCHEMA.md` | Component 4: the unified `prepared_document.json` |
| `docprep/docs/RAG_CONSUMER_CONTRACT.md` | **The retrieval-safety contract**, the single source for the meaning of reliability, readiness, flags, table structure and ambiguous numbers |
| `docprep/schemas/prepared_document.schema.json` | JSON Schema 2020-12, version 1.1.0 |

## Stages (each a CLI module, run in this order by Studio)

```text
python -m docprep.structure <nass_run> --out <root>
python -m docprep.relations <nass_run> --structure <structure_out> --out <root>
python -m docprep.quality   <nass_run> --structure <..> --relations <..> --out <root>
python -m docprep.prepare   <nass_run> --structure <..> --relations <..> --quality <..> --out <root>
```

- **Output folders:** each stage writes a new folder `<root>/<run_id>__<UTC timestamp>/` and prints
  `written: <path>`. Studio parses that line.
- **Exit codes:**
  - `0` OK
  - `2` input or configuration error
  - `3` output folder already exists
  - `4` validation errors (outputs are still written, for review)
- **Default output roots:** without `--out`, the defaults come from `DOCPREP_OUTPUT_ROOT` (structure
  stage) or the original machine's `D:\agenticdoc_e2\results\…` paths. Pass `--out` when you run
  DocPrep yourself.

## Input contract (from Nass)

| File | Use |
|---|---|
| `document.json` | Elements, pages, types, boxes, reading order, validation states (required) |
| `validation_report.json` | Nass flags: reading order, incomplete regions, review |
| `execution_manifest.json` | Provenance hash |
| `plan/layout.json` | PP-DocLayout boxes, title associations, region decisions (scanned pages) |
| `source.pdf` | Typography of digital and mixed pages (PyMuPDF spans) |

## `prepared_document.json` (schema 1.1.0)

**Top-level keys:**
- `schema_version`, `kind`, `document_id`, `source_run_id`, `source_document` (name, SHA-256, bytes,
  pages), `language`, `page_count`, `metadata`
- `pages`, `sections`, `elements`, `derived_spans`, `lists`, `visuals`, `logical_tables`, `nodes`
- `relationships`, `evidence_audit`, `quality`, `indexes`, `provenance`

**Elements** (`elements[]`, one per Nass element, exactly once):
- **Identity and text:** `element_id` (the Nass ID, e.g. `p6_e1`), `original_type`, `text`
  (byte-identical to Nass `content`)
- **Position:** `page`, `reading_order`, `document_order`, `bbox_pt`, and `geometry_reliability`
  (`precise`, `approximate`, `page_level_only` or `unavailable`). A scanned `page_text` box is a page
  container, not a sentence position.
- **Source state:** `validation_state`, plus `source_extraction`, which holds every other Nass field:
  cells, crop, provenance, dedup.
- **Structure and links:** `structure_assignments`, `section_ids`, `roles`, `relationship_ids`,
  `quality_finding_ids`
- **Quality and safety:** `quality_inventory` (checks attempted or skipped, and the reliability
  dimensions including `retrieval_readiness`), `rag_flags`, `evidence_reliability`,
  `evidence_reliability_reasons`

**Derived spans:** scanned `page_text` is segmented into `derived_spans` (role hints: heading,
paragraph, caption, list…). Spans have text offsets and `geometry: null` unless a single layout block
contains them. A recorded container box is still not the span's position.

**Relationship types:**
- `caption_of`, `title_inside_visual`, `titled_by_section`, `describes`
- `table_continuation` (with a `logical_table` node), `composite_part`, `aside_of`
- `list_nesting`, `merged_list_items`, `introduces_list`
- `document_metadata`, `known_missing`
- `leaked_text`, and the weak `co_located_in_section` (proximity only, never semantic)

Each relationship carries `confidence`, `evidence`, `method`, `conflicts`, `limitations` and
`review_recommended`.

**Caption sources:** a `caption_of` may start at a `caption_span` node, a verbatim span of the page OCR
with no coordinates. It is accepted only when several independent signals converge, and is capped at
medium confidence. Candidates that fail are kept as `unresolved_caption_candidate` findings and
`evidence_audit` entries, never as relationships.

**Visuals** (`visuals[]`):
- **Titles and captions:** internal title, caption, description paragraphs
- **Composites:** composite parent and parts
- **Margin leakage:** `margin_leakage_risk`
- **Tables:**
  - `table_structure`: `source_cells` or `none`
  - `cell_level_evidence_allowed`
  - `table_consumer_rule`

## Retrieval-safety metadata (summary; the contract document is authoritative)

| Field | Answers the question | Values |
|---|---|---|
| `validation_state` | How did Nass handle the element? | `ok`, `needs_review`, `pending_review`, `included_in_composite` |
| `evidence_reliability` | Is a content-affecting finding recorded? | `unassessed` (none recorded; **not** "verified"), `medium`, `low` |
| `rag_flags` | Which specific content risks apply? | `table_cells_unknown`, `possible_incomplete_content`, `possible_foreign_text`, `ocr_noise`, `ocr_repetition_loop`, `bidi_order_suspect`, `chart_without_values`, `probable_missing_numeric_evidence`, `known_missing`, `needs_review`, … |
| `quality_inventory.reliability_dimensions.retrieval_readiness.state` | How may the element be used? | `no_recorded_concern`, `content_unflagged_placement_or_links_uncertain`, `content_concern_recorded`, `use_with_flags`, `review_before_use` |
| `visuals[].table_structure` / `cell_level_evidence_allowed` | May cell-specific values be answered from this table? | `source_cells` / `true`, or `none` / `false` |
| numeric tokens with dimension `ambiguous_number` | Is a number's value established? | `٦,٤`, `2,15` etc. claim no value |

These dimensions are separate:
- **Source extraction state:** `validation_state`
- **Content reliability:** `evidence_reliability`, `rag_flags`
- **Structural placement and relationship confidence:** reported per element in
  `reliability_dimensions`

**Absence of a recorded concern is never proof that a fact is correct.**

## Provenance and validation

- **Provenance:** `provenance` records the DocPrep version, configuration and its hash, input folders,
  and SHA-256 hashes of every input before and after.
- **Validation:** `prepared_document_validation.json` lists 19 checks, including:
  - `source_inventory_exact`
  - `source_text_verbatim`
  - ID and relationship integrity
  - `input_hashes_unchanged`
  - **`json_schema_valid`**: full JSON Schema validation; a missing validator or schema counts as an
    error
- **Indexing gate:** any error makes `docprep.prepare` exit with code 4. Such a document **must not be
  indexed** for retrieval.

## Other DocPrep tools

- **`docprep.evaluate`, `docprep.evaluate_relations`, `docprep.evaluate_quality`:** measure outputs
  against frozen reference files. The original v1 reference is private and is not distributed; see
  `docs/TESTING.md`.
- **`docprep.inspector`:** a local review UI. Copy `docprep/config/inspector_documents.example.json` to
  `inspector_documents.json` and set your own paths.
