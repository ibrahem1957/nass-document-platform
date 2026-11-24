# Component 4: Unified Prepared Document Builder

Component 4 creates one self-contained, versioned document representation from matching saved outputs of Nass,
Document Structure, Content Relationships, and Document Quality. It is an integration boundary for a future,
independent consumer. It does not correct text, merge content, create chunks, embed, retrieve, rerank, or infer.

Machine-readable contract: [`schemas/prepared_document.schema.json`](../schemas/prepared_document.schema.json).

## CLI and write-once output

```powershell
python -m docprep.prepare <run_dir> --structure <structure_dir> --relations <relationships_dir> --quality <quality_dir>
```

Default output root: `D:\agenticdoc_e2\results\prepared_documents`.

Every invocation creates `<run_id>__<UTC timestamp>` through a staging directory followed by an atomic rename.
An existing target is never overwritten. Output files are marked read-only by default:

- `prepared_document.json`
- `prepared_document_manifest.json`
- `prepared_document_validation.json`
- `prepared_document_report.md`
- `prepared_document_report.html`

Exit codes are `0` for valid output, `2` for input/compatibility errors, `3` for an existing output path, and `4`
when the output is written but integrity validation reports errors.

## Actual upstream contracts

Component 4 reads the contracts implemented by the current project; it does not rename upstream fields:

| Layer | Required artifacts | Required contract |
| --- | --- | --- |
| Nass | `document.json` | schema 1, `automatic_extraction_document`; `pages`, `elements` |
| Structure | `section_tree.json`, `assignments.json`, `manifest.json` | schema 1; section hierarchy, source inventory, assignments, derived spans |
| Relationships | `relationships.json`, `visuals.json`, `lists.json`, `manifest.json` | schema 1; nodes, relationships, visual inventory, list views |
| Quality | `quality_findings.json`, `quality_summary.json`, `quality_inventory.json`, `quality_manifest.json` | schema 1; findings, scorecards, inventory, skipped checks |

Every artifact must carry the same source run ID. `kind` and `schema_version` are checked explicitly before any
assembly. When an upstream manifest records its source/Structure/Relationships directory, that lineage must match
the directories supplied to the CLI. A missing optional element field such as `model` or `provenance` becomes
absent metadata, but a missing required array or an incompatible contract fails clearly.

## Prepared document contract 1.1.0

The top-level `schema_version` is semantic and currently `1.1.0`. Version 1.1 is an additive extension of 1.0.
The top-level fields are:

| Field | Meaning |
| --- | --- |
| `document_id` / `source_run_id` | Stable document and source-run identities |
| `source_document` | Verbatim saved document metadata; no PDF bytes |
| `language` / `metadata` | Only supported recorded/discovered metadata; unknown values are null |
| `pages` | Saved Nass page records, including route, direction, and element order |
| `sections` | Ordered, non-flattened hierarchy plus element/unit navigation and assignment confidence |
| `elements` | Every original Nass element exactly once, in document order, with verbatim text |
| `derived_spans` | Structure and relationship spans with exact parent IDs and offsets |
| `lists` | Original structure, relationship view, and a convenient nested representation |
| `visuals` | Every figure/table, including unlinked visuals, original content and annotations |
| `logical_tables` | One logical node per multi-page table with ordered original parts and evidence |
| `nodes` / `relationships` | Exact Component 2 graph and all supported relationship evidence |
| `evidence_audit` | Accepted/rejected relationship candidates and explicit quality-check decisions (`checked`, `skipped` with reason, or `flagged`) with deterministic evidence |
| `quality` | Exact findings, inventory, summary and Component 3 validation |
| `indexes` | Deterministic data indexes only; no search or retrieval behavior |
| `provenance` | Input paths/hashes, upstream manifests, component/schema versions and configuration hash |

### Source preservation

Each `elements[]` item contains the stable ID, original type, exact `text` (including Arabic and English), page,
reading order, document order, saved bounding box, geometry reliability, Nass validation state, extraction
metadata, all Structure assignments, section IDs, roles, relationship IDs, quality finding IDs, and the Quality
inventory record. `needs_review`, `pending_review`, and `included_in_composite` are never suppressed.

The saved bounding box is provenance, not a claim of paragraph precision. `geometry_reliability` preserves the
Quality layer classification. Source text is never normalized; numeric normalization may occur only inside
copied evidence annotations.

### Sections and lists

Sections remain in upstream order with `parent` and `children`; no new headings are generated. Added navigation
fields list the section's unit and element IDs in reading order and retain per-unit assignment confidence.

Lists retain the original Structure record and Component 2 relationship view. `nested_items` is derived only from
saved item IDs and `parent_id`; it does not replace original elements. Derived spans retain the original parent,
start/end offsets, marker data, confidence and exact substring.

### Visuals and logical tables

Visual records preserve Component 2 captions, internal titles, describing paragraphs, numeric evidence, leaked
text evidence, composite membership, logical-table membership, validation states and review flags. Component 4
adds only navigation references and the original element content. Unlinked visuals remain present.

Explicit upstream skip/drop records appear as `known_missing` visual nodes. They retain the complete source record,
page and coordinates when available (`null` otherwise), and carry `known_missing` RAG flags. No gap-based missing
content inference is permitted.

Every element and visual exposes `rag_flags`, `evidence_reliability`, and
`evidence_reliability_reasons`. The default reliability is `unassessed`; absence of findings never promotes an item
to `high`. Every downgrade retains a non-empty reason. These fields describe source/content evidence only:
relationship-quality and missing-description findings remain linked through `quality_finding_ids` without being
reclassified as corrupted content. These annotations communicate evidence concerns only and do not implement retrieval
policy.

Logical tables come only from saved `logical_table` nodes. Their source parts remain separate original elements.
The unified node records ordered parts, page sequence, repeated-header and column compatibility evidence,
relationships, captions and quality references. No rows are merged, deleted, corrected or deduplicated.

### Relationships and uncertainty

All Component 2 relationship objects are copied without reinterpretation. Supported types include
`caption_of`, `title_inside_visual`, `titled_by_section`, `leaked_text`, `describes`, `table_continuation`,
`composite_part`, `aside_of`, `list_nesting`, `merged_list_items`, `introduces_list`, `document_metadata`, and
`known_missing`, plus the weak `co_located_in_section` relation. Co-location is always low confidence and never
means that a paragraph explains a visual. Confidence, evidence, conflicts, limitations, offsets, methods, pages and validation states stay
attached. A medium/low or review-recommended relationship remains uncertain.

A `caption_of` may start at an optional `caption_span` node: a verbatim caption-shaped span of full-page OCR (text
offsets known, page coordinates unknown). It is accepted only when the caption pattern, agreement with exactly one
visual's own OCR, and an unambiguous saved `figure_title` region next to that visual converge; it is capped at medium
confidence, recommends review, and records `span_geometry_used: false`. Caption spans that fail any check are kept as
`unresolved_caption_candidate` findings and `evidence_audit` entries with their `failed_checks`. A full-page OCR box
(`page_text` / `qari_full_page`) is a page container and is never used as caption or aside position. Runs made by Nass
versions that record `title_association` on `figure_title` blocks (and `title_candidates` on visuals) supply that
geometry directly; older runs are measured from their saved boxes with the same limits.

Retrieval consumers must follow [RAG_CONSUMER_CONTRACT.md](RAG_CONSUMER_CONTRACT.md): the meaning of
`unassessed`, `no_recorded_concern`, `use_with_flags`, `needs_review` and `pending_review`; the
`table_cells_unknown` / `possible_incomplete_content` / `possible_foreign_text` flags; table `table_structure` and
`cell_level_evidence_allowed`; and `ambiguous_number` tokens. Every prepared document is validated against the
full JSON Schema (`json_schema_valid` check).

A scanned visual may carry an optional `margin_leakage_risk` (finding `possible_caption_margin_leakage`): layout
title/text blocks outside the visual's detected box but inside its margin-expanded crop, from Nass
`dedup.suspected_margin_leakage` or, for older runs, measured from the saved layout pixels. Records are never
confirmed: caption labels found in the crop text are listed unattributed (OCR can reorder text). The visual text is
unchanged; a title attachment contested by another visual is not used as independent title evidence.

`co_located_in_section` is expected to be low confidence and never independently creates an actionable relationship
finding or content-reliability downgrade. A genuinely uncertain `describes`, caption, or other semantic link remains a
reviewable relationship finding, but its uncertainty does not by itself add content `needs_review`. Missing caption or
description remains an informational completeness finding.

All Component 3 findings are copied without promotion. Status distinguishes open, needs-review, unverified, and
not-applicable findings. Inventory skipped checks and clean coverage stay separate. No overall extraction
accuracy percentage is created.

### Metadata and nulls

`null` means no supported upstream value was available. The title may use the saved Structure document-title
section and records that section ID. Author is never inferred from a generic sentence; saved document-metadata
annotations remain available verbatim. Language is set only when the source document explicitly records it.

### Navigation indexes

`indexes` supports constant-time array navigation without implementing search:

- `element_by_id`, `section_by_id`, `list_by_id`, `logical_table_by_id` map IDs to array indexes;
- `visual_by_id` maps every extracted or known-missing visual ID to its array index;
- `known_missing_visuals` lists array indexes for explicit missing-content placeholders;
- `elements_by_section` maps a section to element indexes in reading order;
- `visuals_by_describing_source` maps a paragraph/node to described visual IDs;
- `quality_findings_by_element` maps original elements to finding IDs;
- `relationships_by_node` maps any graph node to all relationship IDs involving it.

## Validation

`prepared_document_validation.json` records explicit checks for exact source inventory, verbatim strings,
validation states, section and relationship references, quality references, span offsets, page ranges, acyclic
section/list hierarchies, logical table ordering, namespace uniqueness, unknown metadata, and unchanged hashes.
Broken references are retained and reported; they are never dropped to obtain a passing result.

Input SHA-256 hashes cover every available required artifact in all four layers before and after integration.
The manifest records both sets, output hashes, the configuration hash, required artifacts, and validation summary.

## Version compatibility

- Prepared schema `1.x`: additive optional fields may be introduced without changing existing required meaning.
- A change to required semantics, ID conventions, or source preservation increments the major version.
- Component 4 currently accepts only upstream `schema_version: 1` with the exact expected `kind` values.
- `docprep_version` is recorded for traceability; schema compatibility is controlled by artifact schema and kind.
- Consumers must reject unsupported prepared major versions rather than silently ignoring required semantics.

## Tests written, not executed

Synthetic tests are in `tests/docprep/test_prepare.py`. They cover English and Arabic preservation, inventory,
hierarchy, order, nested and merged lists, spans, visuals, numeric evidence, multi-page tables, quality attachment,
validation states, unknown metadata, compatibility errors, broken references, immutability, atomic write-once
output, schema declarations, escaping, indexes and deterministic logical content.

Per the implementation restriction these tests were **NOT EXECUTED**. Future commands:

```powershell
python -m pytest tests/docprep/test_prepare.py -q
python -m pytest tests/docprep -q
```

## Exact future saved-run commands

Traffic report:

```powershell
python -m docprep.prepare "D:\agenticdoc_e2\results\automatic_extraction\run_20261119_205601_e079cc" --structure "D:\agenticdoc_e2\results\document_structure\run_20261119_205601_e079cc__20261123T132328Z" --relations "D:\agenticdoc_e2\results\document_relationships\run_20261119_205601_e079cc__20261123T141226Z" --quality "D:\agenticdoc_e2\results\document_quality\run_20261119_205601_e079cc__20261123T151146Z"
```

Arabic scanned:

```powershell
python -m docprep.prepare "D:\agenticdoc_e2\results\automatic_extraction\run_20261119_212013_46ebeb" --structure "D:\agenticdoc_e2\results\document_structure\run_20261119_212013_46ebeb__20261123T132330Z" --relations "D:\agenticdoc_e2\results\document_relationships\run_20261119_212013_46ebeb__20261123T141227Z" --quality "D:\agenticdoc_e2\results\document_quality\run_20261119_212013_46ebeb__20261123T151146Z"
```

Arabic digital/mixed:

```powershell
python -m docprep.prepare "D:\agenticdoc_e2\results\automatic_extraction\run_20261119_210113_2d84c4" --structure "D:\agenticdoc_e2\results\document_structure\run_20261119_210113_2d84c4__20261123T130731Z" --relations "D:\agenticdoc_e2\results\document_relationships\run_20261119_210113_2d84c4__20261123T141227Z" --quality "D:\agenticdoc_e2\results\document_quality\run_20261119_210113_2d84c4__20261123T151146Z"
```

## Limitations

- Component 4 can preserve only evidence present in upstream artifacts; it does not reopen the PDF.
- Unknown language/author values remain null even when a human might infer them from text.
- The nested list view follows saved parent IDs and does not resolve upstream ambiguity.
- Logical-table evidence does not prove cell correctness.
- Quality findings and uncertain relationships remain review annotations, not established facts.
- The output is consumer-ready structured data, not RAG chunks or a retrieval index.
