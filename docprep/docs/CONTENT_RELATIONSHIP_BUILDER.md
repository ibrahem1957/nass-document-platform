# Content Relationship Builder — Component 2

Status: implemented locally. Tests are written but **NOT EXECUTED**, as required by the implementation task.

## Architecture

- `relation_inputs.py` loads the saved Nass run and matching Document Structure output read-only, validates run identity,
  and hashes every available source artifact.
- `relation_model.py` defines derived nodes, relationships, and the component result contract.
- `relation_text.py` provides comparison-only Arabic/English normalization, caption parsing, numeric tokens with units,
  lexical comparison, and conservative list-marker parsing. Stored output always retains verbatim source text.
- `relation_builder.py` applies deterministic evidence rules and creates a combined node inventory.
- `relation_validation.py` checks reference integrity, offsets, caption cardinality, visual coverage, logical tables, list
  cycles, validation-state preservation, input hashes, and ambiguity flags.
- `relation_reports.py` renders local-only Markdown and self-contained HTML. All source-derived HTML is escaped.
- `relation_output.py` writes the six required artifacts atomically to a new timestamped directory.
- `relations.py` exposes the Component 2 CLI.
- `evaluate_relations.py` creates write-once CSV review templates and separately scores completed reviews.

## Implemented evidence rules

- `caption_of`: explicit Arabic/English patterns, saved Structure caption roles, saved PP-DocLayout `figure_title`
  geometry, page/order/section proximity, and target type. Near-tied targets remain unresolved or carry conflicts.
- `title_inside_visual`: a short, lexical, non-numeric first OCR line separated from later lines, or a conservative
  single-line OCR prefix independently supported by its enclosing heading, explicit caption, or verified title
  metadata. Exact offsets and the original substring are retained. `titled_by_section` requires at least two shared
  terms and a lexical threshold.
- `leaked_text`: exact or strong normalized overlap, with raw offsets only when directly supported. Short repetitions are
  ignored.
- `describes`: an explicit numbered/relative figure or table reference, or at least two independently compatible numeric
  matches. Topic words never create a relationship.
- `table_continuation`: consecutive pages plus saved continuation evidence and/or repeated headers, compatible column
  count, and section context. Source parts remain unchanged inside a derived `logical_table` node.
- `composite_part`: saved parent provenance or unique geometric containment only. Text similarity is never used.
- `aside_of`: shared vertical area, horizontal gap, page and section context. Missing geometry does not produce a link.
- `list_nesting`: saved list membership plus indentation and contextual marker changes. `introduces_list` retains saved
  intro links. Multiple source markers can create exact-offset `merged_list_items` spans without changing the parent.
- `document_metadata`: anchored author/preparer/reviewer labels plus footer or document-ending context.
- `known_missing`: explicit saved validation/execution evidence only. Its placeholder node has `text`, `source_id`,
  `start`, and `end` set to `null`; the complete skip record stays in `provenance`, with
  `missing_source_content` retained in `review_flags`. It never infers a visual, text, offset, or box.
- `co_located_in_section`: low-confidence structural proximity only. It never implies explanation or factual support.

## Numeric comparison

Arabic-Indic, Persian and Western digits are compared in a common numeric form. Arabic decimal `٫`, Arabic thousands
`٬`, Western separators, `%`, `٪`, `‰`, and unambiguous `K`/`M` suffixes are supported. Units are normalized only for
comparison and must match; percent, per-mille, scalars and different textual units are not interchangeable. Every match
stores both original numeric substrings and offsets. No value is inferred or rewritten.

## Confidence thresholds

- Relationship confidence is `high` at score `>= 0.82`, `medium` at `>= 0.62`, otherwise `low`.
- Caption targets less than `0.03` apart are treated as equivalent and left unresolved; alternatives within `0.12` are
  preserved as conflicts.
- Internal-title/section matching requires at least two shared content terms and lexical Jaccard `>= 0.45`.
- Approximate leakage requires normalized sequence similarity `>= 0.72`; a match without exact raw offsets is capped at
  medium confidence.
- Numeric `describes` evidence requires at least two compatible values. A numeric-only tie within `0.08` remains unlinked.
- Numeric comparison records raw forms, normalized values, scale, units, rounding method and tolerance. Supported
  equivalences include K/thousand/million expressions, small Arabic/English number words, rounded percentages and
  explicit AM/PM versus Arabic hour expressions. One rounded match remains insufficient, and aligned conflicts reject
  numeric-only candidates.
- Internal visual titles require a leading title-shaped span plus structural, caption, section-heading or explicit
  upstream support. `titled_by_section` requires normalized similarity of at least `0.72` and two shared content terms.
- Flattened single-line OCR has no structural-line support. Its candidate length is therefore bounded by the independent
  heading/caption title after comparison-only removal of structural numbering and trailing heading punctuation. It must
  begin at the first OCR word, end on a real word boundary, contain at least two non-generic content terms, and be an
  exact normalized match or a conservative near-exact alignment. Exact candidates are checked shortest-first. A
  near-exact match must have the same token count and may contain only one spelling-like substitution (token similarity
  at least `0.80`); insertion, deletion and an unrelated trailing label are rejected. Conflicting prefix lengths are
  rejected. Accepted evidence records the source ID, method (`prefix_heading_match` or `prefix_caption_match`), exact
  offsets, boundary rule, token alignment and similarity details.
- `relationships.json` contains `evidence_audit`, separating every accepted title/description candidate from rejected
  candidates and recording the exact rule and rejection reason.
- Table continuation and aside links require a combined evidence score of at least `0.62`.
- Geometry-free associations are capped or withheld; ambiguity never increases confidence.

## Scanned and missing-data limitations

Saved `figure_title`, `chart`, `table`, and `image` layout evidence can support associations. A layout title region without
a source text element and verifiable coordinates is reported as unresolved; the builder does not fabricate text or a box.
Page-text segments without per-span geometry are never upgraded to verified geometry. A missing visual placeholder is
produced only when Nass explicitly recorded one; absent metadata means completeness is unknown.

## Outputs and validation

Each invocation creates a write-once folder under:

`D:\agenticdoc_e2\results\document_relationships\<run_id>__<UTC timestamp>\`

It contains `relationships.json`, `visuals.json`, `lists.json`, `relations_report.md`, `relations_report.html`, and
`manifest.json`. Existing destinations are never overwritten. Before/after hashes cover available Nass source artifacts,
the source PDF, and Structure JSON inputs.

Validation checks resolved references, no self-links, exact derived offsets, valid null-offset missing placeholders, one
primary target per caption, complete visual inventory, ordered unique logical-table parts, acyclic lists, preserved
validation states, unchanged inputs, and review flags for ambiguity. Text-derived nodes still require integer offsets
that reproduce their source substring exactly.

## Tests

`tests/docprep/test_relations.py` contains synthetic Arabic/English coverage for captions, conflicts, internal titles,
leakage, numeric units and abbreviations, explicit/topic-only descriptions, table continuations, composites, asides,
nested and cross-page lists, RTL merged items, metadata, scanned layout titles, known-missing evidence, human review,
input immutability, and write-once output.

**The tests have NOT been executed.**

## Commands for the three saved runs

Run from the repository's `docprep/` folder.

Traffic report:

```powershell
python -m docprep.relations "D:\agenticdoc_e2\results\automatic_extraction\run_20261119_205601_e079cc" --structure "D:\agenticdoc_e2\results\document_structure\run_20261119_205601_e079cc__20261123T132328Z"
```

Arabic scanned:

```powershell
python -m docprep.relations "D:\agenticdoc_e2\results\automatic_extraction\run_20261119_212013_46ebeb" --structure "D:\agenticdoc_e2\results\document_structure\run_20261119_212013_46ebeb__20261123T132330Z"
```

Arabic digital/mixed:

```powershell
python -m docprep.relations "D:\agenticdoc_e2\results\automatic_extraction\run_20261119_210113_2d84c4" --structure "D:\agenticdoc_e2\results\document_structure\run_20261119_210113_2d84c4__20261123T130731Z"
```

Create the requested review template after selecting a new relationship output folder:

```powershell
python -m docprep.evaluate_relations --make-template "D:\agenticdoc_e2\results\document_relationships\<run_id>__<timestamp>" --pages 3,5,6,7,16,17
```

Evaluate a completed template:

```powershell
python -m docprep.evaluate_relations --evaluate "<completed-review.csv>" --relationships "D:\agenticdoc_e2\results\document_relationships\<run_id>__<timestamp>"
```

Run tests separately when ready:

```powershell
python -m pytest tests\docprep\test_relations.py -q
python -m pytest -q
```

## Known risks

- OCR text without line breaks yields an internal-title span only when independent heading/caption evidence identifies
  one strong, unambiguous leading prefix. Unsupported, generic, one-word KPI, numeric-dominated and conflicting cases
  remain unresolved.
- Repeated numbers across multiple visuals remain ambiguous and unlinked.
- Table continuation is deliberately conservative when headers/cells are unavailable.
- Layout-only `figure_title` regions cannot supply text that Nass did not save.
- Nested-list depth depends on saved ordering, marker context, and indentation; contradictions remain flat.

