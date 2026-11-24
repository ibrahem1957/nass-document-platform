# RAG consumer contract for `prepared_document.json` (schema 1.1.0)

This contract tells a retrieval or answering system how to read DocPrep's evidence signals. None of these
signals certifies that a statement in the source is true. They describe what was checked, what was found and how
the evidence may be used.

## Three independent axes

| Axis | Field | What it describes |
|---|---|---|
| Source extraction state | `elements[].validation_state` (from Nass) | Whether Nass processed the element normally (`ok`), processed it but asked for review (`needs_review`), or did not process it (`pending_review`, usually empty text). |
| Content evidence reliability | `elements[].evidence_reliability` and `rag_flags` | Whether a recorded quality finding concerns the element's own text or completeness. |
| Retrieval eligibility | `elements[].quality_inventory.reliability_dimensions.retrieval_readiness.state` | How the element may be used for retrieval, combining the above with structural and link uncertainty. |

`quality_inventory.categories_attempted` and `skipped_checks` record assessment coverage: which checks ran or
were skipped, and why.

## Status values

| Value | Field | Meaning | What it does **not** mean |
|---|---|---|---|
| `pending_review` | `validation_state` | Nass did not process this region (no OCR result), or holds it for review. Text is usually empty. | It does not mean the region was empty on the page. |
| `needs_review` | `validation_state`, or a `rag_flags` entry | Nass or a DocPrep finding asks for human review before the content is relied on. | It does not mean the content is wrong. |
| `unassessed` | `evidence_reliability` | No content-affecting finding is recorded for this element. | It does not mean "verified" or "high reliability". |
| `medium` / `low` | `evidence_reliability` | At least one content flag applies; `low` means a severe finding or known-missing content. | It does not mean an error is confirmed. |
| `no_recorded_concern` | `retrieval_readiness` | No flag, no recorded content concern, and certain placement and links. | It does not mean the facts are verified. |
| `content_unflagged_placement_or_links_uncertain` | `retrieval_readiness` | The text carries no flag, but its section placement or relationships are uncertain. | — |
| `content_concern_recorded` | `retrieval_readiness` | A source-text or completeness finding without a specific flag is recorded (often a page-level layout warning); details in `unflagged_content_concerns`. | — |
| `use_with_flags` | `retrieval_readiness` | One or more `rag_flags` apply. Show them with any answer that uses this element. | — |
| `review_before_use` | `retrieval_readiness` | Low reliability or known-missing content. Do not use it as evidence without review. | — |

## RAG flags

| Flag | Meaning | Required consumer behaviour |
|---|---|---|
| `table_cells_unknown` | The table is a flattened transcription; row/column associations are unknown. | Never answer a cell-specific question (a value for a given row and column) from it. |
| `possible_incomplete_content` | The saved region border cuts another layout block; content may be missing. | Do not treat the element as the complete content of that region. |
| `possible_foreign_text` | The crop margin contains a title or text block that may belong to a neighbouring visual (not confirmed). | Do not use the element's leading or trailing caption-like text as this visual's title or caption. |
| `ocr_noise`, `ocr_repetition_loop`, `bidi_order_suspect` | Suspected OCR corruption at recorded offsets. | Quote with caution; the text is preserved unchanged. |
| `chart_without_values`, `probable_missing_numeric_evidence` | The chart's values were probably not transcribed. | Do not infer values from it. |
| `known_missing` | The visual is recorded as missing. | No content is available. |
| `needs_review` | Review is requested (see above). | Use with flags or after review. |

## Tables

`visuals[].table_structure` is `source_cells` when the source run recorded row/column cells (e.g. a native
text-layer table), and `none` when only a flattened transcription exists. `cell_level_evidence_allowed` is true
only for `source_cells`. When it is false, the table text may still be retrieved as context, but must not be
used as verified evidence for a cell-specific numerical answer.

## Numbers

A number whose comma is not a valid thousands separator (e.g. `٦,٤`, `2,15`, `12,5`) is recorded with dimension
`ambiguous_number` and no value. It may be an OCR rendering of the Arabic decimal separator `٫` or a decimal
comma. Such tokens are never matched or compared. The source text keeps the original characters.

## Validation

`prepared_document_validation.json` includes `json_schema_valid` (full JSON Schema 2020-12 validation against
`schemas/prepared_document.schema.json`). A document with any validation error, including a missing validator,
makes the prepare stage exit non-zero and must not be indexed for retrieval.
