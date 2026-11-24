# RAG evidence policy (answer-time rules)

This policy says what the future RAG module **must enforce**. The meaning of every field it relies on is
defined in one place only: **`docprep/docs/RAG_CONSUMER_CONTRACT.md`**, together with
`docprep/schemas/prepared_document.schema.json`. If this file and the contract ever disagree, the
contract wins, and this file must be corrected.

Storing safety metadata in vector-store records is **not** enforcement. The rules below must be
applied in code, with tests that prove unsafe evidence is rejected or qualified.

## P0: admission to the index

| Rule | Check |
|---|---|
| Index only validated documents | `prepared_document_validation.json` → `summary.error == 0`, including the `json_schema_valid` check. Otherwise refuse the whole document. |
| Index DocPrep output, not raw extraction | Never index Nass `rag_chunks.jsonl`, `document.md`, or raw OCR alone. |
| Keep identity | Every chunk records `document_id`, `source_document.sha256`, `element_id`(s), `page`, `original_type`, `section_ids`, and, for spans, `span_id` and offsets. |
| Keep safety metadata | Every chunk records the `rag_flags`, `evidence_reliability`, `retrieval_readiness.state`, `validation_state` and, for tables, `table_structure` and `cell_level_evidence_allowed` of every source element in it. |
| Respect `review_before_use` | Do not retrieve these elements as answer evidence. They may be indexed only so the UI can show that unreviewed content exists. |
| Empty and placeholder content | `pending_review` elements with empty text produce no chunk. A `known_missing` visual produces at most a "content missing" note. |

## P1: answering

| Situation | Required behaviour |
|---|---|
| No retrieved evidence supports the answer | Say that the documents do not contain the information. **Never answer from model knowledge.** |
| Evidence is uncertain (`use_with_flags`, `needs_review`, `content_concern_recorded`, `medium`/`low` reliability) | Answer only with an explicit qualification naming the limitation. Show the source. |
| Conflicting evidence | Present the conflict with both sources; do not pick one silently. |
| Every substantive claim | Cite the PDF name, the page and, where possible, the `element_id`. |
| Retrieval score | Relevance is not reliability. A highly ranked chunk can still be flagged. |

## P2: tables (critical)

- **When `cell_level_evidence_allowed` is false** (`table_structure: "none"`, flag
  `table_cells_unknown`), **never answer a cell-specific question**, meaning a value for a given row
  and column. The flattened text does not establish which numbers belong to which row label.
  - Verified case: in a scanned table the row label follows its values, so a naive reading shifts every
    row by one.
  - Required response: explain that the extracted table structure is insufficient for a verified
    answer, and point to the page.
- **The flattened text may still serve as context** for non-cell questions, such as "what does this
  table cover?", with the flag shown.
- **When `cell_level_evidence_allowed` is true** (`source_cells`), use the cells in
  `elements[].source_extraction.cells`. Keep the row/column association as recorded, and keep units
  and headers.

## P3: numbers

- **Reproduce numbers exactly as written:** keep units, percentages, currencies, dates, signs and
  decimal precision.
- **Ambiguous separators:** a token recorded as `ambiguous_number` (e.g. `٦,٤`, `2,15`) has **no
  established value**.
  - Do not convert it. In particular, `٦,٤` must not become `64`, and must not be asserted as `6.4`
    either.
  - Quote it verbatim and state the ambiguity.
- **No inferred associations:** never infer that a number belongs to a label because the two are
  adjacent in OCR text, unless the structure is verified (`source_cells`, or an accepted relationship
  with evidence).
- **No arithmetic on uncertain numbers:** do not compute totals, differences or rankings from
  uncertain or flattened numbers.

## P4: visuals and captions

| Signal | Rule |
|---|---|
| `possible_foreign_text` | The visual's OCR **may** contain a neighbouring caption or text. Do not use its leading or trailing caption-like text as that visual's title or caption. It is potential, not confirmed, contamination. |
| Caption text inside crop OCR | Not proof that the caption belongs to that visual. |
| Unresolved caption candidates (`unresolved_caption_candidate`, `unresolved_layout_figure_title`, rejected `evidence_audit` entries) | Never treat as relationships. |
| `caption_of`, `describes`, `title_inside_visual` | Use only as recorded, with their `confidence`; `review_recommended` relationships must be qualified. |
| `co_located_in_section` | Proximity only, **never** a semantic link. |
| `chart_without_values`, `probable_missing_numeric_evidence` | Do not infer chart values. |

## P5: structure and ordering

- **Scanned documents:** do not rely on section hierarchy or reading-order adjacency.
  - They often have a single section.
  - Their reading order may be `proportional_to_text_block_area`, which is approximate.
  - Chunk by element and derived span, not by assumed paragraphs across visuals.
- **Visual boundaries:** do not merge text across `figure` / `table` boundaries into one chunk.

## P6: required tests for the RAG module

The RAG test suite must include, at least:

1. A document with a validation error is refused at indexing.
2. A cell-specific question against a `table_structure: "none"` table is answered with the
   "insufficient structure" response, never with a value.
3. A question whose only evidence is an `ambiguous_number` quotes it verbatim and states the
   ambiguity; it never outputs 64 for `٦,٤`.
4. A visual flagged `possible_foreign_text` is not cited as the source of its leaked caption.
5. A question with no supporting evidence produces an explicit "not found in the documents" answer.
6. Every answer cites the document, page and element.
7. The Studio extraction and preparation flows keep working with RAG disabled or not installed.
