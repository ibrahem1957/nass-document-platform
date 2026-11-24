# Document Understanding & Preparation

Component 2 documentation: [Content Relationship Builder](docs/CONTENT_RELATIONSHIP_BUILDER.md).

Component 3 documentation: [Document Quality & Validation Layer](docs/DOCUMENT_QUALITY_VALIDATION_LAYER.md).

Component 4 documentation: [Unified Prepared Document Builder](docs/PREPARED_DOCUMENT_SCHEMA.md).

Local review UI: [DocPrep Document Inspector](docs/DOCUMENT_INSPECTOR.md).

**Component 1: Document Structure Builder** (package `docprep`)

The Structure Builder reads a finished Nass Automatic Extraction run. It finds the document's real headings, infers their hierarchy, and assigns every piece of content to a section, with the evidence for every decision.

It does **not** chunk, embed, retrieve, rerank or generate answers. It runs no OCR, re-extracts nothing from the PDF, and calls no model or external service.

Nass stays responsible for PDF extraction and is not modified. This project imports nothing from Nass: it only reads the saved files of a run.

## Inputs (read-only)

`<run_dir>` = `D:\agenticdoc_e2\results\automatic_extraction\<run_id>`

| File | Use | If missing |
| --- | --- | --- |
| `document.json` | Elements, pages, types, boxes, reading order, validation states | Error (exit code 2) |
| `validation_report.json` | Nass page flags; reading-order flags lower assignment confidence | Warning |
| `execution_manifest.json` | Hashed for provenance | Warning |
| `plan/layout.json` | Saved PP-DocLayout boxes (scanned pages) | Warning; no layout signals |
| `source.pdf` | Typography of digital and mixed pages (PyMuPDF spans) | Warning; no typography signals |

The input files are hashed (SHA-256) before and after processing. Any change is reported as a validation error.

## Usage

```bash
pip install -r requirements.txt
python -m docprep.structure <run_dir> [--out <root>] [--config <toml>] [--no-pdf] [--no-semantic]
```

- **Output root:** `--out`, else the `DOCPREP_OUTPUT_ROOT` environment variable, else `output.root` in `config/docprep.toml` (default `D:\agenticdoc_e2\results\document_structure`).
- **Exit codes:**
  - `0` OK;
  - `2` input or configuration error;
  - `3` the output folder already exists;
  - `4` validation errors (the outputs are still written, for review).

## Outputs (write-once)

`<root>/<run_id>__<UTC timestamp>/` is written to a hidden staging folder, then renamed in one step. Its files are set read-only, and an existing folder is never overwritten.

| File | Content |
| --- | --- |
| `section_tree.json` | Outline; sections (title, level, parent, children, pages, heading source IDs, content units, list nodes, confidence, evidence); all heading candidates; unassigned content; style classes; body typography |
| `assignments.json` | The source inventory (each original element exactly once, with its verbatim text and any derived spans), plus one assignment per assignable unit |
| `structure_report.md` | Outline, candidate counts, promoted, rejected and excluded candidates, low-confidence assignments, unassigned content, classes, warnings, validation |
| `structure_report.html` | Self-contained review page (no external resources) with filters for confidence, promoted headings, unassigned content, page and page type |
| `manifest.json` | Source run, input hashes (before and after), version, full configuration and thresholds, timestamps, output hashes, validation summary |

## Processing stages

1. **Inputs** (`inputs.py`): validate and load the run, keeping the document-wide Nass order (the page's `element_ids`, then `reading_order`).
2. **Units** (`features.py`, `segmentation.py`): every element is one assignable unit. A scanned `page_text` element is split into **derived spans**, and the original element stays in the inventory exactly once.
3. **Signals:**
   - typography from `source.pdf` (`typography.py`);
   - saved layout boxes (`layout.py`);
   - text patterns (`text_patterns.py`);
   - geometry (`features.py`);
   - page furniture (`furniture.py`);
   - the optional semantic check (`semantic.py`).
4. **Heading detection** (`detection.py`): exclusions, then an explainable additive score, document-wide style consistency, and the merging of wrapped lines.
5. **Levels** (`hierarchy.py`): document title, typographic classes, numbering depth, and pattern classes.
6. **Section tree** (`tree.py`): sections, list nodes, and one assignment per unit with confidence and reasons.
7. **Validation** (`validation.py`), then the **outputs** (`pipeline.py`, `report_md.py`, `report_html.py`, `output.py`).

## Signals

Each signal is recorded with its contribution and a detail string. The weights are in `[weights]`.

| Category | Signals |
| --- | --- |
| Source type | Nass type `heading` (a signal, not a decision) |
| Typography (digital and mixed pages only) | Size relative to the character-weighted body size; bold share; distinct colour; mixed sizes |
| Layout (saved PP-DocLayout) | `doc_title` and `paragraph_title` (reduced for a low score or an ambiguous overlap); `text`; a `header` near the top (weak) |
| Patterns | Numbering (1. 1) 1.1 ١) (a) أ) IV.), Arabic ordinals (أولاً, ثانياً …), keywords (Chapter, الفصل …); short colon label; terminal punctuation; short standalone line; bullet prefix |
| Position and context | Space before; centred; first text of the document; followed by body text; followed by a same-style candidate |
| Style | Same style as at least two confident headings (document-wide consistency) |
| Semantic (optional) | Does the following text develop the candidate's topic? (model-free; see below) |

**Typography attachment.**
- Each PDF span is attached to at most one element: the one with the largest overlap.
- The attachment is accepted only when the span characters (NFKC, order-insensitive) match the element's text, with at least 80% of span characters in the element and at least 60% of the element covered.
- Otherwise typography is reported as unavailable, rather than taken from a neighbour.

**Exclusions**, with reasons:
- captions (pattern, layout label or derived-span role);
- repeated headers and footers;
- page numbers;
- numeric labels;
- axis-like number sequences;
- OCR fragments;
- bullet items without a colon;
- text that is too long or spans too many lines;
- an ordinary sentence ending with a colon (more than `colon_heading_max_words` words, with no strong typography or layout).

**Thresholds** (`[thresholds]`):

| Setting | Default | Meaning |
| --- | --- | --- |
| `promote` | 0.45 | Score needed to promote a candidate |
| `high` | 0.70 | Detection confidence "high" |
| `medium` | 0.55 | Detection confidence "medium" |
| `ambiguity_band` | 0.08 | Decisions this close to `promote` are marked ambiguous |

Every other parameter is in `config/docprep.toml`.

## Headings that Nass classified as `text`

The Nass element type is only one signal. A `text` element becomes a heading when other evidence adds up, for example:
- bold or larger type;
- a short colon label followed by body text;
- numbering;
- a layout heading region.

A colon alone never makes a heading. A short colon label with no other evidence stays a *rejected, ambiguous* candidate and is listed for review. The report marks every heading promoted from a non-heading type.

## Scanned `page_text` elements

The page OCR of a scanned page is one element, usually a single line with no coordinates.

- **Cuts come only from text evidence:**
  - sentence ends;
  - inline figure and table captions (`الشكل 1:` / `Figure 1:`);
  - inline bullets and ordinals;
  - the end of a short colon label;
  - a title echoed at the start of the following sentence.
- **Exact partition:** the spans cover the original text exactly, with contiguous offsets, and the concatenated spans equal the original. Punctuation-only remainders join the previous span, so nothing is dropped. IDs are `<element_id>_span_001`, `_span_002`, ….
- **Layout corroborates, it does not locate.** The segment's layout blocks (`body_segment.text_blocks`) can confirm that, for example, a `paragraph_title` exists in the segment (support `both`), but never where it is in the text.
  - With a single layout block, each span gets that block as a *container* only.
  - When spans and blocks match one to one by order and role, each span gets its block as an *approximate* location.
  - In all other cases a span has no geometry. The full-page box is never used as a paragraph position.
- **Uncertain spans:** a span is marked uncertain when its role rests on text alone (title echo, colon label), or when no boundary was found.
- **Detection confidence:** a heading built from an uncertain span is capped at medium.

## Levels

1. **Document title (level 0):**
   - when typography is available: the largest heading among the first units;
   - otherwise: a `doc_title` layout region, or (pattern-only documents) the first heading-like text.
2. **Typographic classes:** classes of size, weight and colour, ordered by size, then weight, then a colour different from the body. The class rank gives levels 1, 2, …
   - **Confidence:** high with a size gap of at least `distinct_size_gap` between classes; medium when classes differ only by weight or colour, or when there is a single class.
3. **Numbering depth:** within a class, numbering depth adds levels (1 → 1.1). When numbering depth contradicts typography, this is recorded as a level conflict, with low level confidence.
4. **Headings without typography:**
   - when the document has typographic headings, a heading takes the level of typographic headings with the same text pattern; failing that, the top level, marked ambiguous;
   - in documents with no typography at all, headings are ranked by text-pattern class.
   - **Confidence:** always low; a flat single-pattern result is marked ambiguous.
5. **Clamping:** levels are clamped to `max_levels`.

## Assignments

- **Section boundaries:** a section ends at the next heading of the same or a higher level. Sections continue across pages, and a page boundary never starts a section.
- **Page furniture:** repeated headers, footers and page numbers are taken out and do not interrupt a section.
- **Heading at the bottom of a page:** the first content on the next page is assigned to it, with a reason.
- **Lists:** consecutive bullets, and runs of two or more numbered items that are not headings, form list nodes.
- **Figures, tables and captions** stay separate roles inside their section.
- **Unassigned content:** content before the first heading (when there is no document title), and everything in a document with no headings, is *unassigned*, with reasons.

**Confidence**, kept separate for heading detection, heading level and assignment:
- An assignment is capped by its section heading's detection confidence.
- It is capped at medium for an uncertain derived span.
- It is capped at medium or low by a Nass reading-order flag on the page.
- It is low after a near-miss heading candidate.
- Low means *review recommended* and non-deterministic.
- The Nass `validation_state` is preserved unchanged: structure confidence never overrides OCR confidence.

## Semantic verification (optional, model-free)

`HeuristicVerifier` looks at the text that follows a candidate:
- **Supporting cues:**
  - capped content-term overlap with light Arabic and English stemming;
  - a colon introducing the following text;
  - a longer explanatory passage;
  - explanatory openings.
- **Weakening cues:**
  - a continuation opening;
  - a sentence-like candidate;
  - very short following text.

**Limits:**
- Its weight is small (from −0.05 to +0.10), so it can never promote a heading on its own.
- Shared words are not proof, and their absence is not a rejection.
- Every result records its limitations.

A local semantic model can be added later by implementing the `SemanticVerifier` protocol and registering it in `make_verifier`, without changing the pipeline.

## Validation

1. Each original element is represented exactly once.
2. Every derived span has an existing parent element.
3. The spans partition their element's text exactly.
4. Each assignable unit has exactly one assignment.
5. Section IDs are unique.
6. Section headings are verbatim source text.
7. The hierarchy is acyclic and consistent.
8. The original reading order is preserved.
9. Page references are valid.
10. Multi-page section ranges match their content.
11. Every low-confidence assignment is marked for review.
12. The input files are unchanged (SHA-256).

## Tests

The tests are in `tests/docprep/`. They use synthetic data only; the PDF tests generate fixtures with PyMuPDF's built-in fonts. On 2026-11-23, 227 tests passed and two PDF tests were skipped because `pymupdf` was absent from the active Python 3.12 environment.

```bash
python -m pytest tests/docprep -q
```

## Round 2 changes (accuracy and evaluation)

### Body-style text gets no heading bonus
- **Body style** is the predominant size class, font weight and colour, measured character by character over the document. Fake bold is not part of it.
- **What a body-style unit loses:** it gets a 0.0 `body_style` signal, no size prominence, and no `style_consistency` bonus.
- **What can still promote it:** numbering, layout, an isolated short label, or real weight or colour differences.

### List introductions (`list_intro`)
A line becomes a list introduction only when all of these hold:
- it ends with a colon;
- the next content unit is a list item (a start or end marker, or a numbered sentence);
- it has no numbering, list marker, size, weight or layout evidence of a heading;
- its style matches the body text, or typography is unavailable;
- at least one of:
  - a sentence cue: an opening connective or verb (`لذلك`, `ويمكن`, `وتتضمن`, `These`…) or a closing introducing phrase (`مثل:`, `ما يلي:`, `من خلال:`, `بـ:`, `include:`…);
  - a body-like length: `max(list_intro_min_words, ceil(median anchor-heading words × list_intro_heading_factor))`. Anchors are Nass headings and visually prominent lines.

A list introduction stays in the enclosing section and is recorded as the list's `intro_unit_id`. A colon alone is never enough, and a numbered or visually prominent colon heading followed by a list remains a heading.

### Levels
- **Raw level:** the style and numbering class still decides nesting (raw level, used for stack ordering).
- **Final level:** the parent's level + 1 (a top-level section is 1; the document title is 0).
- **Collapsed gaps:** a collapsed gap such as L1 → raw L3 is recorded in `level_notes`, and caps level confidence at medium.
- **Uncertain parents:** when the parent's own level confidence is low, the level is marked ambiguous.
- **Genuine nesting** (L1 > L2 > L3 with real intermediate headings) is unchanged.

### Fake bold (Word fill-and-stroke text)
**How it is detected.** `page.get_texttrace()` records filled (type 0) and stroked (type 1) text. A stroked trace counts only when:
- an identical fill trace exists within 1 pt;
- its centre lies inside the element box;
- its characters match the element text (NFKC, order-insensitive).

**Status per element:** `fill_and_stroke` (≥ `bold_share` of the characters), `none`, or `unknown` (no trace, or no reliable mapping).

**Weight and limits:**
- The weight is `fake_bold` = 0.05.
- It is ignored when the body text itself is mostly fake bold.
- It is never part of the style class.
- A candidate is not promoted when the threshold is passed only because of fake bold.

### Lists
- **Markers:** start markers (Arabic and Western bullets, dashes), and end-of-line markers on RTL pages after logical reordering (`Power BI -`, lines of at most 12 words).
- **Numbered lists:** runs of two or more consecutive numbered non-headings. A numbered line followed directly by the next numbered line gets a `numbered_list_sequence` penalty of −0.10.
- **Wrapped lines:** a short line (≤ 8 words) after an unfinished item is linked as a continuation only when the geometry confirms it: a gap of at most one line and horizontal overlap.
  - Otherwise, for example on another page or without coordinates, it stays a separate paragraph, flagged as a possible continuation.
- **Lists continue across pages.** Items keep their element and span IDs, offsets, page and marker position.

### Callouts and side boxes (`aside_label`, `aside_text`)
**Detection:** a short or colon-ended, unnumbered label is an aside label when:
- it is narrower than 60% of the document's body column (the median width of paragraphs with at least 12 words);
- it sits beside another element (vertical overlap, horizontal gap of at least 4 pt), or above a narrow column that runs beside one.

**Multi-column pages are skipped:** three or more column-width text blocks on both halves of the page.

**Handling:**
- Aside labels are excluded from the heading hierarchy.
- They stay in the reading flow of their section, marked `role_uncertain`.
- Linking a callout to the content it refers to is left to the future Content Relationship Builder.

### Confidence
| Dimension | Basis |
| --- | --- |
| Heading detection | Score thresholds: high ≥ 0.70, medium ≥ 0.55, promoted ≥ 0.45. An uncertain derived span caps it at medium. |
| Heading level | High only for a distinct typographic class (size gap ≥ 1 pt). Medium for weight or colour only, a single class, a collapsed gap, or a body-like class with numbering or layout evidence. Low for a body-like class without that evidence, pattern-only levels, or conflicts. |
| Assignment | Capped by its section heading's detection confidence, an uncertain derived span, or a Nass reading-order flag. One step lower after a semantic warning. Low after a near-miss heading candidate. |

The Nass `validation_state` is always preserved separately.

### Semantic checks never move content
- **Next heading is a child:** a heading whose next unit is a child heading gets no "no following content" penalty. The check is deferred, and the section tree records "content is in child sections".
- **Content that matches the next heading:** a paragraph that shares at least 2 terms with the **next** heading and none with its own is flagged and lowered one confidence step, and review is recommended. It is never moved.

### Evaluation framework
```bash
python -m docprep.evaluate <structure_out_dir> --reference docprep/eval/references/traffic_report_review_v1.json [--baseline <previous_out_dir>] [--out <root>]
```
**The reference** `traffic_report_review_v1.json` is frozen:
- its SHA-256 is in `docprep/eval/references/SHA256SUMS`, and a modified copy is refused;
- corrections must go into a new versioned file;
- its provenance says the judgments came from the first automated review, so they are not human-verified.

**Metrics:**
- They use confirmed, individually labelled cases only. Uncertain and unverified cases are listed separately.
- Every metric shows its denominator, and is `n/a` when it has no confirmed cases.
- Historical aggregate counts are reported separately.

**Outputs:** `evaluation.json`, `evaluation.md` and `manifest.json` go into `<results>/document_structure_evaluation/eval__<output>__<timestamp>/`. They are write-once and read-only, and an output root inside an input folder is refused.

**Exit codes:** `0`, `2` (input error), `3` (the output folder exists).

## Round 3 changes

### Semantic checks: document-wide terms and generic headings
- **Common terms:** a term counts as common when it appears in at least `max(2, ceil(common_term_section_share × sections))` sections (default share 0.30). Each section's title and directly assigned content are counted once. Common terms are ignored when comparing paragraphs with headings.
  - With fewer than `min_sections_for_df` sections (default 3), no term is common.
- **Stopwords:** Arabic and English stopword lists, plus light stemming.
- **When the check is skipped:**
  - the paragraph's own heading or the next heading has fewer than `min_heading_terms` (default 2) meaningful words;
  - the next heading's detection confidence is low (`skip_when_next_heading_uncertain`), so an unconfirmed structure is never used as evidence;
  - the paragraph is in the last section.
- **Recording:** a skipped check never changes confidence. The reason is stored in the assignment's `semantic_check` field (`skipped`, `passed` or `flagged`, with the shared terms).
- **Flagged paragraphs** are lowered one confidence step and marked for review. They are never moved.

### Human review template and human (v2) references
```bash
python -m docprep.evaluate --make-template <structure_out_dir> --pages 1,2,13 [--out <root>]
```
**Template output:** a new folder `<results>/document_structure_review/review__<output>__<timestamp>/` (it is never reused), containing:
- `review_template.csv` (UTF-8 with BOM) and `review_template.json`;
- `README_review.md` and `manifest.json`.

These files stay writable, so you can fill them in.

**Rows:** one per heading candidate on the selected pages (promoted, rejected, and excluded, except plain list items and page furniture), plus every `list_intro` and `aside_label`.

**Columns:**
- identifiers: `row_id`, `element_id` (merged elements joined with `+`), `span_id`, `unit_ids`, `candidate_id`;
- the source text and the tool's view: `page`, `source_text` (verbatim), `tool_verdict`, `tool_role`, `tool_level`, `tool_reason`;
- **the human columns, always blank:** `human_verdict`, `human_level`, `human_note`.

The template contains no page images.

**Filling it in:**
- **`human_verdict`:** `heading`, `not_heading`, `excluded` or `uncertain`. Blank means not reviewed.
- **`human_level`:** an integer from 0 to 9, only with `heading`.
- **Everything else must stay unchanged.** On loading, identifiers are checked against the structure output, `source_text` must be verbatim, and invalid verdicts or levels are rejected with row numbers.
- Save the completed copy under a new name.

**Evaluate:**
```bash
python -m docprep.evaluate <structure_out_dir> --reference docprep/eval/references/traffic_report_review_v1.json --human-reference <completed.csv> [--baseline <previous_out_dir>]
```
**The report keeps three separate views:**
1. the v1 reference (tool-reviewed);
2. the human reference, only its reviewed rows, with coverage counts and a "partial review" note;
3. a combined heading view in which a **labelled** human row replaces the v1 label with exactly the same element(s) or span. Its composition is stated, and assignments and continuations stay v1-only.

**Also reported:** disagreements between v1 and human labels. Uncertain human labels are never counted as correct. Neither reference is modified.

### Scanned heading diagnostics
`section_tree.json` (`layout_heading_diagnostics`), `structure_report.md` and `structure_report.html` list every saved `doc_title` / `paragraph_title` region. Each entry shows:
- the page, label, box (pt) and detector score;
- the matched element, span and text;
- the evidence and the match confidence;
- whether it became a section heading, or why it was not matched.

**Association levels:**
- **verified:** an element's own box lies in the region (containment ≥ 0.8), or the page OCR segment was built by Nass from exactly this block.
- **approximate:** a derived span was matched to the block by order and role (medium), or it is the only heading-like span of a multi-block segment (low).
- **none:** the region lies inside a figure or table crop, is referenced by no extracted element, or its segment has no (or several) heading-like spans.

An unmatched region is never treated as a heading, and no coordinates are invented for OCR spans.

## Known limitations

- **Arabic typography in the tests:** the synthetic PDFs use base-14 fonts (English only), so Arabic typography is covered by unit tests on spans, and by real runs once they are inspected.
- **Scanned pages:** these have no paragraph coordinates, so levels on scanned-only documents are pattern-based with low confidence.
- **Title echo:** this is a heuristic. It is marked uncertain and only supported by layout presence.
- **Multi-level decimal numbering** can match quantities such as "3.2 million". It is only a supporting signal.
- **Wrapped headings** are merged only with matching typography and geometry (not for scanned spans).
- **Running headers on scanned pages** are detected only as repeated leading OCR text on three or more pages.
- **Tables continuing across pages** keep their Nass elements unchanged; this component does not merge them.
- **The semantic layer** is lexical only.

## Commands for the reference runs (not executed yet)

```bash
python -m docprep.structure "D:\agenticdoc_e2\results\automatic_extraction\run_20261119_205601_e079cc"
python -m docprep.structure "D:\agenticdoc_e2\results\automatic_extraction\run_20261119_212013_46ebeb"
python -m docprep.structure "D:\agenticdoc_e2\results\automatic_extraction\run_20261119_210113_2d84c4"
```
