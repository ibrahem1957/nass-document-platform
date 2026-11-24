# Component 3: Document Quality & Validation Layer

Component 3 is a deterministic, read-only inspection layer over one matching set of saved Nass, Structure,
and Relationship outputs. It reports evidence and review priorities. It never corrects OCR, changes source
text, invents missing content, changes structure or relationships, runs a model, or reads Ground Truth.

## Command

Run from the repository root in Windows PowerShell:

```powershell
python -m docprep.quality <run_dir> --structure <structure_out_dir> --relations <relations_out_dir>
```

The default root is `D:\agenticdoc_e2\results\document_quality`. Each invocation creates a new, write-once
`<run_id>__<UTC timestamp>` directory through a staging directory and atomic rename. Existing output is never
overwritten. `--out <directory>` selects another output root.

Exit codes are `0` for a valid output, `2` for invalid/mismatched inputs, `3` for a pre-existing output path,
and `4` when the output was written but integrity validation contains an error.

## Read-only inputs and provenance

The loader requires matching run IDs and reads:

- Nass `document.json`, `validation_report.json`, and `execution_manifest.json`;
- Structure `assignments.json`, `section_tree.json`, and `manifest.json`;
- Relationships `relationships.json`, `visuals.json`, `lists.json`, and `manifest.json`.

All available inputs are SHA-256 hashed before and after inspection. The manifest records both hash sets and
`inputs_unchanged`. Exact source text remains verbatim; findings reference source IDs and, when available,
exact offsets. Missing or ambiguous evidence stays missing or ambiguous.

## Outputs

| File | Purpose |
| --- | --- |
| `quality_findings.json` | Ranked findings, evidence, source/page references, limitations, and review excerpts |
| `quality_summary.json` | Document, page, section, and element-type review scorecards |
| `quality_inventory.json` | Every original source element exactly once, checks attempted, skipped-check reasons, and findings |
| `quality_report.md` | Portable review report |
| `quality_report.html` | Self-contained, escaped local HTML review report |
| `quality_manifest.json` | Input/output hashes, configuration, provenance, warnings, and validation summary |

Scorecards are review queues, not OCR accuracy percentages. Finding priority combines severity and confidence,
with extra weight for source-extraction and numeric-integrity evidence. It does not claim correctness.

## Checks

The implementation groups checks into nine categories:

1. Source extraction: saved `needs_review` states, explicit OCR confidence, validation flags, and saved failed,
   skipped, or incomplete execution records.
2. OCR corruption: conservative malformed arithmetic, truncation, repeated-token, mojibake, invalid-character,
   punctuation, and mixed-script signals. Technical tokens are excluded where practical.
3. Numeric integrity: agreement or conflict only across an existing `describes` relationship with compatible
   units/dimensions and locally aligned labels. Unrelated numbers are not compared.
4. Duplicate/leakage: saved leakage relationships and strong same-page near duplicates, excluding known
   headers, footers, captions, and intentional logical-table continuation headers.
5. Structure quality: uncertain headings/levels, low-confidence or unassigned content, unresolved list flow,
   and approximate layout associations.
6. Relationship quality: Component 2 validation findings, review-recommended semantic relationships, unlinked visuals,
   and unresolved composite parents. The intentionally weak `co_located_in_section` relation is preserved but does not
   create `uncertain_existing_relationship`. An unlinked visual is not called an extraction failure.
7. Table quality: inconsistent Markdown widths, flat table representations, malformed numeric expressions, and
   confirmed repeated continuation headers.
8. Figure/chart quality: sparse saved OCR is flagged as unverified, never as proven missing content.
9. Cross-page continuity: saved list uncertainty, headings at page boundaries, and conservative paragraph
   continuation candidates supported by order, section, geometry, and punctuation together.

Important thresholds are recorded in `quality_manifest.json`: OCR confidence below `0.60`, same-page duplicate
similarity at least `0.90` for text of at least 40 characters, sparse visual text below three content terms,
two locally shared label terms for numeric comparison, and bottom-of-page Y at least 700 points for paragraph
continuation followed by next-page Y at most 120 points. These are review heuristics, not truth labels.

### Saved-status semantics

- `recorded_processing_failure` requires an explicit failed/error processing, extraction, OCR, or page-job record.
- A failed optional layout/validation/inspection step is `optional_verification_failure` or
  `optional_layout_verification_failure`; it does not count as a confirmed extraction failure.
- A nonfatal `layout_review` remains a low-severity `nonfatal_layout_warning` with its original evidence.
- A saved skipped operation is `recorded_processing_skip`.
- An `incomplete_region` warning is `incomplete_region_coverage`: content may be missing, but failure is unverified.
- An incomplete or otherwise unknown execution status remains `unknown_processing_status`.

### Geometry reliability

Every inventory record exposes one of four geometry classes:

- `precise`: a paragraph/line element has its own valid source or assignment bounding box;
- `approximate`: a derived text span has a saved, approximate bounding box;
- `page_level_only`: a `page_text` element has only a full-page OCR container;
- `unavailable`: no usable geometry is saved.

Cross-page paragraph continuity requires reliable paragraph-level geometry on both pages, a previous paragraph
near the page bottom, a next paragraph near the page top, adjacent reading order, the same section, and no
terminal punctuation. Page-level-only or unavailable geometry produces no finding; the inventory records the
check as skipped with its reason. No coordinates are inferred.

## Human review

Create a blank review template for selected pages:

```powershell
python -m docprep.evaluate_quality --make-template "<quality_output_dir>" --pages 3,5,6,7,9,12,16,17
```

Fill only `human_verdict` and `reviewer_note`. Allowed verdicts are `confirmed_issue`, `false_positive`,
`uncertain`, and `not_reviewed`. Then create a separate write-once evaluation:

```powershell
python -m docprep.evaluate_quality --evaluate "<completed_template.csv>" --quality "<quality_output_dir>"
```

Templates and evaluations are stored beside the quality root under `quality_reviews`; they never rewrite the
quality output. Partial review coverage is reported with its denominator.

## Tests

The synthetic suite is `tests/docprep/test_quality.py`. The false-positive validation run on 2026-11-23 produced
44 targeted passes and 227 full DocPrep passes. Two PDF-dependent tests were skipped because the active Python
3.12 environment has no `pymupdf` package; no package was installed automatically. Run with:

```powershell
python -m pytest tests/docprep/test_quality.py -q
python -m pytest tests/docprep -q
```

## Exact saved-run commands for later execution

Traffic report:

```powershell
python -m docprep.quality "D:\agenticdoc_e2\results\automatic_extraction\run_20261119_205601_e079cc" --structure "D:\agenticdoc_e2\results\document_structure\run_20261119_205601_e079cc__20261123T132328Z" --relations "D:\agenticdoc_e2\results\document_relationships\run_20261119_205601_e079cc__20261123T141226Z"
```

Arabic scanned:

```powershell
python -m docprep.quality "D:\agenticdoc_e2\results\automatic_extraction\run_20261119_212013_46ebeb" --structure "D:\agenticdoc_e2\results\document_structure\run_20261119_212013_46ebeb__20261123T132330Z" --relations "D:\agenticdoc_e2\results\document_relationships\run_20261119_212013_46ebeb__20261123T141227Z"
```

Arabic digital/mixed:

```powershell
python -m docprep.quality "D:\agenticdoc_e2\results\automatic_extraction\run_20261119_210113_2d84c4" --structure "D:\agenticdoc_e2\results\document_structure\run_20261119_210113_2d84c4__20261123T130731Z" --relations "D:\agenticdoc_e2\results\document_relationships\run_20261119_210113_2d84c4__20261123T141227Z"
```

## Limitations

RAG-readiness evidence signals are annotations, not retrieval policy:

- `bidi_order_suspect` requires a standalone LTR line interrupting an unfinished Arabic sequence with a following
  Arabic connector; ordinary mixed-language sentences are not enough.
- `chart_without_values` requires an explicit saved chart classification and no usable numeric token.
- `ocr_vocabulary_noise` requires OCR-derived visual text, edit distance 1–2, and strong independent local alignment:
  either the same surrounding phrase in at least two distinct body elements, or an aligned enclosing heading, explicit
  caption, or explicitly verified same-source alternative. Shared topics, section membership and global frequency are
  insufficient. Common morphology and ambiguous equal-strength candidates are rejected. Retained findings preserve
  exact offsets and supporting source IDs and never replace the token.
- Malformed table expressions become `source_content_anomaly` only with explicit source-origin provenance;
  otherwise they remain `unverified_anomaly`.
- Inventory records expose RAG flags and evidence reliability. The default is `unassessed`, never automatically
  `high` merely because no finding exists.
- Only high-confidence `ocr_vocabulary_noise` with recorded `strong` evidence adds `ocr_noise`/`needs_review` and a
  proportionate `medium` reliability state. Abstained or weak lexical candidates do not change RAG reliability;
  independent upstream `needs_review` remains preserved.
- Content reliability, relationship reliability, and description completeness remain separate. Relationship-quality
  findings stay attached and reviewable through finding/relationship IDs but do not add content `needs_review` or lower
  element/visual evidence reliability. `visual_without_caption_or_description` remains informational only. Every actual
  content-reliability downgrade records at least one explicit provenance or finding reason.

- No finding proves what the original page says; source comparison remains the authority.
- Saved metadata quality limits all checks. A region absent from every input cannot be reconstructed here.
- Text-pattern rules can produce false positives in technical, multilingual, and deliberately stylized text.
- Numeric conflicts require an existing trusted relationship and compatible context, so conservative logic can
  miss real conflicts.
- Flat OCR cannot establish table cell coordinates or row/column semantics.
- Cross-page continuity is a review candidate only and never causes automatic merging.
- Human review is required before treating findings as validated defects.
