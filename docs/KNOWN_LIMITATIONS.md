# Known limitations (verified)

Each limitation below was observed in real saved runs or in the source code. "Observed on" refers to
benchmark runs on the original development machine. Those outputs are not distributed, because they
derive from private PDFs.

## Extraction evidence

| Limitation | Evidence | Consequence |
|---|---|---|
| **Scanned tables have no trustworthy cell structure.** Qari returns flattened text; Nass records `cells: null`. | Arabic scanned benchmark, page 6 (`p6_e1`). The source row «الخط 1 = 3.2 / 38,920 / 412» is transcribed as «الخط 3.2 38,920 412 الخط 1 4.8 …», so each row label *follows* its values. | DocPrep sets `table_structure: "none"`, `cell_level_evidence_allowed: false`, flag `table_cells_unknown`. No cell-specific answers are allowed. |
| **Scanned reading order is approximate.** Qari gives no paragraph coordinates. | 8 of 10 pages of the scanned benchmark use `proportional_to_text_block_area`. | Body text may be placed one segment early or late relative to visuals. This is reported as `ambiguous_reading_order` (structural dimension). |
| **Many scanned figure captions remain unresolved.** | Scanned benchmark: 18 caption spans; 2 linked (page 1, verified visually); 16 unresolved with recorded reasons. | Answers to "Figure N" questions on scanned documents are unreliable. |
| **Caption-title attachment can be wrong** when a title sits between two visuals. | Page 7 of the scanned benchmark: the table caption had been attached to the form below it. This is now fixed: such titles are `ambiguous` unless the geometry is decisive. | Ambiguous titles stay unattached and flagged. |
| **Margin leakage diagnostics are not proof.** | `dedup.suspected_margin_leakage` / `margin_leakage_risk` record geometric overlap. Caption labels in crop text are unattributed. | `possible_foreign_text` is a warning, not a confirmed error. |
| **Body text can repeat inside visual crops.** | Scanned benchmark `p5_e0`/`p5_e2` and `p9_e0`/`p9_e3` share an identical sentence. | No quality finding covers this yet, so retrieval can return the same sentence twice. |
| **Scanned documents often have no reliable section hierarchy.** | The scanned benchmark has 1 section: the visible headings are mostly table and figure titles. | Do not chunk scanned documents by section. |
| **Ambiguous number separators.** | OCR renders «٦٫٤» (6.4) as «٦,٤». | DocPrep records `ambiguous_number` with no value; it was previously misread as 64. |
| **Chart values are not structurally associated with labels.** | Any chart OCR | Flags such as `chart_without_values` and `probable_missing_numeric_evidence` apply when values are missing. Otherwise the values are plain text. |
| **Region coverage may be incomplete.** | A Nass `incomplete_region` flag means the region border cuts another block; the proposed expansion is never applied. | Flag `possible_incomplete_content`. |

## Infrastructure and environment

| Limitation | Detail |
|---|---|
| **Qari OCR depends on external GPU infrastructure.** | The previous RunPod service is stopped. No working endpoint is included. Pod configuration is a handoff dependency (`docs/RUNPOD_SETUP.md`). |
| **Studio cannot start an extraction without configuration.** | Both `QARI_SERVICE_URL` (trusted for its mode) and `LAYOUT_MODEL_DIR` are required, even for digital PDFs. |
| **The PP-DocLayout model is not bundled.** | Download `PP-DocLayout_plus-L_infer` separately (see README). The app never downloads it. |
| **DocPrep default output roots point to `D:\agenticdoc_e2\…`.** | These are only used when `--out` is omitted. Studio always passes `--out`. Set `DOCPREP_OUTPUT_ROOT` or pass `--out` yourself. |
| **The Nass Extraction Lab README describes the original machine.** | `nass/lab/README.md` paths (`D:\agenticdoc_e2`, `Project sdaia`) refer to the original layout; the Nass root is now `nass/`. |
| **Studio's readable views show a generic "requires review" mark.** | The specific `rag_flags`, `table_structure` and `retrieval_readiness` are visible only in JSON View or download. |
| **Single-user, localhost only.** | Studio is a development application, not a multi-tenant server. |

## Tests

| Limitation | Detail |
|---|---|
| **19 pre-existing Nass test failures.** | All are outside the Automatic Extraction path. They cover hosted-API evaluation (oxen, staged budgets, saved-request reuse, the shipped document registry), a stale `pp_doclayout_local` signature test, and a reorder-config default (`min_arabic_letter_share` 0.3 vs 0.0). The identical failure set exists in the original project. See `docs/TESTING.md`. |
| **5 Nass tests skip.** | User benchmark PDFs are absent, and LibreOffice is not installed. |
| **4 DocPrep tests skip.** | The private frozen v1 structure reference is not distributed. |

## RAG MVP scope (recommended)

1. **Digital** Arabic, English and mixed documents first. Their text layer, typography, hierarchy and
   native table cells (`source_cells`) are reliable enough for grounded answers with citations.
2. **Scanned documents: narrative content only.** Apply the `rag_flags` qualifications, and never use
   section or reading-order adjacency as evidence.
3. **No cell-specific answers from flattened scanned tables** (`cell_level_evidence_allowed=false`).
4. **No values for ambiguous numbers.** Quote them verbatim.
