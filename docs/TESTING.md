# Testing

## Run everything

From the repository root, with the root virtual environment activated:

```powershell
.\scripts\run_tests.ps1
```

Each suite runs from its own folder, because each component has its own import root:

```powershell
cd docprep; python -m pytest tests/docprep -q        # DocPrep
cd studio;  python -m pytest -q                      # Studio
cd nass;    python -m pytest tests/lab -q            # Nass
```

**No test contacts** Qari, RunPod, PP-DocLayout weights or any network service. All tests use
synthetic fixtures generated inside the repository, or temporary files.

## Results at handoff (2026-11-25, Windows 11, Python 3.12.10, this repository)

| Suite | Passed | Failed | Skipped | Notes |
|---|---:|---:|---:|---|
| DocPrep | 439 | 0 | 4 | The 4 skips need the private v1 reference (below). |
| Studio | 25 | 0 | 0 | The 21 original tests plus 4 for the monorepo-portable configuration (`tests/test_config_paths.py`). |
| Nass | 731 | **19** | 5 | The 19 failures are pre-existing and identical in the original project (below). |

**Baselines from the original projects:**
- DocPrep: 443 passed. Here, 439 passed and the remaining 4 are the private-reference skips.
- Studio: 21 passed.
- Nass: 731 passed, 19 failed, 5 skipped.

The Nass failure set was compared test by test with the original project's failure list: **no new
failures and none removed.**

### DocPrep skips (private data)

The frozen structure reference `docprep/docprep/eval/references/traffic_report_review_v1.json` was
built from a user-owned PDF, and is not distributed. Its SHA-256 is still recorded in `SHA256SUMS`. The
4 tests that check its frozen hash and content skip, with an explicit reason, when it is absent:
- `test_evaluate.py::test_frozen_reference_is_unchanged`
- `test_evaluate.py::test_frozen_reference_content_and_provenance`
- `test_evaluate.py::test_modified_frozen_reference_is_refused`
- `test_round3.py::test_frozen_v1_reference_is_still_unchanged`

To run them, place the original file at that path. It is git-ignored.

### Nass skips

- 2 skips: the user's API benchmark PDFs are absent.
- 1 skip: a user PDF is absent (oxen).
- 1 skip: the traffic PDFs are absent (e2d batch).
- 1 skip: LibreOffice (`soffice`) is not installed.

### Nass pre-existing failures (19)

None are in the Automatic Extraction path that Studio uses (`lab/core/auto`, `lab/core/hybrid` title
and region rules, `dedup`, `assemble`). They are stale expectations or behaviour drift in the
hosted-API evaluation tooling:

| Test | Observed cause |
|---|---|
| `test_api_documents.py::test_shipped_registry_lists_exactly_the_verified_documents` | Registry lists 4 extra `registered-agenticdoc-visual-*` documents |
| `test_api_oxen.py` (8 tests) | Expected cost 0.252 vs 0.504 (and 0.756 vs 1.512); provider limits changed (timeout 120→300 s, `max_tokens` 2048→4096) |
| `test_api_oxen_streaming.py` (2 tests) | Same provider-limit and identity changes |
| `test_e2d_saved_reuse.py` (2 tests) | Expected `submit`, got `reuse` / `blocked` |
| `test_e2d_staged_budget.py` (4 tests) | Expected `COMPLETE`, got `INCOMPLETE`; staging message count |
| `test_hybrid_layout_regions.py::test_local_layout_source_is_a_placeholder` | `pp_doclayout_local()` now requires `run_dir` |
| `test_lpe_arabic_reorder.py::test_config_is_relative_where_possible_and_round_trips` | `min_arabic_letter_share` default is 0.0; the test expects 0.3. **The owner must decide which value is intended.** |

## Integration checks performed (no OCR, no network)

| Check | Result |
|---|---|
| Studio starts from `studio/` with the shipped config | OK. `GET /`, `/assets/app.js`, `/assets/app.css` → 200; `/api/library` and `/api/jobs` → empty lists; path traversal → 404 |
| Studio readiness without `nass/.env` | Reports Nass not ready: `QARI_SERVICE_URL is not configured` (expected) |
| Readiness with a loopback Qari URL and the local PP-DocLayout model | `ready: true`, with no network request (configuration inspection only) |
| Previously processed document via `existing_documents` (temporary config pointing at a saved scanned run and its latest prepared document) | Library listing, Nass reading view, DocPrep reading view, inline PDF, and byte-identical `prepared-json` download all OK |
| `scripts/check_setup.py` | Core OK; warns about the missing `nass/.env` and Qari URL |
| Imports from the new paths | All three suites import and run from their folders |
| Safety metadata through every stage | Covered by the DocPrep tests `test_rag_safety.py`, `test_margin_leakage.py` and `test_caption_grounding.py`, and by the Nass tests `test_margin_leakage.py` and `test_title_association.py` |

**Not executed in this handoff:**
- **A new end-to-end upload → Nass → Qari → DocPrep run.** The Qari/RunPod service is stopped, and
  starting it requires the owner's authorization.
- **The `health()` connectivity snippet in `docs/RUNPOD_SETUP.md`.** There was no service to call.
- **A clean-machine installation from `requirements.txt` into a new virtual environment.** The tests
  ran with the original, equivalent environment, recorded in `requirements-lock.txt`.

## Regression fixtures for the RAG work

- **Real benchmark outputs:** the English and Arabic traffic reports, the Arabic scanned and the Arabic
  mixed documents. They live on the original machine only, because their source PDFs are private.
- **Without them:** use the synthetic builders in `docprep/tests/docprep/helpers.py` and
  `test_prepare.py`, or process non-sensitive PDFs through Studio.
