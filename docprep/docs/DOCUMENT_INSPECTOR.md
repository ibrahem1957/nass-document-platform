# DocPrep Document Inspector

The Document Inspector is a local, read-only view of validated Component 4 output. It compares the
manifest-verified source PDF with the exact prepared-document content. It performs no extraction,
OCR, correction, model call, indexing, or RAG operation.

## Launch on Windows

From PowerShell in the project directory:

```powershell
python -m docprep.inspector --config config\inspector_documents.json
```

Open `http://127.0.0.1:8765` in a browser. The command prints the same URL when the server is ready.
Use `Ctrl+C` in PowerShell to stop it. The inspector never opens a browser itself.

The implementation uses only Python's standard library and local HTML, CSS, and JavaScript. No
additional package is required. The server is hard-bound to `127.0.0.1` and cannot be configured to
listen on a network interface.

## Configuring other prepared documents

`config/inspector_documents.json` is an explicit allowlist. Each entry supplies:

- a safe local inspector ID;
- a display label;
- the exact path to `prepared_document.json`;
- the exact Nass source-run directory.

The inspector accepts additive Component 4 schemas `1.0.0` and `1.1.0`, requires a sibling successful
`prepared_document_validation.json`, validates critical ID/reference namespaces, and verifies the
Nass run identity. It locates a PDF only by matching a PDF in that explicitly allowed Nass directory
against the byte length and SHA-256 recorded in `execution_manifest.json`. A missing or ambiguous
match disables only the preview; structured inspection remains available.

## Views

- Original PDF with page navigation and native-renderer zoom.
- Original elements in reading order, with verbatim text, provenance, structure, relationships, and
  quality references.
- Actual section hierarchy and section-scoped content.
- Relationships with navigable original and virtual endpoints.
- Quality findings with severity and review filters. Confidence and status remain visible.
- Figures, tables, and ordered logical multi-page table parts.
- Exact read-only JSON.
- Local navigation search by element ID, original text, page, or section.

Selecting an element synchronizes the PDF page. The inspector does not draw geometry overlays in the
native PDF frame. In particular, `page_level_only` and unavailable geometry never produce a fake
precise box.

## Security and integrity

- Only `GET` and `HEAD` are supported; mutation methods return `405`.
- Static resources and document/PDF routes are fixed allowlists. There is no filesystem browser.
- Path traversal is rejected.
- Document-derived values are inserted with DOM `textContent`, not executable HTML.
- A restrictive Content Security Policy allows only same-origin local resources.
- Responses use `no-store`; the inspector creates no source-file cache.
- Source files are opened read-only. Review status cannot be edited.

The UI is an aid for human inspection. Automated tests and displayed counts do not constitute human
acceptance of document quality.
