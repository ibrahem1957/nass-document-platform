"""Local, read-only browser inspector for validated prepared documents.

The server deliberately has no mutation endpoints and serves PDFs only when a
file inside an explicitly configured Nass run matches the run manifest's hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import unquote, urlsplit

HOST = "127.0.0.1"
DEFAULT_PORT = 8765
EXPECTED_SCHEMA_VERSION = "1.1.0"
SUPPORTED_SCHEMA_VERSIONS = {"1.0.0", EXPECTED_SCHEMA_VERSION}
EXPECTED_KIND = "docprep_prepared_document"
ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/assets/app.css": ("app.css", "text/css; charset=utf-8"),
    "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


class InspectorError(ValueError):
    """Raised when configured inspector data is unsafe or incompatible."""


@dataclass(frozen=True)
class DocumentRecord:
    """An immutable, validated document exposed by the inspector."""

    document_id: str
    label: str
    prepared_path: Path
    nass_dir: Path
    prepared: dict[str, Any]
    pdf_path: Path | None
    pdf_status: str

    def summary(self) -> dict[str, Any]:
        metadata = self.prepared["metadata"]
        return {
            "id": self.document_id,
            "label": self.label,
            "title": metadata.get("title"),
            "source_filename": metadata.get("source_filename"),
            "page_count": self.prepared["page_count"],
            "element_count": len(self.prepared["elements"]),
            "section_count": len(self.prepared["sections"]),
            "relationship_count": len(self.prepared["relationships"]),
            "quality_finding_count": len(self.prepared["quality"]["findings"]),
            "pdf_available": self.pdf_path is not None,
            "pdf_status": self.pdf_status,
        }


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InspectorError(f"cannot read {label}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise InspectorError(f"invalid JSON in {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise InspectorError(f"{label} must contain a JSON object")
    return value


def _unique_ids(items: list[Any], field: str, label: str) -> set[str]:
    values: list[str] = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get(field), str) or not item[field]:
            raise InspectorError(f"{label} contains an invalid {field}")
        values.append(item[field])
    if len(values) != len(set(values)):
        raise InspectorError(f"{label} contains duplicate {field} values")
    return set(values)


def validate_prepared_document(prepared: dict[str, Any], validation: dict[str, Any]) -> None:
    """Validate the compatibility and critical references needed by the UI."""
    if prepared.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise InspectorError(
            f"incompatible prepared schema_version {prepared.get('schema_version')!r}; "
            f"supported versions are {sorted(SUPPORTED_SCHEMA_VERSIONS)!r}"
        )
    if prepared.get("kind") != EXPECTED_KIND:
        raise InspectorError(
            f"incompatible prepared kind {prepared.get('kind')!r}; expected {EXPECTED_KIND!r}"
        )
    required_lists = ("pages", "sections", "elements", "lists", "visuals", "logical_tables", "nodes",
                      "relationships")
    for field in required_lists:
        if not isinstance(prepared.get(field), list):
            raise InspectorError(f"prepared document lacks required list {field!r}")
    quality = prepared.get("quality")
    if not isinstance(quality, dict) or not isinstance(quality.get("findings"), list):
        raise InspectorError("prepared document lacks quality findings")
    if validation.get("kind") != "docprep_prepared_document_validation":
        raise InspectorError("prepared validation record has an incompatible kind")
    summary = validation.get("summary")
    if not isinstance(summary, dict) or summary.get("error") != 0:
        raise InspectorError("prepared document did not pass Component 4 validation")

    page_count = prepared.get("page_count")
    if not isinstance(page_count, int) or page_count < 1:
        raise InspectorError("prepared document has an invalid page_count")
    element_ids = _unique_ids(prepared["elements"], "element_id", "elements")
    section_ids = _unique_ids(prepared["sections"], "section_id", "sections")
    _unique_ids(prepared["relationships"], "rel_id", "relationships")
    _unique_ids(quality["findings"], "finding_id", "quality findings")
    node_ids = _unique_ids(prepared["nodes"], "node_id", "nodes")

    for element in prepared["elements"]:
        page = element.get("page")
        if not isinstance(page, int) or not 1 <= page <= page_count:
            raise InspectorError(f"element {element['element_id']} has an invalid page")
        unresolved = set(element.get("section_ids", [])) - section_ids
        if unresolved:
            raise InspectorError(f"element {element['element_id']} references unknown sections")
    for section in prepared["sections"]:
        references = set(section.get("children", []))
        if section.get("parent") is not None:
            references.add(section["parent"])
        if references - section_ids:
            raise InspectorError(f"section {section['section_id']} has unresolved hierarchy references")
        if set(section.get("element_ids_in_reading_order", [])) - element_ids:
            raise InspectorError(f"section {section['section_id']} references unknown elements")
    for relationship in prepared["relationships"]:
        endpoints = {relationship.get("source_id"), relationship.get("target_id")}
        if None in endpoints or endpoints - node_ids:
            raise InspectorError(f"relationship {relationship['rel_id']} has unresolved endpoints")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_verified_pdf(nass_dir: Path, source_run_id: str) -> tuple[Path | None, str]:
    """Resolve a PDF by manifest identity, never by an unverified guessed name."""
    manifest_path = nass_dir / "execution_manifest.json"
    if not manifest_path.is_file():
        return None, "Nass execution manifest is missing"
    manifest = _read_json_object(manifest_path, "Nass execution manifest")
    if manifest.get("run_id") != source_run_id:
        raise InspectorError("Nass source run does not match the prepared document")
    document = manifest.get("document")
    if not isinstance(document, dict):
        return None, "Nass manifest has no source document identity"
    expected_hash = document.get("sha256")
    expected_bytes = document.get("bytes")
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
        return None, "Nass manifest has no valid PDF SHA-256"
    candidates = []
    try:
        paths = tuple(nass_dir.glob("*.pdf"))
    except OSError:
        return None, "Nass source directory cannot be inspected"
    for candidate in paths:
        try:
            if (
                candidate.is_file()
                and (not isinstance(expected_bytes, int) or candidate.stat().st_size == expected_bytes)
                and _sha256(candidate).lower() == expected_hash.lower()
            ):
                candidates.append(candidate.resolve(strict=True))
        except OSError:
            continue
    if not candidates:
        return None, "No local PDF matches the source identity recorded by Nass"
    if len(candidates) > 1:
        return None, "Multiple PDFs match the source identity; preview is disabled"
    return candidates[0], "Verified against Nass manifest SHA-256"


def load_document(document_id: str, label: str, prepared_path: Path, nass_dir: Path) -> DocumentRecord:
    if not ID_PATTERN.fullmatch(document_id):
        raise InspectorError(f"unsafe document id {document_id!r}")
    prepared_path = prepared_path.resolve(strict=True)
    nass_dir = nass_dir.resolve(strict=True)
    if prepared_path.name != "prepared_document.json" or not prepared_path.is_file():
        raise InspectorError("configured prepared path must identify prepared_document.json")
    if not nass_dir.is_dir():
        raise InspectorError("configured Nass source must be a directory")
    prepared = _read_json_object(prepared_path, f"prepared document {document_id}")
    validation_path = prepared_path.with_name("prepared_document_validation.json")
    validation = _read_json_object(validation_path, f"prepared validation {document_id}")
    validate_prepared_document(prepared, validation)
    pdf_path, pdf_status = resolve_verified_pdf(nass_dir, prepared["source_run_id"])
    return DocumentRecord(
        document_id=document_id,
        label=label,
        prepared_path=prepared_path,
        nass_dir=nass_dir,
        prepared=prepared,
        pdf_path=pdf_path,
        pdf_status=pdf_status,
    )


def load_registry(config_path: Path | str) -> dict[str, DocumentRecord]:
    config = _read_json_object(Path(config_path), "inspector configuration")
    if config.get("schema_version") != 1 or config.get("kind") != "docprep_inspector_documents":
        raise InspectorError("inspector configuration has an incompatible contract")
    documents = config.get("documents")
    if not isinstance(documents, list) or not documents:
        raise InspectorError("inspector configuration must list at least one document")
    registry: dict[str, DocumentRecord] = {}
    for spec in documents:
        if not isinstance(spec, dict):
            raise InspectorError("invalid document entry in inspector configuration")
        document_id = spec.get("id")
        label = spec.get("label")
        if not isinstance(document_id, str) or not isinstance(label, str) or not label:
            raise InspectorError("every configured document needs a safe id and label")
        if document_id in registry:
            raise InspectorError(f"duplicate configured document id {document_id!r}")
        prepared_path = spec.get("prepared_path")
        nass_dir = spec.get("nass_dir")
        if not isinstance(prepared_path, str) or not isinstance(nass_dir, str):
            raise InspectorError(f"document {document_id!r} has invalid paths")
        registry[document_id] = load_document(
            document_id, label, Path(prepared_path), Path(nass_dir),
        )
    return registry


class InspectorHTTPServer(ThreadingHTTPServer):
    """HTTP server carrying only an immutable document registry."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], registry: dict[str, DocumentRecord], static_root: Path):
        if address[0] != HOST:
            raise InspectorError(f"inspector may bind only to {HOST}")
        self.registry = registry
        self.static_root = static_root.resolve(strict=True)
        super().__init__(address, InspectorRequestHandler)


class InspectorRequestHandler(BaseHTTPRequestHandler):
    server: InspectorHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        # Logs contain only method/status/path, never source text or filesystem paths.
        super().log_message(format, *args)

    def end_headers(self) -> None:
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; "
                         "img-src 'self' data:; connect-src 'self'; object-src 'self'; frame-src 'self'; "
                         "base-uri 'none'; form-action 'none'")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        self._dispatch(head=False)

    def do_HEAD(self) -> None:
        self._dispatch(head=True)

    def do_POST(self) -> None:
        self._method_not_allowed()

    def do_PUT(self) -> None:
        self._method_not_allowed()

    def do_PATCH(self) -> None:
        self._method_not_allowed()

    def do_DELETE(self) -> None:
        self._method_not_allowed()

    def _method_not_allowed(self) -> None:
        # Consume a small request body before replying. On Windows, closing a
        # keep-alive connection with unread bytes can turn the intended 405
        # into a client-visible connection reset.
        length_text = self.headers.get("Content-Length", "0")
        if length_text.isdigit() and 0 < int(length_text) <= 1024 * 1024:
            self.rfile.read(int(length_text))
        elif length_text != "0":
            self.close_connection = True
        self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
        self.send_header("Allow", "GET, HEAD")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _dispatch(self, *, head: bool) -> None:
        try:
            path = unquote(urlsplit(self.path).path, errors="strict")
        except (UnicodeError, ValueError):
            self._json_error(HTTPStatus.BAD_REQUEST, "invalid request path", head=head)
            return
        if ".." in path.split("/") or "\\" in path or "\x00" in path:
            self._json_error(HTTPStatus.NOT_FOUND, "not found", head=head)
            return
        if path in STATIC_FILES:
            self._serve_static(path, head=head)
            return
        if path == "/api/documents":
            self._send_json({"documents": [record.summary() for record in self.server.registry.values()]}, head=head)
            return
        if path == "/api/health":
            self._send_json({"status": "ok", "read_only": True}, head=head)
            return
        parts = path.strip("/").split("/")
        if len(parts) in (3, 4) and parts[:2] == ["api", "documents"]:
            record = self.server.registry.get(parts[2])
            if record is None:
                self._json_error(HTTPStatus.NOT_FOUND, "unknown document", head=head)
                return
            if len(parts) == 3:
                self._send_json(
                    {"summary": record.summary(), "document": record.prepared}, head=head,
                )
                return
            if parts[3] == "pdf":
                self._serve_pdf(record, head=head)
                return
        self._json_error(HTTPStatus.NOT_FOUND, "not found", head=head)

    def _serve_static(self, request_path: str, *, head: bool) -> None:
        filename, content_type = STATIC_FILES[request_path]
        candidate = (self.server.static_root / filename).resolve(strict=True)
        if candidate.parent != self.server.static_root:
            self._json_error(HTTPStatus.NOT_FOUND, "not found", head=head)
            return
        data = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if not head:
            self.wfile.write(data)

    def _serve_pdf(self, record: DocumentRecord, *, head: bool) -> None:
        if record.pdf_path is None:
            self._json_error(HTTPStatus.NOT_FOUND, record.pdf_status, head=head)
            return
        try:
            size = record.pdf_path.stat().st_size
            byte_range = self._parse_range(self.headers.get("Range"), size)
        except (OSError, ValueError):
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size if 'size' in locals() else 0}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start, end = byte_range if byte_range is not None else (0, size - 1)
        self.send_response(HTTPStatus.PARTIAL_CONTENT if byte_range is not None else HTTPStatus.OK)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Disposition", 'inline; filename="source.pdf"')
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if byte_range is not None:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not head:
            with record.pdf_path.open("rb") as stream:
                stream.seek(start)
                self._copy_bytes(stream, end - start + 1)

    @staticmethod
    def _parse_range(value: str | None, size: int) -> tuple[int, int] | None:
        if value is None:
            return None
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
        if match is None or size <= 0:
            raise ValueError("invalid range")
        first, last = match.groups()
        if not first:
            if not last or int(last) <= 0:
                raise ValueError("invalid suffix range")
            length = min(int(last), size)
            return size - length, size - 1
        start = int(first)
        end = int(last) if last else size - 1
        if start >= size or end < start:
            raise ValueError("unsatisfiable range")
        return start, min(end, size - 1)

    def _copy_bytes(self, stream: BinaryIO, remaining: int) -> None:
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            self.wfile.write(chunk)
            remaining -= len(chunk)

    def _send_json(self, value: object, *, status: HTTPStatus = HTTPStatus.OK, head: bool = False) -> None:
        data = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if not head:
            self.wfile.write(data)

    def _json_error(self, status: HTTPStatus, message: str, *, head: bool) -> None:
        self._send_json({"error": message}, status=status, head=head)


def create_server(
    registry: dict[str, DocumentRecord], port: int = DEFAULT_PORT, static_root: Path | None = None,
) -> InspectorHTTPServer:
    if not 0 <= port <= 65535:
        raise InspectorError("port must be between 0 and 65535")
    root = static_root or Path(__file__).with_name("inspector_static")
    return InspectorHTTPServer((HOST, port), registry, root)


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Serve the local read-only DocPrep Document Inspector")
    parser.add_argument(
        "--config", type=Path, default=project_root / "config" / "inspector_documents.json",
        help="explicit allowlist of prepared documents and matching Nass runs",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"localhost port (default: {DEFAULT_PORT})")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        registry = load_registry(args.config)
        server = create_server(registry, args.port)
    except (InspectorError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    host, port = server.server_address
    print(f"DocPrep Document Inspector: http://{host}:{port}")
    print(f"Loaded {len(registry)} validated document(s). Read-only; press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping inspector.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
