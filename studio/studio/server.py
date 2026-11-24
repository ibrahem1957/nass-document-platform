from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import unquote, urlsplit

from .config import ConfigError, ExistingDocument, StudioConfig, load_config
from .jobs import JOB_PATTERN, JobError, JobManager
from .renderers import (
    RenderError,
    load_json,
    nass_markdown,
    nass_reading_view,
    prepared_markdown,
    prepared_reading_view,
)

HOST = "127.0.0.1"
PORT = 8766
LIBRARY_ID = re.compile(r"^[A-Za-z0-9-]{1,64}$")
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/assets/app.css": ("app.css", "text/css; charset=utf-8"),
    "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
DOWNLOAD_NAMES = {
    "nass-json": ("nass_json", "nass_document.json", "application/json; charset=utf-8"),
    "nass-markdown": ("nass_markdown", "nass_extracted_content.md", "text/markdown; charset=utf-8"),
    "prepared-json": ("prepared_json", "prepared_document.json", "application/json; charset=utf-8"),
    "prepared-markdown": ("prepared_markdown", "organized_document.md", "text/markdown; charset=utf-8"),
}


class ServerError(ValueError):
    pass


@dataclass(frozen=True)
class LibraryRecord:
    document_id: str
    label: str
    nass_path: Path
    prepared_path: Path
    pdf_path: Path
    nass: dict[str, Any]
    prepared: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.document_id, "label": self.label,
            "title": self.prepared.get("metadata", {}).get("title"),
            "pages": self.prepared.get("page_count"),
            "nass_elements": len(self.nass.get("elements", [])),
            "prepared_sections": len(self.prepared.get("sections", [])),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _library_record(spec: ExistingDocument) -> LibraryRecord:
    nass = load_json(spec.nass_document, "automatic_extraction_document")
    prepared = load_json(spec.prepared_document, "docprep_prepared_document")
    prepared_validation = load_json(
        spec.prepared_document.with_name("prepared_document_validation.json"),
        "docprep_prepared_document_validation",
    )
    if (prepared_validation.get("summary") or {}).get("error") != 0:
        raise ServerError(f"library prepared document {spec.document_id} has validation errors")
    if nass.get("run_id") != prepared.get("source_run_id"):
        raise ServerError(f"library document {spec.document_id} has mismatched Nass and DocPrep identities")
    manifest_path = spec.nass_document.with_name("execution_manifest.json")
    manifest = load_json(manifest_path, "automatic_extraction_manifest")
    identity = manifest.get("document") or {}
    if identity.get("bytes") != spec.source_pdf.stat().st_size or identity.get("sha256") != _sha256(spec.source_pdf):
        raise ServerError(f"library PDF for {spec.document_id} does not match its Nass manifest")
    return LibraryRecord(spec.document_id, spec.label, spec.nass_document, spec.prepared_document,
                         spec.source_pdf, nass, prepared)


class StudioHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], config: StudioConfig, manager: JobManager,
                 static_root: Path | None = None) -> None:
        if address[0] != HOST:
            raise ServerError(f"Studio may bind only to {HOST}")
        self.config = config
        self.manager = manager
        self.static_root = (static_root or Path(__file__).with_name("static")).resolve(strict=True)
        self.library = {item.document_id: _library_record(item) for item in config.existing_documents}
        try:
            self.readiness = manager.readiness()
        except Exception as exc:  # noqa: BLE001
            self.readiness = {"ready": False, "error": f"{type(exc).__name__}: {exc}"}
        super().__init__(address, StudioHandler)


class StudioHandler(BaseHTTPRequestHandler):
    server: StudioHTTPServer

    def end_headers(self) -> None:
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; "
                         "connect-src 'self'; img-src 'self' data:; frame-src 'self'; object-src 'self'; "
                         "base-uri 'none'; form-action 'self'")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        self._dispatch_get(head=False)

    def do_HEAD(self) -> None:
        self._dispatch_get(head=True)

    def do_POST(self) -> None:
        self._dispatch_post()

    def do_PUT(self) -> None:
        self._not_allowed()

    def do_PATCH(self) -> None:
        self._not_allowed()

    def do_DELETE(self) -> None:
        self._not_allowed()

    def _path(self) -> str | None:
        try:
            value = unquote(urlsplit(self.path).path, errors="strict")
        except (UnicodeError, ValueError):
            return None
        if "\x00" in value or "\\" in value or ".." in value.split("/"):
            return None
        return value

    def _dispatch_get(self, *, head: bool) -> None:
        path = self._path()
        if path is None:
            self._error(HTTPStatus.NOT_FOUND, "not found", head=head)
            return
        if path in STATIC:
            self._static(path, head=head)
            return
        if path == "/api/readiness":
            self._json({"studio": {"local_only": True, "multi_user": False},
                        "nass": self.server.readiness}, head=head)
            return
        if path == "/api/jobs":
            self._json({"jobs": self.server.manager.list_jobs()}, head=head)
            return
        if path == "/api/library":
            self._json({"documents": [item.summary() for item in self.server.library.values()]}, head=head)
            return
        parts = path.strip("/").split("/")
        if len(parts) == 3 and parts[:2] == ["api", "jobs"] and JOB_PATTERN.fullmatch(parts[2]):
            try:
                self._json(self.server.manager.status(parts[2]), head=head)
            except JobError as exc:
                self._error(HTTPStatus.NOT_FOUND, str(exc), head=head)
            return
        if len(parts) >= 5 and parts[:2] == ["api", "items"] and parts[2] in {"job", "library"}:
            self._item_get(parts[2], parts[3], parts[4:], head=head)
            return
        self._error(HTTPStatus.NOT_FOUND, "not found", head=head)

    def _dispatch_post(self) -> None:
        path = self._path()
        if path == "/api/jobs":
            self._upload()
            return
        parts = path.strip("/").split("/") if path else []
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and JOB_PATTERN.fullmatch(parts[2]):
            try:
                if parts[3] == "cancel":
                    self._discard_body()
                    self._json(self.server.manager.cancel(parts[2]), status=HTTPStatus.ACCEPTED)
                    return
                if parts[3] == "retry":
                    self._discard_body()
                    self._require_ready()
                    result = self.server.manager.retry(parts[2])
                    self._json(result, status=HTTPStatus.ACCEPTED)
                    return
            except JobError as exc:
                self._error(HTTPStatus.CONFLICT, str(exc))
                return
        self._not_allowed()

    def _upload(self) -> None:
        try:
            length = self._content_length(self.server.config.max_upload_mb * 1024 * 1024 + 1024 * 1024)
            content_type = self.headers.get("Content-Type", "")
            if not content_type.lower().startswith("multipart/form-data"):
                raise JobError("upload must use multipart/form-data")
            body = self.rfile.read(length)
            message = BytesParser(policy=policy.default).parsebytes(
                b"Content-Type: " + content_type.encode("ascii", "strict")
                + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
            )
            pdf_data, filename = None, None
            if not message.is_multipart():
                raise JobError("invalid multipart upload")
            for part in message.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if name == "pdf":
                    pdf_data, filename = part.get_payload(decode=True), part.get_filename()
            if not isinstance(pdf_data, bytes) or not isinstance(filename, str):
                raise JobError("a PDF file is required")
            self._require_ready()
            status = self.server.manager.create(pdf_data, filename)
            self._json(status, status=HTTPStatus.ACCEPTED)
        except (JobError, ValueError, UnicodeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))

    def _require_ready(self) -> None:
        readiness = self.server.readiness
        if not readiness.get("ready"):
            raise JobError("Nass is not ready: " + str(readiness.get("error") or readiness.get("layout")))

    def _item_get(self, source: str, item_id: str, suffix: list[str], *, head: bool) -> None:
        try:
            paths, nass, prepared, pdf = self._item(source, item_id)
        except (JobError, ServerError) as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc), head=head)
            return
        except RenderError as exc:
            self._error(HTTPStatus.CONFLICT, "saved result cannot be loaded: " + str(exc), head=head)
            return
        if suffix == ["nass"]:
            if nass is None:
                self._error(HTTPStatus.CONFLICT, "Nass output is not available", head=head)
            else:
                self._json(nass_reading_view(nass), head=head)
            return
        if suffix == ["docprep"]:
            if prepared is None:
                self._error(HTTPStatus.CONFLICT, "DocPrep output is not available", head=head)
            else:
                self._json(prepared_reading_view(prepared), head=head)
            return
        if suffix == ["pdf"]:
            self._file(pdf, "application/pdf", inline=True, head=head)
            return
        if len(suffix) == 2 and suffix[0] == "download" and suffix[1] in DOWNLOAD_NAMES:
            key, filename, content_type = DOWNLOAD_NAMES[suffix[1]]
            path = paths.get(key)
            if path is not None and path.is_file():
                self._file(path, content_type, filename=filename, head=head)
                return
            if suffix[1] == "nass-markdown" and nass is not None:
                self._bytes(nass_markdown(nass).encode("utf-8"), content_type, filename=filename, head=head)
                return
            if suffix[1] == "prepared-markdown" and prepared is not None:
                self._bytes(prepared_markdown(prepared).encode("utf-8"), content_type, filename=filename, head=head)
                return
            self._error(HTTPStatus.CONFLICT, "requested artifact is not available", head=head)
            return
        self._error(HTTPStatus.NOT_FOUND, "not found", head=head)

    def _item(self, source: str, item_id: str) -> tuple[dict[str, Path], dict | None, dict | None, Path]:
        if source == "library":
            if not LIBRARY_ID.fullmatch(item_id) or item_id not in self.server.library:
                raise ServerError("unknown library document")
            record = self.server.library[item_id]
            return ({
                "nass_json": record.nass_path, "prepared_json": record.prepared_path,
            }, record.nass, record.prepared, record.pdf_path)
        paths = self.server.manager.result_paths(item_id)
        nass = load_json(paths["nass_json"], "automatic_extraction_document") if paths["nass_json"].is_file() else None
        prepared = load_json(paths["prepared_json"], "docprep_prepared_document") if paths["prepared_json"].is_file() else None
        if not paths["pdf"].is_file():
            raise JobError("job PDF is unavailable")
        return paths, nass, prepared, paths["pdf"]

    def _static(self, request_path: str, *, head: bool) -> None:
        filename, content_type = STATIC[request_path]
        path = (self.server.static_root / filename).resolve(strict=True)
        if path.parent != self.server.static_root:
            self._error(HTTPStatus.NOT_FOUND, "not found", head=head)
            return
        self._bytes(path.read_bytes(), content_type, head=head)

    def _file(self, path: Path, content_type: str, *, filename: str | None = None,
              inline: bool = False, head: bool = False) -> None:
        size = path.stat().st_size
        try:
            byte_range = self._range(self.headers.get("Range"), size)
        except ValueError:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start, end = byte_range if byte_range else (0, size - 1)
        self.send_response(HTTPStatus.PARTIAL_CONTENT if byte_range else HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if byte_range:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if inline:
            self.send_header("Content-Disposition", 'inline; filename="source.pdf"')
        elif filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        if not head:
            with path.open("rb") as stream:
                stream.seek(start)
                self._copy(stream, end - start + 1)

    @staticmethod
    def _range(value: str | None, size: int) -> tuple[int, int] | None:
        if value is None:
            return None
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
        if not match:
            raise ValueError
        first, last = match.groups()
        if not first:
            length = int(last or 0)
            if length <= 0:
                raise ValueError
            return max(0, size - length), size - 1
        start, end = int(first), int(last) if last else size - 1
        if start >= size or end < start:
            raise ValueError
        return start, min(end, size - 1)

    def _copy(self, stream: BinaryIO, remaining: int) -> None:
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            self.wfile.write(chunk)
            remaining -= len(chunk)

    def _content_length(self, maximum: int) -> int:
        value = self.headers.get("Content-Length")
        if value is None or not value.isdigit():
            raise JobError("valid Content-Length is required")
        length = int(value)
        if length <= 0 or length > maximum:
            raise JobError("request is empty or exceeds the upload limit")
        return length

    def _json_body(self) -> dict[str, Any]:
        length = self._content_length(64 * 1024)
        try:
            value = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise JobError("invalid JSON request") from exc
        if not isinstance(value, dict):
            raise JobError("JSON request must be an object")
        return value

    def _discard_body(self) -> None:
        value = self.headers.get("Content-Length", "0")
        if value.isdigit() and 0 < int(value) <= 64 * 1024:
            self.rfile.read(int(value))
        elif value != "0":
            self.close_connection = True

    def _not_allowed(self) -> None:
        self._discard_body()
        self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
        self.send_header("Allow", "GET, HEAD, POST")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _json(self, value: object, *, status: HTTPStatus = HTTPStatus.OK, head: bool = False) -> None:
        data = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._bytes(data, "application/json; charset=utf-8", status=status, head=head)

    def _error(self, status: HTTPStatus, message: str, *, head: bool = False) -> None:
        self._json({"error": message}, status=status, head=head)

    def _bytes(self, data: bytes, content_type: str, *, status: HTTPStatus = HTTPStatus.OK,
               filename: str | None = None, head: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        if not head:
            self.wfile.write(data)


def create_server(config: StudioConfig, manager: JobManager | None = None, port: int = PORT,
                  static_root: Path | None = None) -> StudioHTTPServer:
    if not 0 <= port <= 65535:
        raise ServerError("port must be between 0 and 65535")
    return StudioHTTPServer((HOST, port), config, manager or JobManager(config), static_root)


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent.parent
    value = argparse.ArgumentParser(description="Run the local Nass-to-DocPrep processing studio")
    value.add_argument("--config", type=Path, default=root / "config" / "studio.json")
    value.add_argument("--port", type=int, default=PORT)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config = load_config(args.config)
        server = create_server(config, port=args.port)
    except (ConfigError, ServerError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    host, port = server.server_address
    print(f"Document Processing Studio: http://{host}:{port}")
    print("Local development application; single-user and not a production multi-tenant service.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Studio.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
