"""One adapter per library, each using that library's public text-extraction API.

No OCR and no rendering: text comes from the PDF text layer only. Raw adapters
return exactly what the library produced (no fixing, reordering or normalization).
The processed method `pymupdf_ar_reordered` is separate and clearly labeled.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from lab.local_pdf_extraction import arabic_reorder, tables


@dataclass(frozen=True)
class MethodInfo:
    method_id: str
    label: str
    library: str
    distribution: str
    import_name: str
    api: str
    output_type: str  # "raw" | "processed"
    default_selected: bool = True
    note: str = ""


METHODS: tuple[MethodInfo, ...] = (
    MethodInfo("pymupdf", "PyMuPDF", "PyMuPDF", "pymupdf", "pymupdf", 'Page.get_text("text")', "raw"),
    MethodInfo("pdfplumber", "pdfplumber", "pdfplumber", "pdfplumber", "pdfplumber", "Page.extract_text()", "raw"),
    MethodInfo("pypdf", "pypdf", "pypdf", "pypdf", "pypdf", "PageObject.extract_text()", "raw"),
    MethodInfo("pypdfium2", "pypdfium2", "pypdfium2", "pypdfium2", "pypdfium2", "PdfTextPage.get_text_bounded()", "raw"),
    MethodInfo(
        "pdfminer", "pdfminer.six", "pdfminer.six", "pdfminer.six", "pdfminer",
        "PDFPageInterpreter + TextConverter (LAParams defaults)", "raw", default_selected=False,
        note="Already installed as pdfplumber's dependency; optional fifth library.",
    ),
    MethodInfo(
        "pymupdf_ar_reordered", arabic_reorder.METHOD_LABEL, "PyMuPDF + python-bidi", "pymupdf", "pymupdf",
        'Page.get_text("rawdict") + bidi.algorithm.get_display (+ Page.find_tables() when tables are enabled)',
        "processed",
        note="Not raw library output: conditional per-page Arabic logical-order reconstruction; "
             "optional structured tables saved separately.",
    ),
)
METHOD_BY_ID = {method.method_id: method for method in METHODS}
RAW_METHOD_IDS = tuple(method.method_id for method in METHODS if method.output_type == "raw")
PROCESSED_METHOD_ID = "pymupdf_ar_reordered"


@dataclass
class PageOutput:
    text: str
    extra: dict = field(default_factory=dict)
    files: dict[str, bytes] = field(default_factory=dict)  # side files, e.g. page_NNN_tables.json


class Session(Protocol):
    page_count: int

    def extract(self, page_number: int) -> PageOutput: ...

    def close(self) -> None: ...


def library_version(method: MethodInfo) -> str | None:
    try:
        return importlib.metadata.version(method.distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def availability(method: MethodInfo) -> dict:
    version = library_version(method)
    try:
        importlib.import_module(method.import_name)
        if method.method_id == PROCESSED_METHOD_ID:
            importlib.import_module("bidi.algorithm")
            version = f"{version} + python-bidi {importlib.metadata.version('python-bidi')}"
        return {"available": True, "version": version, "error": None}
    except Exception as error:  # noqa: BLE001 - report any import failure
        return {"available": False, "version": version, "error": f"{type(error).__name__}: {error}"}


# ------------------------------------------------------------------ sessions
class PyMuPDFSession:
    def __init__(self, path: Path, settings: dict):
        import pymupdf

        self._doc = pymupdf.open(path)
        self.page_count = len(self._doc)

    def extract(self, page_number: int) -> PageOutput:
        return PageOutput(self._doc[page_number - 1].get_text("text"))

    def close(self) -> None:
        self._doc.close()


class PdfPlumberSession:
    def __init__(self, path: Path, settings: dict):
        import pdfplumber

        self._pdf = pdfplumber.open(path)
        self.page_count = len(self._pdf.pages)

    def extract(self, page_number: int) -> PageOutput:
        page = self._pdf.pages[page_number - 1]
        try:
            text = page.extract_text()
        finally:
            page.close()  # releases pdfplumber's per-page caches
        # extract_text() returns a str; None is recorded as "" and flagged.
        return PageOutput("" if text is None else text, {"returned_none": text is None})

    def close(self) -> None:
        self._pdf.close()


class PyPdfSession:
    def __init__(self, path: Path, settings: dict):
        from pypdf import PdfReader

        self._handle = open(path, "rb")  # noqa: SIM115 - closed in close()
        try:
            self._reader = PdfReader(self._handle)
            self.page_count = len(self._reader.pages)
        except Exception:
            self._handle.close()
            raise

    def extract(self, page_number: int) -> PageOutput:
        return PageOutput(self._reader.pages[page_number - 1].extract_text())

    def close(self) -> None:
        self._handle.close()


class PdfiumSession:
    def __init__(self, path: Path, settings: dict):
        import pypdfium2 as pdfium

        self._pdf = pdfium.PdfDocument(path)
        self.page_count = len(self._pdf)

    def extract(self, page_number: int) -> PageOutput:
        page = self._pdf[page_number - 1]
        try:
            textpage = page.get_textpage()
            try:
                return PageOutput(textpage.get_text_bounded())
            finally:
                textpage.close()
        finally:
            page.close()

    def close(self) -> None:
        self._pdf.close()


class PdfMinerSession:
    def __init__(self, path: Path, settings: dict):
        from pdfminer.layout import LAParams
        from pdfminer.pdfdocument import PDFDocument
        from pdfminer.pdfinterp import PDFResourceManager
        from pdfminer.pdfpage import PDFPage
        from pdfminer.pdfparser import PDFParser

        self._handle = open(path, "rb")  # noqa: SIM115 - closed in close()
        try:
            document = PDFDocument(PDFParser(self._handle))
            self._pages = list(PDFPage.create_pages(document))
        except Exception:
            self._handle.close()
            raise
        self._resources = PDFResourceManager(caching=True)
        self._laparams = LAParams()
        self.page_count = len(self._pages)

    def extract(self, page_number: int) -> PageOutput:
        from pdfminer.converter import TextConverter
        from pdfminer.pdfinterp import PDFPageInterpreter

        output = io.StringIO()
        device = TextConverter(self._resources, output, laparams=self._laparams)
        try:
            PDFPageInterpreter(self._resources, device).process_page(self._pages[page_number - 1])
        finally:
            device.close()
        return PageOutput(output.getvalue())

    def close(self) -> None:
        self._handle.close()


class ArabicReorderedSession:
    """PROCESSED: PyMuPDF text + conditional Arabic logical-order reconstruction."""

    def __init__(self, path: Path, settings: dict):
        import pymupdf

        self._config = arabic_reorder.ReorderConfig.from_dict(settings.get("reorder"))
        # Tables only when the run asked for them (older runs, e.g. on resume, stay as they were).
        self._tables = bool((settings.get("tables") or {}).get("enabled"))
        self._doc = pymupdf.open(path)
        self.page_count = len(self._doc)

    def extract(self, page_number: int) -> PageOutput:
        page = self._doc[page_number - 1]
        raw = page.get_text("text")
        result = arabic_reorder.process_page(page, raw, self._config)
        extra = {"reorder": result.to_record(self._config)}
        files: dict[str, bytes] = {}
        if self._tables:
            doc = tables.page_tables(self._doc, page_number, self._config)
            json_name, md_name = f"page_{page_number:03d}_tables.json", f"page_{page_number:03d}_tables.md"
            files[json_name] = json.dumps(doc, ensure_ascii=False, indent=2).encode("utf-8", errors="surrogatepass")
            files[md_name] = tables.to_markdown(doc).encode("utf-8", errors="surrogatepass")
            # The page text below is unchanged; the regions tell consumers which text is ALSO
            # available as a structured table (use one or the other, never both).
            extra.update(tables_file=json_name, tables_markdown_file=md_name, tables_status=doc["status"],
                         table_regions=tables.table_regions(doc), table_regions_note=tables.RAG_NOTE)
        return PageOutput(result.text, extra, files)

    def close(self) -> None:
        self._doc.close()


SESSIONS: dict[str, Callable[[Path, dict], Session]] = {
    "pymupdf": PyMuPDFSession,
    "pdfplumber": PdfPlumberSession,
    "pypdf": PyPdfSession,
    "pypdfium2": PdfiumSession,
    "pdfminer": PdfMinerSession,
    "pymupdf_ar_reordered": ArabicReorderedSession,
}


def open_session(method_id: str, path: Path, settings: dict, registry: dict | None = None) -> Session:
    return (registry or SESSIONS)[method_id](path, settings)
