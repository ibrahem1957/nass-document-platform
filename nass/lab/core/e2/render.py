"""Render PDF pages to images for OCR/VLM input (pixels only, no text layer)."""

from __future__ import annotations

import hashlib
import importlib.metadata
from dataclasses import asdict, dataclass
from pathlib import Path

import pypdfium2 as pdfium

DEFAULT_DPI = 200
MAX_LONG_EDGE = 3000


class RenderError(ValueError):
    pass


@dataclass(frozen=True)
class RenderedPage:
    page: int
    path: str
    sha256: str
    width: int
    height: int
    dpi: float
    renderer: str

    def to_dict(self) -> dict:
        return asdict(self)


def render_pdf_page(
    pdf_path: Path,
    page: int,
    out_dir: Path,
    *,
    dpi: int = DEFAULT_DPI,
    max_long_edge: int = MAX_LONG_EDGE,
) -> RenderedPage:
    """Render one 1-based page to PNG. The DPI is reduced only to respect the cap."""
    if dpi < 36 or dpi > 600:
        raise RenderError("DPI must be between 36 and 600")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        if page < 1 or page > len(pdf):
            raise RenderError(f"Page {page} is out of range (1-{len(pdf)})")
        pdf_page = pdf[page - 1]
        try:
            width_pt, height_pt = pdf_page.get_size()
            scale = dpi / 72
            longest = max(width_pt, height_pt) * scale
            if longest > max_long_edge:
                # pdfium rounds the pixel size up, so leave one pixel of headroom.
                scale *= (max_long_edge - 1) / longest
            bitmap = pdf_page.render(scale=scale, draw_annots=True)
            try:
                image = bitmap.to_pil().convert("RGB")
            finally:
                bitmap.close()
        finally:
            pdf_page.close()
    finally:
        pdf.close()
    effective_dpi = round(scale * 72, 2)
    target = out_dir / f"p{page:04d}_{int(round(effective_dpi))}dpi.png"
    image.save(target, "PNG", optimize=False)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return RenderedPage(
        page=page,
        path=str(target),
        sha256=digest,
        width=image.width,
        height=image.height,
        dpi=effective_dpi,
        renderer=f"pypdfium2 {importlib.metadata.version('pypdfium2')}",
    )
