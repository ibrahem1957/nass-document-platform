"""Synthetic PDFs for the Local PDF Extraction Benchmark tests (generated, never real data).

Producers:
  htmlbox_default  PyMuPDF insert_htmlbox, built-in font -> some glyphs have no Unicode
                   mapping (lam extracted as ǁ ƾ Ǆ): the "unmapped glyphs" case
  htmlbox_arial    insert_htmlbox + embedded Arial: logical order, presentation forms,
                   split numbers and flipped brackets in PyMuPDF's raw text
  tounicode_logical  htmlbox_arial with its ToUnicode CMap rewritten to standard letters
                   (lam-alef ligature -> logical "لا"): raw text is already correct
  tounicode_wordlike htmlbox_arial with lam-alef mapped alef-then-lam, like the Word
                   export ("السالمة", "األنماط")
  inserttext_visual  insert_text of a visual-order string (older visual-order producers)
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pymupdf
from bidi.algorithm import get_display

FONT_DIR = Path("C:/Windows/Fonts")
ARIAL = FONT_DIR / "arial.ttf"

CASES: dict[str, str] = {
    "correct": "تقرير السلامة المرورية\nتحليل الأنماط في البيانات",
    "mixed": "جودة البيانات Data Quality مهمة",
    "numbers": "بلغت النسبة 45.5% في عام ٢٠٢٥ و 2024",
    "brackets": "الفئة العمرية (+20 سنة) تقريبًا (تقريبًا)",
    "lamalef": "السلامة الأنماط لا إلا الآن",
    "lamalef_plus_fragment": "السلامة الأنماط لا\nبنسبة 45.5% تقريبًا",
}
TABLE_HTML = ("<table border='1'><tr><td>المنطقة</td><td>العدد</td><td>النسبة</td></tr>"
              "<tr><td>الرياض</td><td>120</td><td>45.5%</td></tr></table>")
ENGLISH = "Road Safety Report 2025 (draft) - Data Quality 45.5%"
PAGE_ORDER = [*CASES, "table", "english"]


def _html(text: str) -> str:
    return "".join(f"<p dir='rtl'>{line}</p>" for line in text.split("\n"))


def htmlbox_pdf(path: Path, *, font: Path | None) -> Path:
    doc = pymupdf.open()
    archive = pymupdf.Archive(str(font.parent)) if font else None
    css = (f"@font-face {{font-family: F; src: url({font.name});}} * {{font-family: F; font-size: 14px;}}"
           if font else "* {font-family: sans-serif; font-size: 14px;}")
    for name in PAGE_ORDER:
        page = doc.new_page(width=595, height=842)
        body = TABLE_HTML if name == "table" else f"<p>{ENGLISH}</p>" if name == "english" else _html(CASES[name])
        page.insert_htmlbox(pymupdf.Rect(50, 50, 545, 800), body, css=css, archive=archive)
    doc.save(path)
    doc.close()
    return path


def inserttext_visual_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    for name in PAGE_ORDER:
        page = doc.new_page(width=595, height=842)
        page.insert_font(fontname="ar", fontfile=str(ARIAL))
        text = ENGLISH if name == "english" else "المنطقة العدد النسبة\nالرياض 120 45.5%" if name == "table" else CASES[name]
        for index, line in enumerate(text.split("\n")):
            page.insert_text((60, 80 + 24 * index), get_display(line, base_dir="R"), fontname="ar", fontsize=14)
    doc.save(path)
    doc.close()
    return path


def _hex_to_text(value: str) -> str:
    return "".join(chr(int(value[i:i + 4], 16)) for i in range(0, len(value), 4))


def _parse_cmap(cmap: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for block in re.findall(r"beginbfchar(.*?)endbfchar", cmap, re.S):
        for source, target in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
            mapping[int(source, 16)] = _hex_to_text(target)
    for block in re.findall(r"beginbfrange(.*?)endbfrange", cmap, re.S):
        for low, high, target in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
            base = _hex_to_text(target)
            for offset, code in enumerate(range(int(low, 16), int(high, 16) + 1)):
                mapping[code] = base[:-1] + chr(ord(base[-1]) + offset)
    return mapping


def _build_cmap(mapping: dict[int, str]) -> str:
    items = sorted(mapping.items())
    lines = ["/CIDInit /ProcSet findresource begin", "12 dict begin", "begincmap",
             "/CIDSystemInfo <</Registry(Adobe)/Ordering(UCS)/Supplement 0>> def",
             "/CMapName /Adobe-Identity-UCS def", "/CMapType 2 def",
             "1 begincodespacerange", "<0000> <FFFF>", "endcodespacerange"]
    for start in range(0, len(items), 100):
        chunk = items[start:start + 100]
        lines.append(f"{len(chunk)} beginbfchar")
        lines += [f"<{code:04X}> <" + "".join(f"{ord(ch):04X}" for ch in text) + ">" for code, text in chunk]
        lines.append("endbfchar")
    lines += ["endcmap", "CMapName currentdict /CMap defineresource pop", "end", "end"]
    return "\n".join(lines)


def rewrite_tounicode(source: Path, target: Path, *, wordlike: bool) -> Path:
    """Map presentation forms to standard letters; optionally lam-alef as alef + lam (Word-like)."""
    doc = pymupdf.open(source)
    done: set[str] = set()
    for page in doc:
        for font in page.get_fonts():
            kind, value = doc.xref_get_key(font[0], "ToUnicode")
            if kind != "xref" or value in done:
                continue
            done.add(value)
            xref = int(value.split()[0])
            mapping = _parse_cmap(doc.xref_stream(xref).decode("latin-1"))
            new = {}
            for code, text in mapping.items():
                norm = unicodedata.normalize("NFKC", text)
                if wordlike and len(norm) == 2 and norm[0] == "ل" and norm[1] in "اأإآ":
                    norm = norm[1] + norm[0]
                new[code] = norm
            doc.update_stream(xref, _build_cmap(new).encode("latin-1"))
    doc.save(target)
    doc.close()
    return target


def all_producers(folder: Path) -> dict[str, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    arial = htmlbox_pdf(folder / "htmlbox_arial.pdf", font=ARIAL)
    return {
        "htmlbox_default": htmlbox_pdf(folder / "htmlbox_default.pdf", font=None),
        "htmlbox_arial": arial,
        "tounicode_logical": rewrite_tounicode(arial, folder / "tounicode_logical.pdf", wordlike=False),
        "tounicode_wordlike": rewrite_tounicode(arial, folder / "tounicode_wordlike.pdf", wordlike=True),
        "inserttext_visual": inserttext_visual_pdf(folder / "inserttext_visual.pdf"),
    }


def small_pdf(path: Path, pages: int = 3) -> Path:
    """Tiny mixed English/Arabic PDF for storage/runner tests."""
    doc = pymupdf.open()
    for number in range(1, pages + 1):
        page = doc.new_page(width=420, height=300)
        page.insert_htmlbox(pymupdf.Rect(30, 30, 390, 270),
                            f"<p>Page {number}: Road safety 45.5%</p><p dir='rtl'>صفحة {number} البيانات</p>")
    doc.save(path)
    doc.close()
    return path
