"""Arabic-aware text extraction for digital PDFs (PyMuPDF + bidi).
Fixes: lam-alef ligature split (Word exports), visual->logical order, numbers & Latin inside RTL lines."""
import sys, re, unicodedata, pymupdf
from bidi.algorithm import get_display

ALEFS = set('اأإآ')
# PDF already stores logical bracket code points; stop bidi from mirroring them again
PROTECT = str.maketrans({'(': '¦', ')': '§', '[': '¶', ']': '†'})
RESTORE = str.maketrans({'¦': '(', '§': ')', '¶': '[', '†': ']'})

def clusters_from_page(page):
    out = []  # (x0, x1, yc, visual_text)
    for b in page.get_text('rawdict')['blocks']:
        for l in b.get('lines', []):
            for s in l['spans']:
                ch = s['chars']; i = 0
                while i < len(ch):
                    c = ch[i]; t = c['c']; x0, y0, x1, y1 = c['bbox']
                    # lam-alef ligature: zero-width alef followed by lam  ->  logical "لا"
                    if t in ALEFS and abs(x1 - x0) < 0.2 and i + 1 < len(ch) and ch[i+1]['c'] == 'ل':
                        lx0, _, lx1, _ = ch[i+1]['bbox']
                        logical = 'ل' + t; x0, x1 = min(x0, lx0), max(x1, lx1); i += 2
                    elif t.isspace():
                        logical = ' '; i += 1     # kept for LTR lines; RTL lines rebuild spaces from geometry
                    else:
                        logical = t; i += 1
                    # attach following combining marks (harakat / tanween)
                    while i < len(ch) and unicodedata.category(ch[i]['c']) == 'Mn':
                        logical += ch[i]['c']; i += 1
                    out.append([x0, x1, (y0 + y1) / 2, y1 - y0, logical])
    return out

def page_text(page, ytol=0.45, gap=18):
    cl = clusters_from_page(page)
    cl.sort(key=lambda c: c[2])
    lines = []
    for c in cl:
        if lines and abs(lines[-1][-1][2] - c[2]) < ytol * max(c[3], 1):
            lines[-1].append(c)
        else:
            lines.append([c])
    res = []
    for ln in lines:
        ln.sort(key=lambda c: c[0])           # visual order, left -> right
        if not any(unicodedata.bidirectional(ch) in ('AL', 'R') for c in ln for ch in c[4]):
            res.append(re.sub(r' +', ' ', ''.join(c[4] for c in ln)).strip()); continue
        ln = [c for c in ln if c[4] != ' ']
        if not ln: continue
        segs = [[ln[0]]]
        for c in ln[1:]:
            if c[0] - segs[-1][-1][1] > gap: segs.append([c])   # table cells / big gaps
            else: segs[-1].append(c)
        parts = []
        for sg in segs:
            vis = []
            prev = None
            for c in sg:
                if prev is not None and c[0] - prev[1] > 0.12 * max(c[3], 1): vis.append(' ')
                prev = c
                t = c[4]
                # Arabic clusters must appear reversed in a visual string so bidi flips them back
                if any(unicodedata.bidirectional(ch) in ('AL', 'R') for ch in t):
                    t = ''.join(reversed([t[0]] + ([t[1:]] if len(t)>1 and unicodedata.category(t[1])=='Mn' else list(t[1:]))))
                vis.append(t)
            v = ''.join(vis).translate(PROTECT)
            parts.append(get_display(v, base_dir='R').translate(RESTORE))
        rtl = any(unicodedata.bidirectional(ch) in ('AL', 'R') for c in ln for ch in c[4])
        line = ' | '.join(reversed(parts) if rtl else parts).strip()
        line = unicodedata.normalize('NFKC', line)          # Arabic presentation forms -> standard letters
        # some producers store visual (pre-mirrored) brackets: ")..(" -> "(..)"
        o, c = line.find('('), line.find(')')
        if c != -1 and (o == -1 or c < o) and line.count('(') == line.count(')'):
            line = line.translate(str.maketrans('()', ')('))
        res.append(line)
    return '\n'.join(r for r in res if r)

if __name__ == '__main__':
    d = pymupdf.open(sys.argv[1])
    pages = [int(p) for p in sys.argv[2:]] or range(1, len(d) + 1)
    for p in pages:
        print(f'===== page {p}'); print(page_text(d[p - 1]))
