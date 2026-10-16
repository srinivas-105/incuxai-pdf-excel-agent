"""PDF table extraction.

Digital PDFs : pdfplumber (ruled tables first, then text-aligned tables).
Scanned PDFs : PyMuPDF render -> OpenCV (illumination flattening, orientation,
               fine deskew, ruling-line / header-tick detection) -> Tesseract
               word boxes -> grid reconstruction -> numeric-column re-OCR.

Public API: extract_pdf(pdf_bytes, use_ocr=True, max_pages=50, lang="eng")
returns (stitched_tables, page_log).
"""
import io
import re
from bisect import bisect_right

import numpy as np
import pdfplumber
import pytesseract
from PIL import Image

try:
    import pymupdf
except ImportError:  # older PyMuPDF
    import fitz as pymupdf

# OpenCV is loaded defensively: if it is broken in the environment, digital
# PDFs still work and scanned pages report a clear error instead of crashing.
try:
    import cv2
    _CV2_ERROR = None
except Exception as _exc:  # ImportError, libGL.so.1 missing, etc.
    cv2 = None
    _CV2_ERROR = _exc

from src.table_stitcher import normalize_table, stitch_tables

OCR_METHOD = "PyMuPDF + OpenCV + Tesseract OCR"
RENDER_DPI = 300
MAX_RENDER_SIDE = 5000
MIN_WORD_CONF = 30

# A cell that is "number-like": digits with optional sign, separators, %, ().
_NUM_RE = re.compile(r"^[\(\-+]?\s*\d[\d,]*(\.\d+)?\s*%?\)?$")
_NUM_LOOSE_RE = re.compile(r"^[\d\s.,()%\-+|}{\]\[]+$")
_NUM_WHITELIST = "0123456789.,-%()"


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------
def _clean_cell(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _tidy(rows):
    """Clean cells, pad to equal width, drop empty rows and empty columns."""
    rows = [[_clean_cell(c) for c in r] for r in rows if r]
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    keep = [j for j in range(width) if any(r[j] for r in rows)]
    return [[r[j] for j in keep] for r in rows]


def _looks_like_table(rows, min_rows=2, min_fill=0.35, max_avg_len=None):
    if len(rows) < min_rows or not rows or len(rows[0]) < 2:
        return False
    cells = [c for r in rows for c in r]
    filled = [c for c in cells if c]
    if not filled or len(filled) / len(cells) < min_fill:
        return False
    if max_avg_len and sum(map(len, filled)) / len(filled) > max_avg_len:
        return False
    return True


# --------------------------------------------------------------------------
# Digital PDFs (selectable text)
# --------------------------------------------------------------------------
def _digital_tables(page):
    strategies = [
        (
            {
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "intersection_tolerance": 5,
            },
            False,
        ),
        (
            {
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "snap_tolerance": 3,
            },
            True,
        ),
    ]

    for settings, text_based in strategies:
        try:
            raw_tables = page.extract_tables(settings) or []
        except Exception:
            continue

        result = []
        for table in raw_tables:
            rows = _tidy(table)
            # Text-based detection easily mistakes paragraphs for tables,
            # so it must pass stricter checks.
            if text_based:
                ok = _looks_like_table(
                    rows, min_rows=3, min_fill=0.5, max_avg_len=60
                )
            else:
                ok = _looks_like_table(rows, min_rows=2, min_fill=0.3)
            if ok:
                result.append(rows)

        if result:
            return result

    return []


# --------------------------------------------------------------------------
# Scanned PDFs: environment checks, rendering, image preparation
# --------------------------------------------------------------------------
def _ensure_ocr_stack():
    if cv2 is None:
        raise RuntimeError(
            f"OpenCV failed to import ({type(_CV2_ERROR).__name__}: "
            f"{str(_CV2_ERROR)[:150]}). Use only 'opencv-python-headless' "
            "in requirements.txt, add 'libgl1' to packages.txt, then "
            "delete and redeploy the app."
        )
    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise RuntimeError(
            "Tesseract binary not found. Add 'tesseract-ocr' and "
            "'tesseract-ocr-eng' to packages.txt."
        ) from exc


def _render_gray(doc, page_index):
    page = doc[page_index]
    scale = RENDER_DPI / 72.0
    longest = max(page.rect.width, page.rect.height) * scale
    if longest > MAX_RENDER_SIDE:
        scale *= MAX_RENDER_SIDE / longest
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(scale, scale),
        colorspace=pymupdf.csGRAY,
        alpha=False,
    )
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    arr = arr.reshape(pix.height, pix.stride)[:, : pix.width]
    return np.ascontiguousarray(arr)


def _fix_orientation(gray):
    """Rotate 90/180/270 degree scans upright using Tesseract OSD."""
    try:
        h, w = gray.shape
        s = 1500 / max(h, w)
        small = (
            cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            if s < 1
            else gray
        )
        osd = pytesseract.image_to_osd(
            Image.fromarray(small),
            config="--psm 0",
            output_type=pytesseract.Output.DICT,
        )
        angle = int(osd.get("rotate", 0)) % 360
        conf = float(osd.get("orientation_conf", 0))
        if angle and conf >= 2.0:
            k = {90: 3, 180: 2, 270: 1}.get(angle, 0)
            if k:
                return np.ascontiguousarray(np.rot90(gray, k))
    except Exception:
        pass
    return gray


def _flatten(gray):
    """Remove uneven background / shaded bands so text and lines stand out."""
    h, w = gray.shape
    s = 0.25
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    bg = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    bg = cv2.medianBlur(bg, 15)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    bg = np.maximum(bg, 1)
    return cv2.divide(gray, bg, scale=255)


def _deskew(gray):
    """Correct skew of up to +-5 degrees with sub-0.1 degree precision.

    Even a 0.1 degree tilt bends a 3500 px ruling line by ~6 px, which is
    enough to break line detection and row grouping, so no dead-zone is used.
    """
    h, w = gray.shape
    s = 1600 / max(h, w)
    small = (
        cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if s < 1
        else gray
    )
    _, bw = cv2.threshold(
        small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    sh, sw = bw.shape

    def score(angle):
        m = cv2.getRotationMatrix2D((sw / 2, sh / 2), float(angle), 1.0)
        rotated = cv2.warpAffine(bw, m, (sw, sh), flags=cv2.INTER_NEAREST)
        profile = rotated.sum(axis=1, dtype=np.float64)
        return float(np.sum(np.diff(profile) ** 2))

    coarse = max(np.arange(-5.0, 5.01, 0.25), key=score)
    best = max(np.arange(coarse - 0.25, coarse + 0.2501, 0.02), key=score)
    if abs(best) < 0.03:
        return gray

    m = cv2.getRotationMatrix2D((w / 2, h / 2), float(best), 1.0)
    return cv2.warpAffine(
        gray,
        m,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=int(np.median(gray)),
    )


def _binary(gray):
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 12
    )


def _line_masks(binary):
    """Return (horizontal_lines, vertical_lines) binary masks."""
    h, w = binary.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 30, 30), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 30, 30)))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk)
    return horiz, vert


# --------------------------------------------------------------------------
# OCR word extraction
# --------------------------------------------------------------------------
def _parse_words(data, dx=0, dy=0, scale=1.0):
    words = []
    for i in range(len(data["text"])):
        text = str(data["text"][i]).strip()
        if not text or re.fullmatch(r"[|_]+", text):
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            continue
        if conf < MIN_WORD_CONF:
            continue
        x = int(data["left"][i] / scale) + dx
        y = int(data["top"][i] / scale) + dy
        w = max(int(data["width"][i] / scale), 0)
        h = max(int(data["height"][i] / scale), 0)
        if w <= 0 or h <= 0:
            continue
        words.append({
            "text": text,
            "x0": x, "x1": x + w,
            "y0": y, "y1": y + h,
            "xc": x + w / 2.0,
            "yc": y + h / 2.0,
            "h": h,
        })
    return words


def _ocr_words(gray, lang):
    image = Image.fromarray(gray)
    words = []
    for psm in (6, 11):
        data = pytesseract.image_to_data(
            image,
            lang=lang,
            config=f"--oem 3 --psm {psm} -c preserve_interword_spaces=1",
            output_type=pytesseract.Output.DICT,
        )
        words = _parse_words(data)
        if len(words) >= 4:
            break
    return words


# --------------------------------------------------------------------------
# Grid reconstruction from word boxes
# --------------------------------------------------------------------------
def _cluster_rows(words, med_h):
    """Group words into visual rows by vertical centre."""
    if not words:
        return []
    tol = max(0.6 * med_h, 6)
    groups = []
    current = []
    mean_y = 0.0
    for wd in sorted(words, key=lambda d: d["yc"]):
        if current and abs(wd["yc"] - mean_y) <= tol:
            current.append(wd)
            mean_y = sum(d["yc"] for d in current) / len(current)
        else:
            if current:
                groups.append(current)
            current = [wd]
            mean_y = wd["yc"]
    if current:
        groups.append(current)
    return [sorted(g, key=lambda d: d["x0"]) for g in groups]


def _segments(group, min_gap):
    count = 1
    for a, b in zip(group, group[1:]):
        if b["x0"] - a["x1"] > min_gap:
            count += 1
    return count


def _column_edges(groups, width, min_gap):
    """Find column separators as vertical gaps free of words.

    The gap must stay free of words in (almost) every row, so word spaces
    inside a cell do not count while narrow gaps between a right-aligned
    number column and the next text column still do.
    """
    if not groups:
        return []
    coverage = np.zeros(width + 2, dtype=np.int32)
    for g in groups:
        mask = np.zeros(width + 2, dtype=np.int32)
        for wd in g:
            mask[max(wd["x0"], 0): min(wd["x1"], width) + 1] = 1
        coverage += mask

    covered = np.flatnonzero(coverage > 0)
    if covered.size == 0:
        return []

    n = len(groups)
    threshold = 0 if n < 8 else int(round(0.03 * n))
    low = coverage <= threshold
    lo, hi = int(covered[0]), int(covered[-1])

    edges = []
    start = None
    for x in range(lo, hi + 1):
        if low[x]:
            if start is None:
                start = x
        elif start is not None:
            if x - start >= min_gap:
                edges.append((start + x - 1) / 2.0)
            start = None
    return edges


def _tick_edges(binary, x, y, bw, bh, med_h, words):
    """Column separators from vertical ticks/lines of any length.

    Scanned ledgers often draw column dividers only in the header row, which
    long-line detection misses. Short vertical strokes taller than a text
    line are accepted here and then validated against the OCR words.
    """
    k = max(int(1.6 * med_h), 24)
    region = binary[y:y + bh, x:x + bw]
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
    vs = cv2.morphologyEx(region, cv2.MORPH_OPEN, vk)
    profile = (vs > 0).sum(axis=0)
    cols = np.flatnonzero(profile >= k * 0.9)
    if cols.size == 0:
        return []

    clusters, start, prev = [], int(cols[0]), int(cols[0])
    for c in cols[1:]:
        c = int(c)
        if c - prev > 4:
            clusters.append((start, prev))
            start = c
        prev = c
    clusters.append((start, prev))

    centres = [x + (a + b) / 2.0 for a, b in clusters]
    pad = max(8, int(0.5 * med_h))
    centres = [c for c in centres if x + pad < c < x + bw - pad]

    # Merge ticks that are closer than half a text line.
    merged = []
    for c in centres:
        if merged and c - merged[-1] < 0.5 * med_h:
            merged[-1] = (merged[-1] + c) / 2.0
        else:
            merged.append(c)

    # Reject separators that cut through many words (e.g. glyph strokes).
    ok = []
    for c in merged:
        cut = sum(1 for d in words if d["x0"] + 2 < c < d["x1"] - 2)
        if cut <= max(1, 0.02 * len(words)):
            ok.append(c)
    return ok


def _assign_cells(groups, edges, med_h):
    """Assign words of each row to columns; header phrases stay together."""
    ncols = len(edges) + 1
    rows = []
    for ri, g in enumerate(groups):
        cells = [[] for _ in range(ncols)]
        if ri == 0:
            # Multi-word header labels ("Unit Price", "Line Total") should
            # land in a single column even if their words straddle an edge.
            phrases, cur = [], [g[0]]
            for a, b in zip(g, g[1:]):
                if b["x0"] - a["x1"] <= 0.55 * med_h:
                    cur.append(b)
                else:
                    phrases.append(cur)
                    cur = [b]
            phrases.append(cur)
            for ph in phrases:
                x0, x1 = ph[0]["x0"], ph[-1]["x1"]
                lo = bisect_right(edges, x0)
                hi = bisect_right(edges, x1)
                if lo == hi:
                    idx = lo
                else:
                    best, idx = -1.0, lo
                    for ci in range(lo, hi + 1):
                        left = edges[ci - 1] if ci > 0 else -1e9
                        right = edges[ci] if ci < len(edges) else 1e9
                        ov = min(x1, right) - max(x0, left)
                        if ov > best:
                            best, idx = ov, ci
                cells[idx].extend(ph)
        else:
            for wd in g:
                lo = bisect_right(edges, wd["x0"] + 2)
                hi = bisect_right(edges, wd["x1"] - 2)
                m = re.fullmatch(r"(\d+)([A-Za-z]+)", wd["text"])
                if lo != hi and m:
                    # OCR fused a right-aligned number with the next column's
                    # text across a column edge ("2m"): split at the edge.
                    e = edges[lo]
                    for part, cidx, a, b in (
                        (m.group(1), lo, wd["x0"], e),
                        (m.group(2), hi, e, wd["x1"]),
                    ):
                        cells[cidx].append(dict(
                            wd, text=part, x0=a, x1=b, xc=(a + b) / 2.0))
                else:
                    cells[bisect_right(edges, wd["xc"])].append(wd)
        rows.append([
            " ".join(d["text"] for d in sorted(c, key=lambda d: d["x0"]))
            for c in cells
        ])
    return rows


def _line_positions(mask, axis, min_cover):
    profile = (mask > 0).sum(axis=axis)
    idx = np.flatnonzero(profile >= min_cover)
    if idx.size == 0:
        return []
    out = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev > 4:
            out.append((start + prev) // 2)
            start = i
        prev = i
    out.append((start + prev) // 2)
    return out


def _merge_wrapped_rows(rows, row_y, hl_inner, ncols):
    """Join continuation lines of wrapped cells (same ruled band, sparse)."""
    if not hl_inner:
        return rows, row_y
    out_rows, out_y, out_band = [], [], []
    limit = max(1, int(0.34 * ncols))
    for r, y in zip(rows, row_y):
        band = bisect_right(hl_inner, y)
        filled = sum(1 for c in r if c)
        if out_rows and out_band[-1] == band and 0 < filled <= limit:
            prev = out_rows[-1]
            out_rows[-1] = [
                (a + " " + b).strip() if b else a for a, b in zip(prev, r)
            ]
            continue
        out_rows.append(r)
        out_y.append(y)
        out_band.append(band)
    return out_rows, out_y


# --------------------------------------------------------------------------
# Numeric column re-OCR (fixes dropped decimal points, "(0)", "3}" ...)
# --------------------------------------------------------------------------
def _is_numeric_column(cells):
    vals = [c for c in cells if c]
    if len(vals) < 3:
        return False
    hits = sum(1 for v in vals if _NUM_LOOSE_RE.match(v))
    return hits / len(vals) >= 0.7


def _clean_number(text):
    t = text.strip().replace(" ", "")
    t = t.strip("|[]{}")
    # "(0)" / "(10" style noise from adjacent ruling lines
    m = re.fullmatch(r"\((\d[\d,]*(?:\.\d+)?)\)?", t)
    if m:
        return m.group(1)
    m = re.fullmatch(r"(\d[\d,]*(?:\.\d+)?)\)", t)
    if m:
        return m.group(1)
    return t



def _restore_dot(image, text, cx0, cx1, yc, med_h):
    """Re-insert a decimal point Tesseract dropped from an integer-looking cell.

    Looks for a small blob sitting on the baseline between two digit blobs.
    Only applied when the digit-blob count matches the OCR digit count.
    """
    if not re.fullmatch(r"\d{2,}", text):
        return text
    y0 = max(int(yc - 0.8 * med_h), 0)
    y1 = min(int(yc + 0.8 * med_h), image.shape[0])
    crop = image[y0:y1, max(cx0, 0):cx1]
    if crop.size == 0:
        return text
    bw = (crop < 190).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    digits, dots = [], []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h >= 0.45 * med_h:
            digits.append((x, y, w, h))
        elif area >= 4 and h <= 0.35 * med_h and w <= 0.4 * med_h:
            dots.append((x, y, w, h))
    digits.sort()
    if len(digits) != len(text) or not dots:
        return text
    base = float(np.median([y + h for _, y, _, h in digits]))
    for dx, dy, dw, dh in dots:
        if abs((dy + dh) - base) > 0.12 * med_h + 2:
            continue  # not on the baseline (comma, noise, line remnant)
        centre = dx + dw / 2.0
        for i in range(len(digits) - 1):
            right_of = digits[i][0] + digits[i][2]
            if right_of - 1 <= centre <= digits[i + 1][0] + 1:
                return text[: i + 1] + "." + text[i + 1:]
    return text


def _refine_numeric(rows, row_y, edges, x0, x1, image, med_h, lang):
    """Re-read numeric columns with a digits-only whitelist at 2x scale."""
    if len(rows) < 3:
        return rows
    ncols = len(edges) + 1
    bounds = [x0] + list(edges) + [x1]
    h = image.shape[0]
    tol = 0.6 * med_h
    top = max(int(min(row_y) - 1.2 * med_h), 0)
    bottom = min(int(max(row_y) + 1.2 * med_h), h)

    for ci in range(ncols):
        body = [r[ci] for r in rows[1:]]
        if not _is_numeric_column(body):
            continue
        cx0 = max(int(bounds[ci]) + 2, 0)
        cx1 = min(int(bounds[ci + 1]) - 2, image.shape[1])
        if cx1 - cx0 < 10 or bottom - top < 10:
            continue
        crop = image[top:bottom, cx0:cx1]
        sc = 2.0
        big = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        big = cv2.copyMakeBorder(big, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        try:
            data = pytesseract.image_to_data(
                Image.fromarray(big),
                lang=lang,
                config=(
                    "--oem 3 --psm 6 "
                    f"-c tessedit_char_whitelist={_NUM_WHITELIST}"
                ),
                output_type=pytesseract.Output.DICT,
            )
        except Exception:
            continue
        words = _parse_words(data, scale=sc)
        for wd in words:  # undo the 20 px border and add crop offsets
            off = int(20 / sc)
            wd["xc"] += cx0 - off
            wd["yc"] += top - off
        if not words:
            continue
        for ri in range(1, len(rows)):
            near = [w for w in words if abs(w["yc"] - row_y[ri]) <= tol]
            if not near:
                continue
            new = "".join(w["text"] for w in sorted(near, key=lambda d: d["xc"]))
            new = _clean_number(new)
            old = rows[ri][ci]
            if not new:
                continue
            if not old or not _NUM_RE.match(_clean_number(old)):
                rows[ri][ci] = new
            elif _clean_number(old) != new and _NUM_RE.match(new):
                rows[ri][ci] = new
            else:
                rows[ri][ci] = _clean_number(old)
            rows[ri][ci] = _restore_dot(
                image, rows[ri][ci], cx0, cx1, row_y[ri], med_h
            )
    return rows


# --------------------------------------------------------------------------
# Table builders
# --------------------------------------------------------------------------
def _edit1(a, b):
    """True when two strings differ by exactly one substitution/insert/delete."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if len(a) > len(b):
        a, b = b, a
    return any(b[:i] + b[i + 1:] == a for i in range(len(b)))


def _fill_blanks(rows, row_y, edges, x0, x1, image, med_h, lang):
    """Re-read empty cells in columns that are otherwise almost always filled."""
    if len(rows) < 4:
        return rows
    bounds = [x0] + list(edges) + [x1]
    for ci in range(len(edges) + 1):
        body = [r[ci] for r in rows[1:]]
        if sum(1 for v in body if v) < 0.8 * len(body):
            continue
        for ri in range(1, len(rows)):
            if rows[ri][ci]:
                continue
            cx0, cx1 = max(int(bounds[ci]) + 2, 0), int(bounds[ci + 1]) - 2
            ya, yb = max(int(row_y[ri] - 0.8 * med_h), 0), int(row_y[ri] + 0.8 * med_h)
            crop = image[ya:yb, cx0:cx1]
            if crop.size == 0:
                continue
            big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            big = cv2.copyMakeBorder(big, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
            try:
                txt = pytesseract.image_to_string(
                    Image.fromarray(big), lang=lang, config="--oem 3 --psm 7"
                )
            except Exception:
                continue
            rows[ri][ci] = _clean_cell(txt)
    return rows


def _snap_categories(rows):
    """Fix one-off OCR slips in short categorical columns (e.g. 'pes' -> 'pcs')."""
    body = rows[1:]
    if len(body) < 15:
        return rows
    for ci in range(len(rows[0])):
        vals = [r[ci] for r in body if r[ci]]
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        if len(counts) > 12 or len(counts) < 2:
            continue
        common = [v for v, c in counts.items() if c >= 2]
        for r in body:
            v = r[ci]
            if v and counts.get(v, 0) <= 1:
                match = [c for c in common if _edit1(v, c)]
                if len(match) == 1:
                    r[ci] = match[0]
    return rows


def _build_grid(ws, x, bw, y, bh, binary, horiz, words_all, width, med_h,
                min_gap, image, lang, use_ticks=True):
    groups = _cluster_rows(ws, med_h)
    if not groups:
        return None

    # Column separators: header ticks / drawn lines first, white-space second.
    edges = []
    if use_ticks:
        edges = _tick_edges(binary, x, y, bw, bh, med_h, ws)
    if len(edges) < 2:
        gap = max(int(0.25 * med_h), 6)
        edges = _column_edges(groups, width, gap)

    rows = _assign_cells(groups, edges, med_h)
    row_y = [float(np.mean([d["yc"] for d in g])) for g in groups]

    hl_all = [v + y for v in _line_positions(horiz[y:y + bh, x:x + bw], 1, 0.6 * bw)]
    rows, row_y = _merge_wrapped_rows(rows, row_y, hl_all[1:-1], len(edges) + 1)

    rows = _refine_numeric(rows, row_y, edges, x, x + bw, image, med_h, lang)
    rows = _fill_blanks(rows, row_y, edges, x, x + bw, image, med_h, lang)
    return _snap_categories(_tidy(rows))


def _ruled_tables(binary, horiz, vert, words, width, med_h, min_gap, image, lang):
    """Tables with drawn ruling lines. Returns (tables, used_boxes)."""
    h, w = horiz.shape
    lines = cv2.bitwise_or(horiz, vert)
    joined = cv2.dilate(lines, np.ones((5, 5), np.uint8), iterations=2)
    found = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[-2]

    tables, used = [], []
    pad = 5

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < w * 0.12 or bh < med_h * 2:
            continue

        hl = _line_positions(horiz[y:y + bh, x:x + bw], 1, 0.6 * bw)
        vl = _line_positions(vert[y:y + bh, x:x + bw], 0, 0.6 * bh)
        if len(hl) < 3 and len(vl) < 3:
            continue

        ws = [
            d for d in words
            if x - pad <= d["xc"] <= x + bw + pad
            and y - pad <= d["yc"] <= y + bh + pad
        ]
        if not ws:
            continue

        rows = _build_grid(ws, x, bw, y, bh, binary, horiz, words, width,
                           med_h, min_gap, image, lang)
        if rows and _looks_like_table(rows, min_rows=2, min_fill=0.25):
            tables.append((y, rows))
            used.append((x - pad, y - pad, x + bw + pad, y + bh + pad))

    return tables, used


def _borderless_tables(words, width, med_h, min_gap, image, lang):
    """Tables without ruling lines, found by aligned columns of words."""
    if not words:
        return []

    groups = _cluster_rows(words, med_h)
    runs, current, pending = [], [], []
    prev_bottom = None

    def flush():
        nonlocal current, pending
        if current:
            runs.append(current)
        current, pending = [], []

    for g in groups:
        top = min(d["y0"] for d in g)
        bottom = max(d["y1"] for d in g)

        if prev_bottom is not None and top - prev_bottom > 2.0 * med_h:
            flush()

        if _segments(g, min_gap) >= 2:
            current.extend(pending)
            pending = []
            current.append(g)
        elif current:
            pending.append(g)
            if len(pending) > 1:
                flush()
        prev_bottom = bottom
    flush()

    tables = []
    for run in runs:
        if len(run) < 3:
            continue
        edges = _column_edges(run, width, max(int(0.6 * med_h), 10))
        rows = _assign_cells(run, edges, med_h)
        row_y = [float(np.mean([d["yc"] for d in g])) for g in run]
        x0 = min(d["x0"] for g in run for d in g) - 5
        x1 = max(d["x1"] for g in run for d in g) + 5
        rows = _refine_numeric(rows, row_y, edges, x0, x1, image, med_h, lang)
        rows = _tidy(rows)
        if _looks_like_table(rows, min_rows=3, min_fill=0.4):
            top = min(d["y0"] for g in run for d in g)
            tables.append((top, rows))
    return tables


def _extract_scanned_page(doc, page_index, lang="eng"):
    _ensure_ocr_stack()

    gray = _render_gray(doc, page_index)
    if np.count_nonzero(gray < 160) < 0.0005 * gray.size:
        return []  # blank page

    gray = _fix_orientation(gray)
    gray = _flatten(gray)
    gray = _deskew(gray)
    height, width = gray.shape

    binary = _binary(gray)
    horiz, vert = _line_masks(binary)

    # Erase long ruling lines (thin mask, so decimal points stay intact).
    line_mask = cv2.dilate(
        cv2.bitwise_or(horiz, vert), np.ones((2, 2), np.uint8), iterations=1
    )
    clean = gray.copy()
    clean[line_mask > 0] = 255

    words = _ocr_words(clean, lang)
    if not words:
        return []

    med_h = float(np.median([d["h"] for d in words]))
    min_gap = max(int(med_h), 15)

    ruled, used = _ruled_tables(
        binary, horiz, vert, words, width, med_h, min_gap, clean, lang
    )

    remaining = [
        d for d in words
        if not any(
            x0 <= d["xc"] <= x1 and y0 <= d["yc"] <= y1
            for x0, y0, x1, y1 in used
        )
    ]
    borderless = _borderless_tables(remaining, width, med_h, min_gap, clean, lang)

    combined = sorted(ruled + borderless, key=lambda t: t[0])
    return [rows for _, rows in combined]


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def extract_pdf(pdf_bytes, use_ocr=True, max_pages=50, lang="eng"):
    """Extract tables from a PDF, log each page, and stitch across pages."""
    page_log = []
    page_tables = []
    ocr_doc = None

    try:
        pdf = pdfplumber.open(io.BytesIO(pdf_bytes))
    except Exception as exc:
        raise ValueError(f"Could not open PDF: {exc}") from exc

    try:
        total_pages = len(pdf.pages)
        if total_pages > max_pages:
            raise ValueError(
                f"PDF has {total_pages} pages; configured limit is {max_pages}."
            )

        for page_index, page in enumerate(pdf.pages):
            page_no = page_index + 1
            method = "pdfplumber"
            note = ""

            try:
                tables = _digital_tables(page) if page.chars else []
            except Exception as exc:
                tables = []
                note = f"pdfplumber failed: {type(exc).__name__}"

            if not tables and use_ocr:
                try:
                    if ocr_doc is None:
                        ocr_doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
                    tables = _extract_scanned_page(ocr_doc, page_index, lang)
                    method = OCR_METHOD
                    note = "" if tables else (
                        "No reliable table was detected. "
                        "Review the scan quality and table layout."
                    )
                except Exception as exc:
                    tables = []
                    method = "OCR failed"
                    note = f"{type(exc).__name__}: {str(exc)[:250]}"
            elif not tables:
                method = "no table detected"
                note = note or "No digital table found; OCR is disabled."

            # "page" is always a string so Streamlit/Arrow can serialise
            # the log even when it contains the "Summary" row.
            entry = {
                "page": str(page_no),
                "method": method,
                "tables": len(tables),
                "note": note,
            }
            page_log.append(entry)

            for table_no, rows in enumerate(tables, start=1):
                try:
                    normalized = normalize_table(rows)
                except Exception as exc:
                    entry["note"] = f"Normalization failed: {str(exc)[:180]}"
                    continue

                if not normalized:
                    entry["note"] = (
                        "Table was detected but could not be normalized."
                    )
                    continue

                page_tables.append({
                    "page": page_no,
                    "table_no": table_no,
                    "columns": normalized["columns"],
                    "data": normalized["data"],
                    "method": method,
                })

            # Release per-page caches (pdfplumber keeps them otherwise).
            try:
                page.flush_cache()
            except Exception:
                pass
    finally:
        pdf.close()
        if ocr_doc is not None:
            ocr_doc.close()

    try:
        stitched = stitch_tables(page_tables)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to combine tables across pages: {exc}"
        ) from exc

    if not stitched:
        page_log.append({
            "page": "Summary",
            "method": "extraction unsuccessful",
            "tables": 0,
            "note": (
                "No usable tables were extracted. Check the OCR dependency, "
                "scan quality, and table layout."
            ),
        })

    return stitched, page_log
