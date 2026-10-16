"""PDF table extraction.

Digital PDFs : pdfplumber (ruled tables first, then text-aligned tables).
Scanned PDFs : PyMuPDF render -> OpenCV (orientation, deskew, ruling-line
               detection) -> Tesseract word boxes -> grid reconstruction.

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
