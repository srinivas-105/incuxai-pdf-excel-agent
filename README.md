# IncuxAI PDF-to-Excel Agent

A Streamlit application that extracts tables from digital PDFs, attempts OCR fallback for scanned pages, merges tables with matching headers across adjacent pages, reports structural warnings, and exports Excel workbooks.

## Features

- PDF upload with configurable page limit (up to 50).
- Digital table extraction using `pdfplumber`.
- OCR fallback using Tesseract for pages where digital table extraction finds no table.
- Conservative table merging when normalized headers match and source pages are adjacent.
- Validation warnings for repeated headers, empty rows, OCR review, and table continuity.
- Editable preview and `.xlsx` export.
- Workbook includes extracted tables, a validation sheet, and processing log.

## Run locally (Windows PowerShell)

Install Python 3.11 or 3.12. Install the Tesseract OCR application separately and make sure `tesseract.exe` is on PATH.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Deploy to Streamlit Community Cloud

1. Push the project to a public GitHub repository.
2. Open Streamlit Community Cloud and create an app from that repository.
3. Set the main file path to `app.py`.
4. Keep `requirements.txt` and `packages.txt` in the repository root.
5. Deploy. `packages.txt` installs the system Tesseract executable on supported Streamlit Cloud builds.

## Test

```powershell
pytest -q
```

## Important limitations

- OCR table reconstruction is heuristic. It splits OCR text on repeated whitespace and cannot reliably reconstruct every complex table. OCR results are explicitly flagged for human review.
- Table stitching only merges adjacent-page tables with exactly matching normalized headers. Tables with changing headers or split rows may require manual review.
- Structural warnings are not a guarantee of semantic correctness. The app does not claim a measured accuracy percentage.
- Large or image-heavy PDFs may take time and memory; test the 50-page limit with representative documents before claiming support for all PDFs.

## Sample deliverables

Put only public, dummy, or otherwise permitted sample documents in `samples/`. For each sample, keep the expected/corrected spreadsheet and document the observed extraction errors in `reports/accuracy_report.md`.

## Dependencies and credits

- Streamlit: application UI
- pdfplumber: digital PDF table extraction
- PyMuPDF: PDF page rendering
- Tesseract OCR and pytesseract: OCR fallback
- pandas and openpyxl: data handling and Excel export
