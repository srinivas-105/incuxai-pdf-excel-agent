# Accuracy and Validation Report

## Status
Initial implementation template. No accuracy percentage is claimed until tests are run against manually verified ground truth.

## Test protocol
For each sample PDF:
1. Manually verify the expected table cells from the source PDF.
2. Run the application and export the workbook.
3. Compare extracted cells against the expected values.
4. Record exact cell-level mismatches, missed rows, duplicated rows, header errors, and OCR substitutions.
5. Report precision/recall or exact-cell accuracy only if the comparison method and denominator are documented.

## Results table

| Sample | Type | Pages | Expected rows | Extracted rows | Cell mismatches | Notes |
|---|---|---:|---:|---:|---:|---|
| Add sample 1 | Digital PDF | — | — | — | — | Not yet tested |
| Add sample 2 | Scanned PDF | — | — | — | — | Not yet tested |
| Add sample 3 | Multi-page table | — | — | — | — | Not yet tested |

## Known limitations
- OCR table reconstruction uses a simple whitespace-based heuristic.
- Multi-page merge requires matching headers and adjacent pages.
- Automatic semantic and arithmetic reconciliation is limited.
