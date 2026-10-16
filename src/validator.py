import re


def validate_tables(tables):
    warnings = []
    for ti, table in enumerate(tables, start=1):
        cols = table.get("columns", [])
        rows = table.get("data", [])
        pages = table.get("pages", [])
        if not cols:
            warnings.append({"table": ti, "severity": "error", "location": "table", "issue": "No column headers found."})
            continue
        if len(set(cols)) != len(cols):
            warnings.append({"table": ti, "severity": "warning", "location": "header", "issue": "Duplicate column names detected."})
        for ri, row in enumerate(rows, start=1):
            if len(row) != len(cols):
                warnings.append({"table": ti, "severity": "error", "location": f"row {ri}", "issue": f"Expected {len(cols)} cells, found {len(row)}."})
            if not any(str(v).strip() for v in row):
                warnings.append({"table": ti, "severity": "warning", "location": f"row {ri}", "issue": "Completely empty row."})
            if [str(v).strip().lower() for v in row] == [str(v).strip().lower() for v in cols]:
                warnings.append({"table": ti, "severity": "warning", "location": f"row {ri}", "issue": "Repeated header appears inside table."})
        if pages and len(pages) > 1:
            warnings.append({"table": ti, "severity": "info", "location": "pages", "issue": f"Table merged across pages: {', '.join(map(str, pages))}."})
        if "ocr" in table.get("method", "").lower():
            warnings.append({"table": ti, "severity": "warning", "location": "extraction", "issue": "OCR/layout heuristic used; manually verify values and column alignment."})
        # Conservative arithmetic check: only inspect columns whose header clearly suggests a total.
        total_cols = [i for i, c in enumerate(cols) if re.search(r"\b(total|amount due|grand total)\b", str(c), re.I)]
        if total_cols:
            warnings.append({"table": ti, "severity": "info", "location": "numeric checks", "issue": "Total-like column found; arithmetic reconciliation is not automatic unless the source defines the formula."})
    return warnings
