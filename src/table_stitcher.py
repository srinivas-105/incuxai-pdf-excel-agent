import re
from collections import Counter


def _cell(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _unique_headers(headers):
    result = []
    counts = Counter()
    for i, header in enumerate(headers):
        name = _cell(header) or f"Column_{i+1}"
        counts[name] += 1
        result.append(name if counts[name] == 1 else f"{name}_{counts[name]}")
    return result


def normalize_table(rows):
    rows = [[_cell(c) for c in row] for row in rows if row and any(_cell(c) for c in row)]
    if len(rows) < 2:
        return None
    width = max(len(r) for r in rows)
    # Keep blank cells; pad rows to the maximum observed width.
    rows = [r + [""] * (width - len(r)) for r in rows]
    headers = _unique_headers(rows[0])
    data = rows[1:]
    # Drop rows that exactly repeat the table header.
    data = [r for r in data if [_cell(c).lower() for c in r] != [_cell(c).lower() for c in headers]]
    if not data:
        return None
    return {"columns": headers, "data": data}


def _same_header(a, b):
    if len(a) != len(b):
        return False
    left = [re.sub(r"\W+", "", str(x).lower()) for x in a]
    right = [re.sub(r"\W+", "", str(x).lower()) for x in b]
    return left == right


def stitch_tables(page_tables):
    """Merge adjacent tables only when normalized headers match exactly."""
    output = []
    for item in page_tables:
        match = None
        for existing in reversed(output):
            if item["page"] <= max(existing["pages"]) + 1 and _same_header(existing["columns"], item["columns"]):
                match = existing
                break
        if match:
            rows = item["data"]
            headers_lower = [x.lower() for x in match["columns"]]
            rows = [r for r in rows if [str(x).strip().lower() for x in r] != headers_lower]
            match["data"].extend(rows)
            match["pages"].append(item["page"])
            if item["method"] not in match["method"]:
                match["method"] += " + " + item["method"]
        else:
            output.append({
                "columns": item["columns"],
                "data": list(item["data"]),
                "pages": [item["page"]],
                "method": item["method"],
            })
    # Remove duplicated page numbers while retaining order.
    for table in output:
        table["pages"] = list(dict.fromkeys(table["pages"]))
    return output
