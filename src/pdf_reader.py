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
                tables = _digital_tables(page)
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
                    cleaned_rows = _tidy(rows)
                    normalized = None

                    # If this page starts with a record rather than a header,
                    # reuse the prior page's header when the column count fits.
                    # Keep every row as data in this branch, including row 1.
                    if (
                        last_columns
                        and cleaned_rows
                        and _looks_like_record_row(cleaned_rows[0])
                        and max(map(len, cleaned_rows), default=0) == len(last_columns)
                    ):
                        width = len(last_columns)
                        data_rows = [r[:width] + [""] * max(0, width - len(r)) for r in cleaned_rows]
                        normalized = {"columns": list(last_columns), "data": data_rows}
                    else:
                        normalized = normalize_table(cleaned_rows)
                except Exception as exc:
                    entry["note"] = f"Normalization failed: {str(exc)[:180]}"
                    continue

                if not normalized:
                    entry["note"] = (
                        "Table was detected but could not be normalized."
                    )
                    continue

                columns = normalized["columns"]
                data = normalized["data"]
                # Update carry-forward headers only when the first row looks
                # like a real header, not a record identifier.
                if cleaned_rows and not _looks_like_record_row(cleaned_rows[0]):
                    last_columns = list(columns)

                page_tables.append({
                    "page": page_no,
                    "table_no": table_no,
                    "columns": columns,
                    "data": data,
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
