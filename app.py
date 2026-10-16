import io
import pandas as pd
import streamlit as st
from src.pdf_reader import extract_pdf
from src.validator import validate_tables
from src.excel_exporter import make_workbook

st.set_page_config(page_title="IncuxAI Document Agent", page_icon="📄", layout="wide")
st.title("PDF → Excel AI Agent")
st.caption("Extract tables from digital or scanned PDFs, review warnings, and export structured Excel.")

with st.sidebar:
    st.header("Processing options")
    use_ocr = st.checkbox("Enable OCR fallback for scanned pages", value=True)
    max_pages = st.number_input("Maximum pages to process", min_value=1, max_value=50, value=50, step=1)
    st.info("For best results, upload a clear PDF with tables. OCR results should be reviewed before use.")

uploaded = st.file_uploader("Upload a PDF (up to 50 pages)", type=["pdf"])
if uploaded:
    raw = uploaded.getvalue()
    if not raw.startswith(b"%PDF"):
        st.error("This file does not appear to be a valid PDF.")
        st.stop()

    if st.button("Extract tables", type="primary", use_container_width=True):
        with st.spinner("Reading pages, extracting tables, and checking data..."):
            try:
                tables, page_log = extract_pdf(raw, use_ocr=use_ocr, max_pages=int(max_pages))
                warnings = validate_tables(tables)
                st.session_state["tables"] = tables
                st.session_state["page_log"] = page_log
                st.session_state["warnings"] = warnings
                st.session_state["filename"] = uploaded.name
            except Exception as exc:
                st.error(f"Processing failed: {exc}")
                st.stop()

if "tables" in st.session_state:
    tables = st.session_state["tables"]
    page_log = st.session_state["page_log"]
    warnings = st.session_state["warnings"]
    st.subheader("Processing summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pages processed", len(page_log))
    c2.metric("Tables found", len(tables))
    c3.metric("Rows extracted", sum(len(t["data"]) for t in tables))
    c4.metric("Warnings", len(warnings))

    with st.expander("Page-by-page processing log", expanded=False):
        st.dataframe(pd.DataFrame(page_log), use_container_width=True, hide_index=True)

    if not tables:
        st.warning("No tables were detected. Try a clearer PDF or check that OCR is enabled for scanned pages.")
    else:
        st.subheader("Extracted tables")
        for i, table in enumerate(tables):
            with st.expander(f"Table {i+1} — {len(table['data'])} rows × {len(table['columns'])} columns", expanded=(i == 0)):
                st.caption(f"Source pages: {', '.join(map(str, table['pages']))} · Method: {table['method']}")
                df = pd.DataFrame(table["data"], columns=table["columns"])
                edited = st.data_editor(df, key=f"table_editor_{i}", use_container_width=True, num_rows="dynamic")
                table["columns"] = [str(c) for c in edited.columns]
                table["data"] = edited.fillna("").astype(str).values.tolist()

        st.subheader("Validation warnings")
        if warnings:
            st.warning("Review these warnings before using the spreadsheet.")
            st.dataframe(pd.DataFrame(warnings), use_container_width=True, hide_index=True)
        else:
            st.success("No basic structural warnings were detected. This does not guarantee perfect extraction.")

        workbook = make_workbook(tables, warnings, page_log)
        st.download_button(
            "Download Excel workbook",
            data=workbook,
            file_name="extracted_tables.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True,
        )
        st.caption("The workbook contains one sheet per extracted table plus Validation and Processing Log sheets.")
