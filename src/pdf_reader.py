
def _ocr_page(pdf_bytes, page_index, dpi=300):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(
            matrix=fitz.Matrix(dpi / 72, dpi / 72),
            alpha=False
        )
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    finally:
        doc.close()

    # Get each recognized word and its position on the page.
    data = pytesseract.image_to_data(
        image,
        config="--psm 6",
        output_type=pytesseract.Output.DICT
    )

    words = []
    for i, raw in enumerate(data["text"]):
        word = _clean_cell(raw)
        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            confidence = -1

        if not word or confidence < 20:
            continue

        left = int(data["left"][i])
        top = int(data["top"][i])
        width = int(data["width"][i])
        height = int(data["height"][i])

        words.append({
            "text": word,
            "x1": left,
            "x2": left + width,
            "cy": top + height / 2,
            "height": height,
        })

    if not words:
        return []

    # Group words into visual rows by their vertical centers.
    words.sort(key=lambda w: w["cy"])
    lines = []

    for word in words:
        tolerance = max(8, word["height"] * 0.65)
        best_line = None
        best_distance = float("inf")

        for line in lines[-3:]:
            center = sum(w["cy"] for w in line) / len(line)
            distance = abs(word["cy"] - center)

            if distance <= tolerance and distance < best_distance:
                best_line = line
                best_distance = distance

        if best_line is None:
            lines.append([word])
        else:
            best_line.append(word)

    # Split each visual row into approximate cells using horizontal gaps.
    rows = []
    for line in lines:
        line.sort(key=lambda w: w["x1"])
        heights = sorted(w["height"] for w in line)
        median_height = heights[len(heights) // 2]
        gap_threshold = max(14, median_height * 1.5)

        cells = []
        current = line[0]["text"]
        previous_x2 = line[0]["x2"]

        for word in line[1:]:
            gap = word["x1"] - previous_x2

            if gap > gap_threshold:
                cells.append(current.strip())
                current = word["text"]
            else:
                current += " " + word["text"]

            previous_x2 = word["x2"]

        cells.append(current.strip())

        if len(cells) >= 2:
            rows.append(cells)

    # Avoid treating a page with only a few unrelated words as a table.
    if len(rows) < 3:
        return []

    return rows
