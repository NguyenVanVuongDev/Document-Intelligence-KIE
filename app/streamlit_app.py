import base64
import os

import requests
import streamlit as st

API_URL = os.getenv("KIE_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Receipt Intelligence", layout="wide")
st.title("Receipt Intelligence")

input_column, detection_column, output_column = st.columns([1.05, 1.15, 0.9], gap="large")

with input_column:
    st.subheader("Input & preprocessing")
    uploaded_file = st.file_uploader("Upload a JPEG or PNG receipt", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        original_bytes = uploaded_file.getvalue()
        st.image(original_bytes, caption="Original", use_container_width=True)

with output_column:
    st.subheader("Structured output")
    analyze_clicked = st.button("Analyze receipt", type="primary", disabled=uploaded_file is None)

if analyze_clicked and uploaded_file:
    try:
        response = requests.post(
            f"{API_URL.rstrip('/')}/analyze",
            files={"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
            timeout=240,
        )
        response.raise_for_status()
        result = response.json()
        st.session_state["analysis_result"] = result
    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")

result = st.session_state.get("analysis_result")
if result:
    with input_column:
        processed_image = base64.b64decode(result["images"]["preprocessed_jpeg_base64"])
        st.image(processed_image, caption="Deskewed and enhanced", use_container_width=True)

    with detection_column:
        st.subheader(f"OCR & spatial detection ({len(result['boxes'])} boxes)")
        overlay = base64.b64decode(result["images"]["annotated_jpeg_base64"])
        st.image(overlay, caption="Blue: OCR text | Red: likely key-value", use_container_width=True)
        with st.expander("Recognized text"):
            st.text(result["full_text"] or "No text detected.")

    with output_column:
        st.json(result["fields"])
        st.caption("Phase latency")
        timings = result["timings_ms"]
        st.metric("Preprocessing", f"{timings['preprocessing_ms']:.0f} ms")
        st.metric("OCR", f"{timings['ocr_ms']:.0f} ms")
        st.metric("KIE", f"{timings['kie_ms']:.0f} ms")