import streamlit as st
import requests
import base64
import fitz
import io

st.set_page_config(layout="wide")
st.markdown('''
            <style>
            .fullwidth {
                width: 100%;
                padding-top: 0px;
                padd
            }
            </style>
            ''', unsafe_allow_html=True)

pdf_url = st.query_params.get("url", None)

if pdf_url is None:
    st.error("No PDF URL provided.")
    st.stop()

# Download PDF
response = requests.get(pdf_url)
pdf_stream = io.BytesIO(response.content)
doc = fitz.open(stream=pdf_stream, filetype="pdf")

# Load the PDF
# doc = fitz.open(pdf_url)

# Highlight a word on the first page
page = doc[0]
text = "Abstract"  # Replace with text you want to highlight

for inst in page.search_for(text):
    page.add_highlight_annot(inst)

# Save the modified PDF to memory
pdf_bytes = io.BytesIO()
doc.save(pdf_bytes)
doc.close()

st.sidebar.title("Paper Dots")

# Convert local file to URL-like object for embedding
# with open(pdf_url, "rb") as f:
base64_pdf = base64.b64encode(pdf_bytes.getvalue()).decode('utf-8')

pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#zoom=200" width="100%" height="2000" type="application/pdf"></iframe>'
st.markdown(f'<div class="fullwidth"> {pdf_display} </div>', unsafe_allow_html=True)