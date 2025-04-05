# frontend/app.py
import streamlit as st
import requests
from dotenv import load_dotenv
import os
import boto3

load_dotenv()  # Load environment variables if necessary

BACKEND_BASE = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{BACKEND_BASE}/upload_pdf/"
CHAT_ENDPOINT = f"{BACKEND_BASE}/chat/"

st.title("📄 PDF & LLM Chat Application")

# Let the user choose between Nvidia Reports and Other PDFs
pdf_source = st.radio("Select PDF Source", options=["Nvidia Reports", "Other PDFs"], index=0)

if pdf_source == "Nvidia Reports":
    st.header("Select Nvidia Report")
    # Choose year (subfolder names)
    nvidia_years = ["2021", "2022", "2023", "2024", "2025"]
    selected_year = st.selectbox("Select Year", options=nvidia_years)
    
    # List available PDFs from S3 in the selected year folder
    s3_bucket = "bigdatassignment5"
    s3_prefix = f"NVIDIA_DATA/PDFs/{selected_year}/"
    s3_client = boto3.client("s3")
    response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)
    if "Contents" in response:
        pdf_files = [obj["Key"].split("/")[-1] for obj in response["Contents"] if obj["Key"].endswith(".pdf")]
    else:
        pdf_files = []
    
    if not pdf_files:
        st.warning("No PDFs found for the selected year.")
    selected_pdf = st.selectbox("Select PDF", options=pdf_files) if pdf_files else None
else:
    st.header("Upload Your PDF")
    uploaded_file = st.file_uploader("Select a PDF to process", type=["pdf"])

# Conversion method, DB choice and chunking strategy
method = st.selectbox("Select PDF Conversion Method", options=["docling", "mistral"])
# Added "Manual" option to the DB choice dropdown
db_choice = st.selectbox("Select Vector DB", options=["Pinecone", "ChromaDB", "Manual"])
chunk_strategy = st.selectbox("Select Chunking Strategy", options=["recursive", "semantic", "sliding"])

if st.button("Process PDF"):
    with st.spinner("Processing PDF..."):
        data = {
            "pdf_source": "nvidia" if pdf_source == "Nvidia Reports" else "other",
            "method": method,
            "chunk_strategy": chunk_strategy,
            "db_choice": db_choice,  # Pass DB choice to backend
        }
        files = None
        if pdf_source == "Nvidia Reports":
            if not selected_pdf:
                st.error("Please select a PDF.")
            else:
                data["pdf_year"] = selected_year
                data["pdf_filename"] = selected_pdf
        else:
            if uploaded_file is None:
                st.warning("Please upload a PDF file.")
            else:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        response = requests.post(UPLOAD_ENDPOINT, files=files, data=data)
        if response.status_code == 200:
            st.success("PDF processed and embeddings upserted successfully!")
            st.session_state.pdf_processed = True
        else:
            st.error("Error processing PDF: " + response.text)
else:
    # Initialize session state variable if not set
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False

st.header("Ask Questions")
if st.session_state.pdf_processed:
    user_query = st.text_input("Enter your question:")
    # Moved LLM selection under Ask Questions
    llm_choice = st.selectbox("Select LLM", options=["gpt-4o", "Gemini Flash Free", "DeepSeek", "Claude-3.5 Haiku"])
    
    if st.button("Send Question"):
        if user_query.strip() == "":
            st.warning("Please enter a question.")
        else:
            payload = {
                "question": user_query,
                "llm_choice": llm_choice,
                "db_choice": db_choice,
            }
            with st.spinner("Generating answer..."):
                chat_resp = requests.post(CHAT_ENDPOINT, json=payload)
                if chat_resp.status_code == 200:
                    result = chat_resp.json()
                    answer = result.get("answer", "No answer received.")
                    token_count = result.get("tokens", None)
                    cost = result.get("cost", None)

                    st.markdown("**Answer:**")
                    st.write(answer)

                    if token_count is not None and cost is not None:
                        st.markdown(f"**Token count:** {token_count}")
                        st.markdown(f"**Estimated cost:** ${cost:.5f}")
                else:
                    st.error("Error from backend: " + chat_resp.text)
else:
    st.info("Please process a PDF first to enable asking questions.")
