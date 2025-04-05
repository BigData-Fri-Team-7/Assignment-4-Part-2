# backend/main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
import os
import boto3

# Imports
from pdf_converter import pdf_to_markdown
from mistral_converter import pdf_to_markdown_mistral
from recursive_chunking import chunk_markdown as chunk_recursive
from semantic_chunking import chunk_semantic
from sliding_chunking import sliding_chunk_markdown
from pinecone_embeds import upsert_embeddings as upsert_pinecone, query_pinecone
from chromadb_embeds import upsert_embeddings as upsert_chroma, query_chroma
from manual_embeddings import upsert_embeddings as upsert_manual, query_manual
from llm_chat import get_llm_response

load_dotenv()
app = FastAPI()

# S3 configuration for Nvidia PDFs
S3_BUCKET = "bigdatassignment5"
S3_PREFIX = "NVIDIA_DATA/PDFs"
s3_client = boto3.client("s3")

class ChatRequest(BaseModel):
    question: str
    llm_choice: str
    db_choice: str

@app.post("/upload_pdf/")
async def upload_pdf(
    pdf_source: str = Form(...),  # "nvidia" or "other"
    method: str = Form(...),       # 'docling' or 'mistral'
    chunk_strategy: str = Form("recursive"),
    db_choice: str = Form(...),    # Accept DB choice: "Pinecone", "ChromaDB", or "Manual"
    file: UploadFile = File(None), # For "other" PDFs
    pdf_year: str = Form(None),    # For Nvidia PDFs
    pdf_filename: str = Form(None) # For Nvidia PDFs
):
    """
    Upload and process a PDF, either from S3 (Nvidia) or user-uploaded (Other).
    Chunk the text, then upsert embeddings to the selected vector DB.
    """
    try:
        # 1. Retrieve PDF bytes
        if pdf_source.lower() == "nvidia":
            if not pdf_year or not pdf_filename:
                raise HTTPException(
                    status_code=400,
                    detail="pdf_year and pdf_filename must be provided for Nvidia PDFs."
                )
            s3_key = f"{S3_PREFIX}/{pdf_year}/{pdf_filename}"
            s3_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
            pdf_bytes = s3_obj["Body"].read()
            metadata = {"source": "nvidia", "year": pdf_year, "filename": pdf_filename}
        else:
            if file is None:
                raise HTTPException(
                    status_code=400,
                    detail="Please upload a PDF file for Other PDFs."
                )
            pdf_bytes = await file.read()
            metadata = {"source": "other", "filename": file.filename}

        # 2. Convert PDF to markdown
        if method == "mistral":
            markdown = pdf_to_markdown_mistral(pdf_bytes)
        else:
            markdown = pdf_to_markdown(pdf_bytes)

        # 3. Chunking
        if chunk_strategy == "semantic":
            chunks = chunk_semantic(markdown, chunk_size=300)
        elif chunk_strategy == "sliding":
            chunks = sliding_chunk_markdown(markdown, chunk_size=300, chunk_overlap=50)
        else:
            chunks = chunk_recursive(markdown, chunk_size=300, chunk_overlap=50)

        # 4. Upsert embeddings based on the chosen vector DB
        if db_choice.lower() == "pinecone":
            upsert_pinecone(chunks, metadata)
        elif db_choice.lower() == "chromadb":
            upsert_chroma(chunks, metadata)
        elif db_choice.lower() == "manual":
            upsert_manual(chunks, metadata)
        else:
            raise HTTPException(status_code=400, detail="Invalid DB choice provided.")

        return JSONResponse(content={"message": "PDF processed successfully!"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/")
def chat(request: ChatRequest):
    """
    Given a user question, retrieve relevant chunks from the selected vector DB,
    then get an LLM-based answer using the selected model.
    """
    try:
        # Query the chosen vector DB
        if request.db_choice.lower() == "pinecone":
            results = query_pinecone(request.question)
        elif request.db_choice.lower() == "chromadb":
            results = query_chroma(request.question)
        elif request.db_choice.lower() == "manual":
            results = query_manual(request.question)
        else:
            raise HTTPException(status_code=400, detail="Invalid DB choice provided.")

        # Consolidate the text chunks
        chunks = [
            m.get("metadata", {}).get("text", "") or m.get("text", "")
            for m in results.get("matches", [])
        ]
        context = " ".join(chunks)

        if not context:
            raise HTTPException(status_code=404, detail="No relevant chunks found.")

        pdf_data = {"pdf_content": context, "tables": []}
        response = get_llm_response(pdf_data, request.question, request.llm_choice)

        return {
            "answer": response["answer"],
            "tokens": response["tokens"],
            "cost": response["cost"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
