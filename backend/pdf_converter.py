# backend/pdf_converter.py

import os
import tempfile
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling_core.types.doc import ImageRefMode

def pdf_to_markdown(pdf_bytes: bytes) -> str:
    """
    Converts PDF bytes to Markdown content in-memory.
    """
    try:
        # Write PDF bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(pdf_bytes)
            tmp_pdf_path = tmp_pdf.name

        doc_converter = DocumentConverter()
        conv_res = doc_converter.convert(Path(tmp_pdf_path))

        # Write markdown to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmp_md:
            md_path = Path(tmp_md.name)
            conv_res.document.save_as_markdown(md_path, image_mode=ImageRefMode.REFERENCED)

        with open(md_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()

        # Clean up temporary files
        os.remove(tmp_pdf_path)
        os.remove(md_path)
        return markdown_content

    except Exception as e:
        if os.path.exists(tmp_pdf_path):
            os.remove(tmp_pdf_path)
        raise e

if __name__ == "__main__":
    # For testing
    pdf_file_path = "C:/path/to/sample.pdf"
    with open(pdf_file_path, "rb") as f:
        pdf_bytes = f.read()
    md = pdf_to_markdown(pdf_bytes)
    print(md)
