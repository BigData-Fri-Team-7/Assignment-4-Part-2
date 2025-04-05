# backend/recursive_chunking.py

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_markdown(markdown_content: str, chunk_size: int = 300, chunk_overlap: int = 50) -> list:
    """
    Splits the given Markdown content into chunks using RecursiveCharacterTextSplitter.
    Returns a list of chunk strings.
    """
    doc = Document(page_content=markdown_content)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "## ", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents([doc])
    return [chunk.page_content for chunk in chunks]

if __name__ == "__main__":
    sample_md = "This is a sample text.\n\nIt has multiple paragraphs.\n\nAnd some more text."
    result = chunk_markdown(sample_md)
    for chunk in result:
        print(chunk)
