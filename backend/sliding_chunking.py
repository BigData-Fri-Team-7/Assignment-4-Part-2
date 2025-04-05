#backend/sliding_chunking.py

def sliding_chunk_markdown(markdown_content: str, chunk_size: int = 300, chunk_overlap: int = 50) -> list:
    """
    Splits the given Markdown content into chunks using a fixed-size sliding window approach.
    Each chunk will have up to `chunk_size` characters, and consecutive chunks overlap by `chunk_overlap` characters.
    
    Parameters:
      - markdown_content (str): The input text.
      - chunk_size (int): Maximum number of characters in each chunk.
      - chunk_overlap (int): Number of overlapping characters between consecutive chunks.
      
    Returns:
      - List[str]: A list of text chunks.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")
    
    chunks = []
    step = chunk_size - chunk_overlap
    content_length = len(markdown_content)
    
    for start in range(0, content_length, step):
        chunk = markdown_content[start: start + chunk_size]
        chunks.append(chunk)
    
    return chunks

if __name__ == "__main__":
    sample_md = (
        "This is a sample text that demonstrates fixed-size sliding window chunking. "
        "It contains multiple sentences and paragraphs so that the overlapping windows "
        "can capture some of the context between segments. "
        "Each chunk is created by taking a fixed number of characters with an overlap between them. "
        "This method is simple and predictable."
    )
    result = sliding_chunk_markdown(sample_md, chunk_size=100, chunk_overlap=20)
    for i, chunk in enumerate(result, 1):
        print(f"--- Chunk {i} ---")
        print(chunk)
