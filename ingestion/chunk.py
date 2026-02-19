"""Text chunking utilities."""
from typing import Iterator
import re

def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    min_chunk_size: int = 50
) -> Iterator[str]:
    """Split text into overlapping chunks."""
    text = re.sub(r"\s+", " ", text.strip())
    
    if len(text) <= chunk_size:
        if len(text) >= min_chunk_size:
            yield text
        return
    
    start = 0
    while start < len(text):
        end = start + chunk_size
        
        if end < len(text):
            search_start = max(start + chunk_size - 100, start)
            search_end = min(start + chunk_size + 50, len(text))
            search_region = text[search_start:search_end]
            
            for pattern in [r"\.\s", r"\n", r";\s"]:
                matches = list(re.finditer(pattern, search_region))
                if matches:
                    last_match = matches[-1]
                    end = search_start + last_match.end()
                    break
        
        chunk = text[start:end].strip()
        if len(chunk) >= min_chunk_size:
            yield chunk
        
        start = end - overlap
        if start <= 0:
            start = end
