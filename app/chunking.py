def chunk_text(text: str, size: int = 50, overlap: int = 10) -> list[str]:
    """Split `text` into chunks of words.

    Rules (see README for examples):
    - Words are separated by whitespace (use str.split()).
    - Each chunk has at most `size` words, joined with a single space.
    - Consecutive chunks share exactly `overlap` words.
    - Stop when a chunk reaches the last word. No extra trailing chunk.
    - Empty or whitespace-only text returns [].
    - Raise ValueError if size <= 0, overlap < 0, or overlap >= size.
    """
    if size <= 0:
        raise ValueError("size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += size - overlap

    return chunks
