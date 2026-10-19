"""Page-local character chunks with exact offsets and overlap."""

class TextChunker:
    def __init__(self, size=1200, overlap=200):
        if not 0 <= overlap < size:
            raise ValueError("Require 0 <= overlap < size")
        self.size = size
        self.overlap = overlap

    def split(self, text):
        """Return exact character spans; chunks never cross a page boundary."""
        size, overlap = self.size, self.overlap
        if not 0 <= overlap < size:
            raise ValueError("Require 0 <= overlap < size")
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if text[start:end].strip():
                chunks.append({"start": start, "end": end, "text": text[start:end]})
            if end == len(text):
                break
            start = end - overlap
        return chunks
