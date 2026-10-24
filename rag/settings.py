"""Pipeline configuration and validation."""
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    output: Path
    chunk_size: int = 1200
    overlap: int = 200
    page_workers: int = 5
    chunk_workers: int = 8
    embedding_model: str = "text-embedding-3-small"
    render_dpi: int = 120
    extract_entities: bool = True
    entity_model: str = "gpt-4.1-mini"

    def validate(self):
        if not 0 <= self.overlap < self.chunk_size:
            raise ValueError("Require 0 <= overlap < chunk_size")
        if self.page_workers < 1 or self.chunk_workers < 1:
            raise ValueError("Worker counts must be positive")
        if not 72 <= self.render_dpi <= 300:
            raise ValueError("render_dpi must be between 72 and 300")
