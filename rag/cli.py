"""Command-line entry point for PDF graph ingestion."""
import argparse
import json
import os
from pathlib import Path

from .settings import Settings
from .stores import Stores
from .pipeline import RagPipeline

async def main():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"))
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("pdf_data"))
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--page-workers", type=int, default=5)
    parser.add_argument("--chunk-workers", type=int, default=8)
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--files-only", action="store_true", help="Skip Neo4j and embeddings")
    args = parser.parse_args()
    if not args.pdf.is_file():
        parser.error("PDF file does not exist")
    settings = Settings(args.output, args.chunk_size, args.overlap, args.page_workers, args.chunk_workers, args.embedding_model)
    settings.validate()
    stores = None
    if not args.files_only:
        required = ("OPENAI_API_KEY", "NEO4J_URI", "NEO4J_PASSWORD", "PINECONE_API_KEY", "PINECONE_INDEX_HOST")
        missing = [key for key in required if not os.getenv(key) or os.getenv(key).startswith("your-")]
        if missing:
            parser.error("Missing configuration: " + ", ".join(missing))
        stores = Stores()
    try:
        if stores:
            await stores.open(settings)
        result = await RagPipeline(settings, stores).ingest(args.pdf.resolve())
        print(json.dumps(result, indent=2))
    finally:
        if stores:
            await stores.close()
