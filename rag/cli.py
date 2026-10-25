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
    parser.add_argument("--embedding-provider", choices=("openai", "pinecone"), default=os.getenv("EMBEDDING_PROVIDER", "openai"))
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL"))
    parser.add_argument("--embedding-dimension", type=int, default=os.getenv("EMBEDDING_DIMENSION") or None)
    parser.add_argument("--render-dpi", type=int, default=120)
    parser.add_argument("--entity-model", default="gpt-4.1-mini")
    parser.add_argument("--skip-entities", action="store_true", help="Skip Trustcall page knowledge extraction")
    default_schema = Path(__file__).with_name("graph_schema.json")
    if not default_schema.exists():
        default_schema = Path(__file__).resolve().parent / "rag" / "graph_schema.json"
    parser.add_argument("--schema-config", type=Path, default=default_schema if default_schema.exists() else None,
                        help="JSON file defining node types, shared hubs, and dedup rules")
    parser.add_argument("--files-only", action="store_true", help="Skip Neo4j and embeddings")
    args = parser.parse_args()
    if not args.pdf.is_file():
        parser.error("PDF file does not exist")
    model = args.embedding_model or ("llama-text-embed-v2" if args.embedding_provider == "pinecone" else "text-embedding-3-small")
    dimension = args.embedding_dimension
    if dimension is None and model == "llama-text-embed-v2":
        dimension = 1024
    settings = Settings(args.output, args.chunk_size, args.overlap, args.page_workers, args.chunk_workers,
                        model, args.render_dpi, not args.skip_entities, args.entity_model, args.schema_config,
                        args.embedding_provider, dimension)
    settings.validate()
    stores = None
    if not args.files_only:
        required = ["NEO4J_URI", "NEO4J_PASSWORD", "PINECONE_API_KEY", "PINECONE_INDEX_HOST"]
        if settings.embedding_provider == "openai" or settings.extract_entities:
            required.append("OPENAI_API_KEY")
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
