"""Compatibility entry point; implementation lives in the rag package."""
import asyncio
from rag.cli import main

if __name__ == "__main__":
    asyncio.run(main())
