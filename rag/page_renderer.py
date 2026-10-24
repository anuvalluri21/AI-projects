"""Render PDF pages in isolated processes; PyMuPDF is not thread-safe."""
import asyncio
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context


def render_pdf_page(pdf, index, destination, dpi):
    import pymupdf
    with pymupdf.open(pdf) as document:
        page = document[index]
        scale = min(dpi / 72, 4096 / max(page.rect.width, page.rect.height))
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
        temporary = str(destination) + ".tmp"
        pixmap.save(temporary, output="png")
        from pathlib import Path
        Path(temporary).replace(destination)
        return {"width": pixmap.width, "height": pixmap.height}


class PageRenderer:
    def __init__(self):
        self.pool = None

    async def render(self, pdf, index, destination, dpi):
        if self.pool is None:
            self.pool = ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn"))
        return await asyncio.get_running_loop().run_in_executor(
            self.pool, render_pdf_page, str(pdf), index, str(destination), dpi
        )

    async def close(self):
        if self.pool is not None:
            await asyncio.to_thread(self.pool.shutdown, wait=True, cancel_futures=True)
            self.pool = None
