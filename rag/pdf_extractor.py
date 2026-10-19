"""Independent PDF readers for concurrent page extraction."""
class PDFExtractor:
    @staticmethod
    def page_count(pdf):
        from pypdf import PdfReader
        return len(PdfReader(pdf).pages)

    @staticmethod
    def extract_page(pdf, index):
        # Each worker owns its reader; mutable PDF reader state is never shared.
        from pypdf import PdfReader
        return PdfReader(pdf).pages[index].extract_text() or ""
