"""Local PDF snapshots and atomic JSON metadata writes."""
import hashlib
import json

class FileStore:
    @staticmethod
    def write_json(path, value):
        """Replace metadata atomically so failed runs do not leave truncated JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def snapshot_pdf(source, settings):
        data = source.read_bytes()
        checksum = hashlib.sha256(data).hexdigest()
        config = (f"{checksum}:{settings.chunk_size}:{settings.overlap}:{settings.embedding_model}:"
                  f"{settings.render_dpi}:{settings.extract_entities}:{settings.entity_model}:v2")
        document_id = hashlib.sha256(config.encode()).hexdigest()
        folder = settings.output.resolve() / document_id
        folder.mkdir(parents=True, exist_ok=True)
        snapshot = folder / "source.pdf"
        snapshot.write_bytes(data)
        return document_id, checksum, folder, snapshot
