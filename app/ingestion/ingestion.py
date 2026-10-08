"""Ingestion pipeline: parse → chunk → save locally → embed → index in Qdrant.

Run from the project root:

    python -m app.ingestion.ingestion                      # ingest everything under DATA/
    python -m app.ingestion.ingestion --data-dir DATA/true_data
    python -m app.ingestion.ingestion --dry-run            # parse + chunk + save JSON, no embedding
    python -m app.ingestion.ingestion --force              # re-ingest even unchanged files

Indexed runs use an ingestion ledger (see ledger.py): unchanged files and byte-identical
copies are skipped, and points of files deleted/moved out of --data-dir are pruned.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

import tiktoken
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, SparseEmbeddings
from qdrant_client import QdrantClient, models

from app.config import Settings, get_settings
from app.ingestion.embeddings import SPARSE_VECTOR_NAME, build_embeddings, build_sparse_embeddings
from app.ingestion.chunking import ChunkingConfig, DocumentSplitter
from app.ingestion.ledger import IngestionLedger, file_hash
from app.ingestion.loaders import LoaderError, get_loader
from app.logging import setup_logging
from app.observability import Tracer

logger = logging.getLogger(__name__)

# Payload field (inside LangChain's "metadata" payload) used to find a file's points.
SOURCE_FIELD = "metadata.source"

# Approximate token count for Langfuse usage reporting. The embedding model's own tokenizer
# differs, so this is an estimate (typically within ~10-20% for English text).
_EMBEDDING_ENCODING = tiktoken.get_encoding("cl100k_base")


def build_qdrant_client(settings: Settings) -> QdrantClient:
    """Qdrant server when QDRANT_URL is set, otherwise the embedded on-disk store. Shared with retrieval."""
    if settings.qdrant_url:
        return QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
            timeout=settings.qdrant_timeout,
        )
    # Embedded mode is SQLite-backed; allow use from worker threads (e.g. the API's thread
    # pool). Callers must still serialise access: embedded Qdrant isn't built for concurrency.
    return QdrantClient(path=str(settings.qdrant_path), force_disable_check_same_thread=True)


def embedding_model_id(settings: Settings) -> str:
    """Dense model identifier stored on every point; retrieval checks it matches the query embedder."""
    return f"{settings.embedding_provider}:{settings.embedding_model}"


def sparse_model_id(settings: Settings) -> str | None:
    """BM25 model identifier in hybrid mode, else None."""
    return settings.sparse_model if settings.retrieval_mode == "hybrid" else None


def _embedding_tokens(chunks: list[Document]) -> int:
    return sum(len(_EMBEDDING_ENCODING.encode(c.page_content, disallowed_special=())) for c in chunks)


@dataclass
class ProcessingResult:
    """Outcome of ingesting one file."""

    source: str
    success: bool
    # ingested: (re)indexed; unchanged: same content already indexed; duplicate: identical
    # copy of another indexed file (duplicate_of); failed: see error.
    status: Literal["ingested", "unchanged", "duplicate", "failed"] = "failed"
    source_type: str | None = None
    duplicate_of: str | None = None
    documents: int = 0
    chunks: int = 0
    output_path: str | None = None
    error: str | None = None
    duration_seconds: float = 0.0


@dataclass
class BatchResult:
    """Outcome of ingesting several files."""

    results: list[ProcessingResult] = field(default_factory=list)
    # Sources removed from the index because the file no longer exists.
    pruned: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> list[ProcessingResult]:
        return [r for r in self.results if r.success]

    @property
    def ingested(self) -> list[ProcessingResult]:
        return [r for r in self.results if r.status == "ingested"]

    @property
    def skipped(self) -> list[ProcessingResult]:
        return [r for r in self.results if r.status in ("unchanged", "duplicate")]

    @property
    def failed(self) -> list[ProcessingResult]:
        return [r for r in self.results if not r.success]

    @property
    def total_chunks(self) -> int:
        return sum(r.chunks for r in self.succeeded)


class ChunkStore:
    """Saves parsed chunks as JSON under ``<root>/<source_type>/``.

    The JSON holds chunk text as well as metadata, so a file can be re-embedded
    (e.g. after changing embedding model) without parsing it again.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def save(self, source: Path, source_type: str, chunks: list[Document]) -> Path:
        directory = self.root / source_type
        directory.mkdir(parents=True, exist_ok=True)
        output_path = directory / f"{self._file_key(source)}.json"

        payload = {
            "source": str(source),
            "source_type": source_type,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "chunk_count": len(chunks),
            "chunks": [
                {
                    "chunk_id": chunk.metadata["chunk_id"],
                    "content": chunk.page_content,
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ],
        }
        # Write then rename, so a crash never leaves a half-written file behind.
        tmp_path = output_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        tmp_path.replace(output_path)
        return output_path

    def delete(self, source: Path, source_type: str) -> None:
        (self.root / source_type / f"{self._file_key(source)}.json").unlink(missing_ok=True)

    def load(self, path: Path) -> list[Document]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [Document(page_content=c["content"], metadata=c["metadata"]) for c in payload["chunks"]]

    @staticmethod
    def _file_key(source: Path) -> str:
        # Stem plus a short hash of the full path: same-named files in different folders don't collide.
        digest = uuid.uuid5(uuid.NAMESPACE_URL, str(source.resolve())).hex[:8]
        return f"{source.stem}_{digest}"


class QdrantIndexer:
    """Embeds chunks and upserts them into a Qdrant collection.

    Point IDs are derived from each chunk's deterministic ``chunk_id``, and a
    file's old points are deleted before its new ones are written, so
    re-ingesting a changed file leaves no stale chunks behind.

    Every point records ``embedding_model`` in its metadata, and the indexer refuses
    to write to a collection holding another model's vectors: vectors from different
    models aren't comparable, even when their dimensions happen to match.

    With ``sparse_embeddings`` (hybrid mode) each point also gets a BM25 sparse vector;
    the collection's sparse configuration must match the mode, so hybrid search never
    runs over points that lack keyword vectors.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embeddings: Embeddings,
        batch_size: int = 64,
        embedding_model: str = "unknown",
        sparse_embeddings: SparseEmbeddings | None = None,
        sparse_model: str | None = None,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.batch_size = batch_size
        self.embedding_model = embedding_model
        self.sparse_embeddings = sparse_embeddings
        self.sparse_model = sparse_model if sparse_embeddings is not None else None
        self._ensure_collection()
        self.vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            sparse_vector_name=SPARSE_VECTOR_NAME,
            retrieval_mode=RetrievalMode.HYBRID if sparse_embeddings is not None else RetrievalMode.DENSE,
        )

    @property
    def index_signature(self) -> str:
        """Everything that determines a point's vectors; stored in the ledger to detect changes."""
        return f"{self.embedding_model}+sparse:{self.sparse_model}" if self.sparse_model else self.embedding_model

    def index(self, source: str, chunks: list[Document]) -> None:
        self.delete_source(source)
        if not chunks:
            return
        for chunk in chunks:
            chunk.metadata["embedding_model"] = self.embedding_model
            if self.sparse_model:
                chunk.metadata["sparse_model"] = self.sparse_model
        ids = [str(uuid.UUID(hex=chunk.metadata["chunk_id"])) for chunk in chunks]
        self.vector_store.add_documents(chunks, ids=ids, batch_size=self.batch_size)

    def point_count(self) -> int:
        return self.client.count(self.collection_name, exact=True).count

    def delete_source(self, source: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key=SOURCE_FIELD, match=models.MatchValue(value=source))]
                )
            ),
        )

    def _ensure_collection(self) -> None:
        vector_size = len(self.embeddings.embed_query("dimension probe"))
        if self.client.collection_exists(self.collection_name):
            # Vectors from a different model/dimension can't share a collection.
            info = self.client.get_collection(self.collection_name)
            existing = info.config.params.vectors
            existing_size = existing.size if isinstance(existing, models.VectorParams) else None
            if existing_size is not None and existing_size != vector_size:
                raise ValueError(
                    f"Collection {self.collection_name!r} has {existing_size}-dim vectors but the embedding "
                    f"model produces {vector_size}. Use a new QDRANT_COLLECTION or recreate the collection."
                )
            has_sparse = SPARSE_VECTOR_NAME in (info.config.params.sparse_vectors or {})
            if has_sparse != (self.sparse_embeddings is not None):
                built, wanted = ("hybrid", "dense") if has_sparse else ("dense", "hybrid")
                raise ValueError(
                    f"Collection {self.collection_name!r} was built for {built} search but RETRIEVAL_MODE is "
                    f"{wanted}. Use a new QDRANT_COLLECTION, or set RETRIEVAL_MODE={built}."
                )
            self._check_model()
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            # IDF is computed by Qdrant over the whole collection, completing BM25 scoring.
            sparse_vectors_config=(
                {SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)}
                if self.sparse_embeddings is not None
                else None
            ),
        )
        # Indexed so per-file deletes and source filters at query time stay fast.
        # (Embedded local Qdrant ignores payload indexes and warns; that's harmless.)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Payload indexes have no effect")
            for field_name in (SOURCE_FIELD, "metadata.file_type", "metadata.source_type"):
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
        logger.info(
            "Created Qdrant collection %r (dim=%d, %s)", self.collection_name, vector_size,
            "hybrid: dense + BM25" if self.sparse_embeddings is not None else "dense",
        )

    def _check_model(self) -> None:
        """Fail if the collection already holds points from a different embedding model."""
        points, _ = self.client.scroll(self.collection_name, limit=1, with_payload=True, with_vectors=False)
        if not points:
            return
        metadata = (points[0].payload or {}).get("metadata", {})
        existing_model = metadata.get("embedding_model")
        if existing_model != self.embedding_model:
            raise ValueError(
                f"Collection {self.collection_name!r} holds vectors from "
                f"{existing_model or 'an unrecorded (older) embedding model'}, not {self.embedding_model}. "
                "Use a new QDRANT_COLLECTION or recreate the collection."
            )
        if self.sparse_model and metadata.get("sparse_model") != self.sparse_model:
            raise ValueError(
                f"Collection {self.collection_name!r} holds keyword vectors from "
                f"{metadata.get('sparse_model') or 'an unrecorded model'}, not {self.sparse_model}. "
                "Use a new QDRANT_COLLECTION or recreate the collection."
            )


class DocumentProcessor:
    """Runs the full ingestion pipeline for files and directories.

    Dependencies can be injected (useful for tests or alternative backends);
    otherwise they are built from ``Settings``. With ``enable_indexing=False``
    the pipeline stops after saving chunks locally (no embedding, no Qdrant, no ledger).
    ``force=True`` re-ingests files even when the ledger says they're unchanged.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        splitter: DocumentSplitter | None = None,
        store: ChunkStore | None = None,
        indexer: QdrantIndexer | None = None,
        enable_indexing: bool = True,
        tracer: Tracer | None = None,
        ledger: IngestionLedger | None = None,
        force: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.tracer = tracer or Tracer(self.settings)
        self.splitter = splitter or DocumentSplitter(
            ChunkingConfig(chunk_size=self.settings.chunk_size, chunk_overlap=self.settings.chunk_overlap)
        )
        self.store = store or ChunkStore(self.settings.processed_data_dir)
        self.indexer = (indexer or self._build_indexer()) if enable_indexing else None
        self.force = force
        self.ledger: IngestionLedger | None = None
        if self.indexer is not None:
            self.ledger = ledger or IngestionLedger(self.settings.ledger_path)
            # A new or emptied collection invalidates the ledger: otherwise files would be
            # skipped as "unchanged" while their points are missing.
            if self.indexer.point_count() == 0:
                cleared = self.ledger.clear(self.indexer.collection_name)
                if cleared:
                    logger.info("Collection %r is empty; reset %d ledger entries", self.indexer.collection_name, cleared)

    def __enter__(self) -> DocumentProcessor:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self.tracer.flush()
        if self.ledger is not None:
            self.ledger.close()
        if self.indexer is not None:
            self.indexer.client.close()

    def process_file(self, path: str | Path, source_type: str | None = None) -> ProcessingResult:
        """Ingest one file.

        ``source_type`` (e.g. ``true_data`` / ``noisy_data``) names the
        ``processed_data/`` subfolder and is stored in chunk metadata so
        retrieval can filter on it. Defaults to the file's parent folder name.
        """
        # Absolute path, so the same file always maps to the same Qdrant "source"
        # however it was referenced, and re-ingests replace its old points.
        path = Path(path).resolve()
        source = str(path)
        started = time.perf_counter()
        result = ProcessingResult(source=source, success=False, source_type=source_type or path.parent.name)

        content_hash = file_hash(path) if self.ledger is not None else None
        if content_hash is not None and self._skip_if_known(content_hash, result):
            result.duration_seconds = round(time.perf_counter() - started, 3)
            return result

        index_attempted = False
        # One Langfuse trace per file, with a child observation per pipeline stage.
        with self.tracer.observation(
            "ingest-file",
            as_type="chain",
            input={"source": source, "source_type": result.source_type},
        ) as file_span:
            try:
                with self.tracer.observation("parse", input={"file": path.name}) as span:
                    documents = get_loader(path).load(path)
                    result.documents = len(documents)
                    for document in documents:
                        document.metadata["source_type"] = result.source_type
                    span.update(output={
                        "documents": result.documents,
                        "file_type": documents[0].metadata.get("file_type"),
                        "characters": sum(len(d.page_content) for d in documents),
                    })

                with self.tracer.observation("chunk") as span:
                    chunks = self.splitter.split(documents)
                    result.chunks = len(chunks)
                    span.update(output={"chunks": result.chunks})
                if not chunks:
                    raise LoaderError(f"No chunks produced from {path}")

                with self.tracer.observation("save-local") as span:
                    result.output_path = str(self.store.save(path, result.source_type, chunks))
                    span.update(output={"path": result.output_path})

                if self.indexer is not None:
                    # "embedding" type + model + token usage lets Langfuse show embedding cost.
                    with self.tracer.observation(
                        "embed-index",
                        as_type="embedding",
                        model=self.settings.embedding_model,
                        input={"chunks": len(chunks), "collection": self.indexer.collection_name},
                    ) as span:
                        index_attempted = True
                        self.indexer.index(source, chunks)
                        span.update(usage_details={"input": _embedding_tokens(chunks)})
                    if self.ledger is not None:
                        self.ledger.record(
                            self.indexer.collection_name, source, content_hash, result.source_type,
                            result.chunks, self.indexer.index_signature,
                        )

                result.success = True
                result.status = "ingested"
                logger.info(
                    "Ingested %s: %d documents → %d chunks", source, result.documents, result.chunks,
                    extra={"source": source, "source_type": result.source_type, "chunks": result.chunks},
                )
            except LoaderError as exc:
                # Expected for unreadable/empty files: no traceback needed.
                result.error = f"{type(exc).__name__}: {exc}"
                logger.warning("Skipped %s: %s", source, exc, extra={"source": source, "source_type": result.source_type})
                file_span.update(level="WARNING", status_message=result.error)
            except Exception as exc:
                result.error = f"{type(exc).__name__}: {exc}"
                logger.exception("Failed to ingest %s", source, extra={"source": source, "source_type": result.source_type})
                file_span.update(level="ERROR", status_message=result.error)
                if index_attempted and self.ledger is not None:
                    # Old points were deleted (or partly replaced): forget the file so it's retried.
                    self.ledger.remove(self.indexer.collection_name, source)
            finally:
                result.duration_seconds = round(time.perf_counter() - started, 3)
                file_span.update(output={
                    "success": result.success,
                    "documents": result.documents,
                    "chunks": result.chunks,
                    "duration_seconds": result.duration_seconds,
                })
        return result

    def _skip_if_known(self, content_hash: str, result: ProcessingResult) -> bool:
        """Mark ``result`` as unchanged/duplicate and return True if the file needn't be ingested.

        ``force`` re-embeds unchanged files but still skips copies (it shouldn't create duplicates).
        """
        collection = self.indexer.collection_name
        source = result.source
        entry = self.ledger.get(collection, source)
        unchanged = entry and entry.content_hash == content_hash and entry.embedding_model == self.indexer.index_signature
        if unchanged and not self.force:
            result.success, result.status = True, "unchanged"
            logger.info("Unchanged, skipped: %s", source, extra={"source": source, "status": "unchanged"})
            return True

        # Identical bytes already indexed under another path that still exists: a copy.
        # (If that path is gone, the file was moved: ingest it here and prune the old path.)
        copies = [s for s in self.ledger.sources_with_hash(collection, content_hash) if s != source and Path(s).exists()]
        if copies:
            if entry:  # this path was indexed before (as other content, or under --force): drop its points
                self.indexer.delete_source(source)
                self.ledger.remove(collection, source)
            result.success, result.status, result.duplicate_of = True, "duplicate", copies[0]
            logger.info(
                "Duplicate of %s, skipped: %s", copies[0], source,
                extra={"source": source, "status": "duplicate", "duplicate_of": copies[0]},
            )
            return True
        return False

    def prune(self, directory: Path) -> list[str]:
        """Remove index entries for files under ``directory`` that no longer exist."""
        if self.ledger is None:
            return []
        collection = self.indexer.collection_name
        pruned = []
        for source in self.ledger.sources(collection):
            path = Path(source)
            if not path.is_relative_to(directory) or path.exists():
                continue
            entry = self.ledger.get(collection, source)
            self.indexer.delete_source(source)
            if entry and entry.source_type:
                self.store.delete(path, entry.source_type)
            self.ledger.remove(collection, source)
            pruned.append(source)
            logger.info("Pruned deleted/moved file: %s", source, extra={"source": source, "status": "pruned"})
        return pruned

    def process_files(self, paths: Iterable[str | Path], source_type: str | None = None) -> BatchResult:
        """Ingest each file; one failure doesn't stop the batch."""
        return self._run([(Path(path), source_type) for path in paths])

    def process_directory(self, directory: str | Path, recursive: bool = True, prune: bool = True) -> BatchResult:
        """Ingest every supported file in ``directory``.

        With ``prune`` (and indexing on), files previously ingested from ``directory``
        that no longer exist have their points, ledger entry and chunk JSON removed.

        Each file's source type is its top-level subfolder, so ``DATA/true_data/a.pdf``
        under ``DATA`` is ``true_data``. Files directly in ``directory`` take the
        directory's own name (so pointing at ``DATA/true_data`` also gives ``true_data``).
        """
        directory = Path(directory).resolve()
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        pattern = "**/*" if recursive else "*"
        paths = sorted(p for p in directory.glob(pattern) if p.is_file() and self._is_supported(p))
        batch = self._run([(path, self._source_type(directory, path)) for path in paths])
        if prune:
            batch.pruned = self.prune(directory)
        return batch

    def _run(self, items: list[tuple[Path, str | None]]) -> BatchResult:
        # Per-file traces share a session, so one run can be viewed together in Langfuse
        # without putting thousands of files into a single trace.
        run_id = f"ingestion-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
        mode = "index" if self.indexer is not None else "dry-run"
        batch = BatchResult()
        with self.tracer.attributes(session_id=run_id, tags=["ingestion", mode], metadata={"run_id": run_id}):
            for number, (path, source_type) in enumerate(items, start=1):
                logger.info("[%d/%d] %s", number, len(items), path.name)
                batch.results.append(self.process_file(path, source_type))

            with self.tracer.observation("ingestion-run-summary", input={"files": len(items)}) as span:
                span.update(output={
                    "ingested": len(batch.ingested),
                    "skipped": len(batch.skipped),
                    "failed": len(batch.failed),
                    "chunks": batch.total_chunks,
                    "failed_files": [Path(r.source).name for r in batch.failed],
                })
        if self.tracer.enabled:
            logger.info("Langfuse session: %s", run_id)
        return batch

    @staticmethod
    def _source_type(root: Path, path: Path) -> str:
        parts = path.relative_to(root).parts
        return parts[0] if len(parts) > 1 else root.name

    @staticmethod
    def _is_supported(path: Path) -> bool:
        try:
            get_loader(path)
            return True
        except LoaderError:
            return False

    def _build_indexer(self) -> QdrantIndexer:
        settings = self.settings
        # Build embeddings first: a config error (e.g. missing key) then fails before Qdrant opens
        # and leaves no store/lock behind.
        embeddings = build_embeddings(settings)
        sparse_embeddings = build_sparse_embeddings(settings)
        return QdrantIndexer(
            build_qdrant_client(settings),
            settings.qdrant_collection,
            embeddings,
            settings.embedding_batch_size,
            embedding_model=embedding_model_id(settings),
            sparse_embeddings=sparse_embeddings,
            sparse_model=sparse_model_id(settings),
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG index.")
    parser.add_argument("--data-dir", type=Path, default=Path("DATA"), help="Folder to ingest (default: DATA)")
    parser.add_argument("--no-recursive", action="store_true", help="Only ingest files directly in --data-dir")
    parser.add_argument("--dry-run", action="store_true", help="Parse, chunk and save JSON; skip embedding/indexing")
    parser.add_argument("--collection", help="Qdrant collection name (overrides QDRANT_COLLECTION)")
    parser.add_argument("--force", action="store_true", help="Re-ingest files even if unchanged since the last run")
    parser.add_argument("--no-prune", action="store_true", help="Keep index entries of files no longer in --data-dir")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return parser.parse_args(argv)


def print_summary(batch: BatchResult, elapsed: float) -> None:
    # Structured record for log tools (e.g. CloudWatch Insights); the prints below are for humans.
    unchanged = sum(r.status == "unchanged" for r in batch.results)
    duplicates = sum(r.status == "duplicate" for r in batch.results)
    logger.info(
        "Ingestion finished: %d ingested, %d unchanged, %d duplicates, %d failed, %d pruned, %d chunks",
        len(batch.ingested), unchanged, duplicates, len(batch.failed), len(batch.pruned), batch.total_chunks,
        extra={
            "files": len(batch.results),
            "ingested": len(batch.ingested),
            "unchanged": unchanged,
            "duplicates": duplicates,
            "failed": len(batch.failed),
            "pruned": len(batch.pruned),
            "chunks": batch.total_chunks,
            "duration_seconds": round(elapsed, 1),
            "failed_files": [Path(r.source).name for r in batch.failed],
        },
    )
    print(f"\nProcessed {len(batch.results)} files in {elapsed:.1f}s")
    print(f"  ingested:   {len(batch.ingested)} ({batch.total_chunks} chunks)")
    print(f"  unchanged:  {unchanged}")
    print(f"  duplicates: {duplicates}")
    for result in batch.results:
        if result.status == "duplicate":
            print(f"    - {Path(result.source).name} = {Path(result.duplicate_of).name}")
    print(f"  pruned:     {len(batch.pruned)}")
    print(f"  failed:     {len(batch.failed)}")
    for result in batch.failed:
        print(f"    - {Path(result.source).name}: {result.error}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    log_dir = setup_logging(settings, level="DEBUG" if args.verbose else None)
    if log_dir is not None:
        logger.info("Logging to %s", log_dir.resolve())

    if args.collection:
        settings = settings.model_copy(update={"qdrant_collection": args.collection})

    if not args.data_dir.is_dir():
        logger.error("Data directory not found: %s", args.data_dir.resolve())
        return 2

    started = time.perf_counter()
    try:
        with DocumentProcessor(settings=settings, enable_indexing=not args.dry_run, force=args.force) as processor:
            batch = processor.process_directory(
                args.data_dir, recursive=not args.no_recursive, prune=not args.no_prune
            )
    except ValueError as exc:  # e.g. missing GOOGLE_API_KEY, collection dimension mismatch
        logger.error("%s", exc)
        return 2

    print_summary(batch, time.perf_counter() - started)
    return 0 if not batch.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
