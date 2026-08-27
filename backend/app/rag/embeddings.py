"""Embedding model wrapper.

We use fastembed (ONNX runtime) rather than sentence-transformers/torch:

  * ~90MB of dependencies instead of ~2.5GB, which keeps the Docker image and
    the cold-start time reasonable on the CPU-only machines this is built for.
  * Comparable quality from BAAI/bge-small-en-v1.5 at 384 dimensions.
  * No GPU assumptions anywhere in the stack.

bge models are asymmetric: queries must carry an instruction prefix that
passages must not. fastembed's `query_embed`/`passage_embed` apply the correct
prefix for each side, so we never call the generic `embed` here.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterable

from app.config import get_settings
from app.logging import get_logger

log = get_logger(__name__)

_model = None
_model_lock = threading.Lock()


def get_model():
    """Lazily load the ONNX model. Thread-safe; the first call downloads weights."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from fastembed import TextEmbedding

                settings = get_settings()
                log.info("embedding_model_loading", model=settings.embedding_model)
                # cache_dir must point at a mounted volume. fastembed does NOT
                # use the HuggingFace cache path, so without this the ~130MB of
                # weights is re-downloaded on every container restart.
                _model = TextEmbedding(
                    model_name=settings.embedding_model,
                    cache_dir=settings.embedding_cache_dir,
                    # Explicit intra-op threads for the single ONNX session,
                    # rather than fastembed's multiprocessing `parallel=N` mode
                    # -- see embed_passages() for why that mode was rejected.
                    threads=os.cpu_count() or 4,
                )
                log.info("embedding_model_ready", model=settings.embedding_model)
    return _model


def embed_passages(texts: Iterable[str], batch_size: int = 256) -> list[list[float]]:
    """Embed corpus chunks for storage.

    Deliberately NOT using fastembed's `parallel=N` multiprocessing mode: on a
    memory-constrained box each worker loads its own full copy of the ONNX
    session, and measured behaviour here was worker processes being OOM-killed
    mid-batch (degrading throughput below even single-process speed, not just
    failing loudly). Single-process ONNX intra-op threading, driven by
    `threads=` on the model, parallelises within one process's memory budget
    and was measurably faster and stable on this hardware profile.
    """
    model = get_model()
    return [v.tolist() for v in model.passage_embed(list(texts), batch_size=batch_size)]


def embed_query(text: str) -> list[float]:
    """Embed a single user query for search."""
    model = get_model()
    return next(iter(model.query_embed([text]))).tolist()


def warm_up() -> None:
    """Load weights ahead of the first request so it does not eat the latency."""
    try:
        embed_query("warm up")
    except Exception as exc:  # noqa: BLE001 - warm-up must never block startup
        log.warning("embedding_warmup_failed", error=str(exc))
