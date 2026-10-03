"""SQL checks on the knowledge base after ingestion.

Each check is a query whose answer should be zero rows, or a comparison that
should hold. They catch the failures that make retrieval quietly worse
without raising anything: an episode that ingested with no chunks, a chunk
missing its embedding, a chunk no keyword search can ever match.

    python -m app.validation --expected-episodes 303

Exits non-zero if any check fails; see `app.validation.checks`.
"""
