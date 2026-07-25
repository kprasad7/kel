"""pgvector (Postgres) adapter for `kel.retrieval.VectorStore`. Requires
`pip install kel[pgvector]` (`psycopg` + `pgvector`).

**`password` is optional by design.** Leaving it unset lets `psycopg`
resolve credentials the way `psql` itself does — `PGPASSWORD` env var, a
`.pgpass` file, peer/trust auth for local sockets — or you can pass a
short-lived token you generated yourself as `password` (e.g. AWS RDS IAM
auth tokens, which are just password-shaped strings with a ~15 min TTL).
None of that requires kel to know anything about IAM; it only needs to not
get in the way by forcing an argument.

Table/column names are developer-controlled construction-time parameters,
not user input — they're interpolated into SQL directly (Postgres doesn't
support parameterizing identifiers), while all actual data values (ids,
text, metadata, embeddings, query text) go through parameterized queries.
Never pass untrusted input as `table_name`.
"""

from __future__ import annotations

import json
from typing import Any

from kel.retrieval.types import Chunk, ScoredChunk


class PgVectorStore:
    def __init__(
        self,
        table_name: str,
        *,
        dsn: str | None = None,
        host: str | None = None,
        port: int = 5432,
        dbname: str | None = None,
        user: str | None = None,
        password: str | None = None,
        dimension: int | None = None,
        connection: Any = None,
    ):
        self.table_name = table_name
        self.dimension = dimension
        self._table_ready = False
        if connection is not None:
            self._conn = connection
        else:
            try:
                import psycopg
                from pgvector.psycopg import register_vector
            except ImportError as exc:
                raise ImportError(
                    "The psycopg and pgvector packages are required to use PgVectorStore. "
                    "Install them with `pip install kel[pgvector]`."
                ) from exc
            if dsn:
                self._conn = psycopg.connect(dsn, autocommit=True)
            else:
                self._conn = psycopg.connect(
                    host=host, port=port, dbname=dbname, user=user, password=password, autocommit=True
                )
            register_vector(self._conn)

    def _ensure_table(self, dimension: int) -> None:
        if self._table_ready:
            return
        dim = self.dimension or dimension
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table_name} "
                f"(id TEXT PRIMARY KEY, text TEXT, metadata JSONB, embedding VECTOR({dim}))"
            )
        self._table_ready = True

    def upsert(self, chunks: list[Chunk]) -> None:
        embedded = [c for c in chunks if c.embedding is not None]
        if not embedded:
            return
        first_embedding = embedded[0].embedding
        assert first_embedding is not None
        self._ensure_table(len(first_embedding))
        with self._conn.cursor() as cur:
            for chunk in embedded:
                cur.execute(
                    f"INSERT INTO {self.table_name} (id, text, metadata, embedding) VALUES (%s, %s, %s, %s) "  # nosec B608 - table_name is developer-controlled, not user input
                    f"ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text, metadata = EXCLUDED.metadata, "
                    f"embedding = EXCLUDED.embedding",
                    (chunk.id, chunk.text, json.dumps(chunk.metadata), chunk.embedding),
                )

    def query(self, embedding: list[float], k: int = 5) -> list[ScoredChunk]:
        if not self._table_ready:
            return []
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT id, text, metadata, 1 - (embedding <=> %s) AS score FROM {self.table_name} "  # nosec B608 - table_name is developer-controlled, not user input
                f"ORDER BY embedding <=> %s LIMIT %s",
                (embedding, embedding, k),
            )
            rows = cur.fetchall()
        return [ScoredChunk(chunk=Chunk(id=r[0], text=r[1] or "", metadata=r[2] or {}), score=r[3]) for r in rows]

    def keyword_query(self, query: str, k: int = 5) -> list[ScoredChunk]:
        if not self._table_ready:
            return []
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT id, text, metadata FROM {self.table_name} WHERE text ILIKE %s LIMIT %s",  # nosec B608 - table_name is developer-controlled, not user input
                (f"%{query}%", k),
            )
            rows = cur.fetchall()
        return [ScoredChunk(chunk=Chunk(id=r[0], text=r[1] or "", metadata=r[2] or {}), score=1.0) for r in rows]

    def get(self, chunk_id: str) -> Chunk | None:
        if not self._table_ready:
            return None
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT id, text, metadata FROM {self.table_name} WHERE id = %s",  # nosec B608 - table_name is developer-controlled, not user input
                (chunk_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return Chunk(id=row[0], text=row[1] or "", metadata=row[2] or {})

    def delete(self, ids: list[str]) -> None:
        if not self._table_ready or not ids:
            return
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table_name} WHERE id = ANY(%s)", (ids,))  # nosec B608 - table_name is developer-controlled, not user input
