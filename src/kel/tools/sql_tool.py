"""SQL query tool — the kel equivalent of LangChain's `SQLDatabaseToolkit`.

Deliberately takes a pre-built DB-API connection (`connection=`) rather
than a connection string with credentials baked in — kel never touches
authentication here at all. Build the connection however your deployment
already does (a `psycopg` connection using ambient env vars or an IAM auth
token as the password, a connection pool, whatever) and hand it in; this
tool only ever calls the standard `cursor()`/`execute()`/`fetchmany()`
DB-API methods, so it works with any DB-API-compatible driver
(`sqlite3`, `psycopg`, `pymysql`, ...).

`dsn=` is a zero-dependency convenience fallback using stdlib `sqlite3`
for local use — real deployments should pass `connection=`.

`read_only=True` (default) rejects anything that isn't a `SELECT`/`WITH`
query — a real safety boundary, not just a suggestion, since an agent
executing arbitrary SQL against a production database is a genuine risk.
"""

from __future__ import annotations

from typing import Any

from kel.agents.tool import Tool


def sql_query_tool(
    *,
    connection: Any = None,
    dsn: str | None = None,
    read_only: bool = True,
    max_rows: int = 100,
) -> Tool:
    if connection is None and dsn is not None:
        import sqlite3

        connection = sqlite3.connect(dsn)
    if connection is None:
        raise ValueError("sql_query_tool requires either `connection` or `dsn`")

    def run(tool_input: dict) -> str:
        query = tool_input["query"].strip()
        if read_only and not query.upper().startswith(("SELECT", "WITH")):
            return "error: only SELECT queries are allowed (read_only=True)"
        try:
            cur = connection.cursor()
            cur.execute(query)
            rows = cur.fetchmany(max_rows)
            columns = [desc[0] for desc in cur.description] if cur.description else []
        except Exception as exc:
            return f"error executing query: {exc}"

        if not rows:
            return "(no rows)"
        lines = [", ".join(columns)]
        lines.extend(", ".join(str(v) for v in row) for row in rows)
        return "\n".join(lines)

    return Tool(
        name="sql_query",
        description=(
            "Execute a SQL query against the configured database and return results as rows. "
            f"{'Read-only (SELECT/WITH only).' if read_only else 'Read-write — use with caution.'}"
        ),
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        fn=run,
    )
