import sqlite3

import pytest

from kel.tools import sql_query_tool


def _seeded_connection():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO users VALUES (1, 'Ada'), (2, 'Grace')")
    conn.commit()
    return conn


def test_sql_query_tool_runs_real_select_against_sqlite():
    tool = sql_query_tool(connection=_seeded_connection())
    result = tool({"query": "SELECT id, name FROM users ORDER BY id"})
    assert result == "id, name\n1, Ada\n2, Grace"


def test_sql_query_tool_read_only_blocks_write_queries():
    tool = sql_query_tool(connection=_seeded_connection())
    result = tool({"query": "DELETE FROM users"})
    assert "only SELECT queries are allowed" in result


def test_sql_query_tool_allows_writes_when_read_only_false():
    conn = _seeded_connection()
    tool = sql_query_tool(connection=conn, read_only=False)
    tool({"query": "DELETE FROM users WHERE id = 1"})
    conn.commit()
    result = tool({"query": "SELECT id, name FROM users"})
    assert result == "id, name\n2, Grace"


def test_sql_query_tool_handles_query_errors_gracefully():
    tool = sql_query_tool(connection=_seeded_connection())
    result = tool({"query": "SELECT * FROM nonexistent_table"})
    assert "error executing query" in result


def test_sql_query_tool_no_rows():
    tool = sql_query_tool(connection=_seeded_connection())
    result = tool({"query": "SELECT * FROM users WHERE id = 999"})
    assert result == "(no rows)"


def test_sql_query_tool_dsn_fallback_uses_sqlite():
    tool = sql_query_tool(dsn=":memory:")
    result = tool({"query": "SELECT 1"})
    assert "1" in result


def test_sql_query_tool_requires_connection_or_dsn():
    with pytest.raises(ValueError):
        sql_query_tool()
