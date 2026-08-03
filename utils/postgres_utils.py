"""Small, read-only PostgreSQL helper used by dataset resolution.

The helper intentionally exposes table-oriented operations instead of accepting
arbitrary SQL from callers.  Schema and table names are composed with
``psycopg2.sql.Identifier`` so identifiers selected by an LLM are never
interpolated into SQL strings.
"""

from __future__ import annotations

from typing import Any

import psycopg2
from psycopg2 import sql


class PostgreSQLDatabase:
    """Read metadata and table contents from one PostgreSQL schema."""

    def __init__(
        self,
        host: str,
        port: str | int,
        user: str,
        password: str,
        database: str,
        schema: str = "public",
    ) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.schema = schema
        self.connection = None

    def connect(self) -> None:
        self.connection = psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            dbname=self.database,
        )

    def close_connection(self) -> None:
        if self.connection is not None and not self.connection.closed:
            self.connection.close()

    def get_database_info(self) -> list[dict[str, Any]]:
        """Return table names and ordered column names for the configured schema."""

        self._require_connection()
        query = """
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = %s
            ORDER BY table_name, ordinal_position
        """
        with self.connection.cursor() as cursor:
            cursor.execute(query, (self.schema,))
            rows = cursor.fetchall()

        tables: dict[str, list[str]] = {}
        for table_name, column_name in rows:
            tables.setdefault(table_name, []).append(column_name)
        return [
            {"dataset_name": table_name, "columns": columns}
            for table_name, columns in tables.items()
        ]

    def read_table(self, table_name: str) -> tuple[list[tuple[Any, ...]], list[str]]:
        """Read a table using safely quoted schema and table identifiers."""

        self._require_connection()
        query = sql.SQL("SELECT * FROM {}.{}").format(
            sql.Identifier(self.schema),
            sql.Identifier(table_name),
        )
        with self.connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [description.name for description in cursor.description]
        return rows, columns

    def _require_connection(self) -> None:
        if self.connection is None or self.connection.closed:
            raise RuntimeError("PostgreSQL database is not connected")
