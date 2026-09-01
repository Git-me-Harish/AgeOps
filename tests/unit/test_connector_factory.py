# tests/unit/test_connector_factory.py
"""
Regression test for a real bug found while wiring a local Postgres up as a
drift-check reference dataset: ConnectorFactory.from_uri()'s documented
postgresql://...?table=foo URI form was never actually parsed anywhere —
PostgreSQLConnector._build_dsn() forwarded the whole URI (query string
included) straight to asyncpg.create_pool() as the DSN, and asyncpg
rejects "table"/"where" as unrecognized connection parameters. This had
never been exercised by any test; the only occurrence of that exact URI
form in the whole repo was the docstring example itself.
"""
from __future__ import annotations

from agents.connectors import ConnectorFactory, _split_postgres_connector_params
from agents.connectors.postgres_connector import PostgreSQLConnector


class TestSplitPostgresConnectorParams:
    def test_table_and_where_are_extracted_not_left_on_dsn(self):
        base, extra = _split_postgres_connector_params(
            "postgresql://user:pw@host:5432/db?table=features&where=created_at>'2025-01-01'"
        )
        assert extra == {"table": "features", "where": "created_at>'2025-01-01'"}
        assert "table=" not in base
        assert "where=" not in base
        assert base.startswith("postgresql://user:pw@host:5432/db")

    def test_real_postgres_params_are_preserved(self):
        base, extra = _split_postgres_connector_params(
            "postgresql://user:pw@host:5432/db?table=features&sslmode=require"
        )
        assert extra == {"table": "features"}
        assert "sslmode=require" in base

    def test_uri_with_no_query_string_is_unaffected(self):
        base, extra = _split_postgres_connector_params("postgresql://user:pw@host:5432/db")
        assert extra == {}
        assert base == "postgresql://user:pw@host:5432/db"


class TestConnectorFactoryPostgresUri:
    def test_from_uri_populates_extra_table_not_the_dsn(self):
        connector = ConnectorFactory.from_uri("postgresql://user:pw@host:5432/db?table=demo_reference_features")
        assert isinstance(connector, PostgreSQLConnector)
        assert connector.config.extra.get("table") == "demo_reference_features"
        assert "table=" not in connector.config.uri

    def test_from_uri_kwargs_extra_still_merges(self):
        connector = ConnectorFactory.from_uri(
            "postgresql://user:pw@host:5432/db", extra={"table": "explicit_table"},
        )
        assert connector.config.extra.get("table") == "explicit_table"
