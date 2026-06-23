"""Postgres repository for per-user MCP OAuth tokens and DCR client records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from deep_agent.aegra.mcp_crypto import decrypt_secret, encrypt_secret
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_TABLES_ENSURED = False

CREATE_OAUTH_CLIENTS_TABLE = """
CREATE TABLE IF NOT EXISTS mcp_oauth_clients (
    mcp_name           TEXT PRIMARY KEY,
    client_id          TEXT NOT NULL,
    client_secret      TEXT,
    registration_data  JSONB,
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

CREATE_OAUTH_TOKENS_TABLE = """
CREATE TABLE IF NOT EXISTS mcp_oauth_tokens (
    user_id        TEXT NOT NULL,
    mcp_name       TEXT NOT NULL,
    access_token   TEXT NOT NULL,
    refresh_token  TEXT,
    expires_at     TIMESTAMPTZ,
    scopes         TEXT[],
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, mcp_name)
);
"""

MIGRATE_OAUTH_TABLES = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'mcp_oauth_tokens'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'mcp_oauth_tokens'
          AND column_name = 'user_id'
    ) THEN
        DROP TABLE mcp_oauth_tokens;
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'mcp_oauth_clients'
    ) AND (
        NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'mcp_oauth_clients'
              AND column_name = 'client_id'
        )
        OR NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'mcp_oauth_clients'
              AND column_name = 'registration_data'
        )
        OR NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'mcp_oauth_clients'
              AND column_name = 'updated_at'
        )
    ) THEN
        DROP TABLE mcp_oauth_clients;
    END IF;
END $$;
"""


@dataclass
class McpOAuthClient:
    """Registered OAuth client for a DCR-backed MCP server."""

    mcp_name: str
    client_id: str
    client_secret: str | None = None
    registration_data: dict[str, Any] | None = None
    updated_at: datetime | None = None


@dataclass
class McpOAuthToken:
    """Stored OAuth tokens for a (user, MCP) pair."""

    user_id: str
    mcp_name: str
    access_token: str
    refresh_token: str | None = None
    expires_at: datetime | None = None
    scopes: list[str] | None = None
    updated_at: datetime | None = None


class McpTokenStore:
    """Async Postgres store for MCP OAuth clients and user tokens."""

    def __init__(self, database_uri: str) -> None:
        """Initialize with a Postgres connection URI."""
        self._uri = database_uri

    async def ensure_tables(self) -> None:
        """Create MCP OAuth tables if they do not already exist."""
        global _TABLES_ENSURED  # noqa: PLW0603
        async with await psycopg.AsyncConnection.connect(self._uri) as conn:
            await conn.execute(MIGRATE_OAUTH_TABLES)
            await conn.execute(CREATE_OAUTH_CLIENTS_TABLE)
            await conn.execute(CREATE_OAUTH_TOKENS_TABLE)
            await conn.commit()
        if not _TABLES_ENSURED:
            _TABLES_ENSURED = True
            logger.info("MCP OAuth tables ensured")

    async def get_client(self, mcp_name: str) -> McpOAuthClient | None:
        """Return the registered OAuth client for *mcp_name*, if any."""
        await self.ensure_tables()
        async with await psycopg.AsyncConnection.connect(
            self._uri, row_factory=dict_row
        ) as conn:
            cur = await conn.execute(
                "SELECT * FROM mcp_oauth_clients WHERE mcp_name = %s",
                (mcp_name,),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            return McpOAuthClient(
                mcp_name=row["mcp_name"],
                client_id=row["client_id"],
                client_secret=decrypt_secret(row["client_secret"]),
                registration_data=row["registration_data"],
                updated_at=row["updated_at"],
            )

    async def upsert_client(
        self,
        mcp_name: str,
        client_id: str,
        client_secret: str | None = None,
        registration_data: dict[str, Any] | None = None,
    ) -> McpOAuthClient:
        """Insert or update the OAuth client record for *mcp_name*."""
        await self.ensure_tables()
        enc_secret = encrypt_secret(client_secret)
        async with await psycopg.AsyncConnection.connect(self._uri) as conn:
            await conn.execute(
                """
                INSERT INTO mcp_oauth_clients (
                    mcp_name, client_id, client_secret, registration_data, updated_at
                )
                VALUES (%s, %s, %s, %s, now())
                ON CONFLICT (mcp_name) DO UPDATE SET
                    client_id = EXCLUDED.client_id,
                    client_secret = EXCLUDED.client_secret,
                    registration_data = EXCLUDED.registration_data,
                    updated_at = now()
                """,
                (
                    mcp_name,
                    client_id,
                    enc_secret,
                    Jsonb(registration_data) if registration_data is not None else None,
                ),
            )
            await conn.commit()
        return McpOAuthClient(
            mcp_name=mcp_name,
            client_id=client_id,
            client_secret=client_secret,
            registration_data=registration_data,
        )

    async def get_token(self, user_id: str, mcp_name: str) -> McpOAuthToken | None:
        """Return stored OAuth tokens for *(user_id, mcp_name)*."""
        await self.ensure_tables()
        async with await psycopg.AsyncConnection.connect(
            self._uri, row_factory=dict_row
        ) as conn:
            cur = await conn.execute(
                "SELECT * FROM mcp_oauth_tokens WHERE user_id = %s AND mcp_name = %s",
                (user_id, mcp_name),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            return McpOAuthToken(
                user_id=row["user_id"],
                mcp_name=row["mcp_name"],
                access_token=decrypt_secret(row["access_token"]) or "",
                refresh_token=decrypt_secret(row["refresh_token"]),
                expires_at=row["expires_at"],
                scopes=list(row["scopes"]) if row["scopes"] else None,
                updated_at=row["updated_at"],
            )

    async def upsert_token(
        self,
        user_id: str,
        mcp_name: str,
        access_token: str,
        refresh_token: str | None = None,
        expires_at: datetime | None = None,
        scopes: list[str] | None = None,
    ) -> McpOAuthToken:
        """Insert or update OAuth tokens for *(user_id, mcp_name)*."""
        await self.ensure_tables()
        enc_access = encrypt_secret(access_token)
        enc_refresh = encrypt_secret(refresh_token)
        async with await psycopg.AsyncConnection.connect(self._uri) as conn:
            await conn.execute(
                """
                INSERT INTO mcp_oauth_tokens (
                    user_id, mcp_name, access_token, refresh_token,
                    expires_at, scopes, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (user_id, mcp_name) DO UPDATE SET
                    access_token = EXCLUDED.access_token,
                    refresh_token = EXCLUDED.refresh_token,
                    expires_at = EXCLUDED.expires_at,
                    scopes = EXCLUDED.scopes,
                    updated_at = now()
                """,
                (
                    user_id,
                    mcp_name,
                    enc_access,
                    enc_refresh,
                    expires_at,
                    scopes,
                ),
            )
            await conn.commit()
        return McpOAuthToken(
            user_id=user_id,
            mcp_name=mcp_name,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            scopes=scopes,
        )

    @staticmethod
    def expires_at_from_token_response(data: dict[str, Any]) -> datetime | None:
        """Compute expiry from an OAuth token endpoint JSON body."""
        expires_in = data.get("expires_in")
        if expires_in is None:
            return None
        try:
            return datetime.now(UTC) + timedelta(seconds=int(expires_in))
        except (TypeError, ValueError):
            return None
