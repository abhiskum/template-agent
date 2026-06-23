"""Symmetric encryption for MCP OAuth tokens and client secrets at rest."""

from __future__ import annotations

import os
from typing import cast

from cryptography.fernet import Fernet, InvalidToken

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_fernet: Fernet | None = None


def _get_fernet() -> Fernet:
    """Return a process-wide Fernet instance keyed by ``MCP_TOKEN_ENCRYPTION_KEY``."""
    global _fernet  # noqa: PLW0603
    if _fernet is not None:
        return _fernet

    key = os.environ.get("MCP_TOKEN_ENCRYPTION_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "MCP_TOKEN_ENCRYPTION_KEY is required for MCP OAuth token storage. "
            'Generate one with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"'
        )

    _fernet = Fernet(key.encode())
    return _fernet


def encrypt_secret(plaintext: str | None) -> str | None:
    """Encrypt a secret value for Postgres storage."""
    if plaintext is None:
        return None
    return cast(str, _get_fernet().encrypt(plaintext.encode()).decode())


def decrypt_secret(ciphertext: str | None) -> str | None:
    """Decrypt a value previously stored by :func:`encrypt_secret`."""
    if ciphertext is None:
        return None
    try:
        return cast(str, _get_fernet().decrypt(ciphertext.encode()).decode())
    except InvalidToken as exc:
        logger.error("Failed to decrypt MCP secret — key mismatch or corrupt data")
        raise RuntimeError("MCP token decryption failed") from exc
