"""Unit tests for MCP OAuth handler edge cases."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.testclient import TestClient

from deep_agent.aegra.http_app import app
from deep_agent.aegra.mcp_oauth_handlers import (
    _callback_html,
    _register_dcr_client,
    handle_mcp_connect,
    handle_mcp_oauth_callback,
)


@pytest.mark.asyncio
class TestHandleMcpConnect:
    async def test_client_credentials_rejects_connect(self):
        server_cfg = {
            "enabled": True,
            "auth_mode": "oauth",
            "oauth": {
                "grant_type": "client_credentials",
                "token_endpoint": "https://auth.example.com/token",
                "client_id": "cid",
            },
        }
        with patch(
            "deep_agent.aegra.mcp_oauth_handlers._get_mcp_server_config",
            return_value=server_cfg,
        ):
            with pytest.raises(HTTPException) as exc:
                await handle_mcp_connect("user-1", "cc-mcp")
            assert exc.value.status_code == 400
            assert "client_credentials" in exc.value.detail
            assert "not required" in exc.value.detail

    async def test_authorization_code_proceeds_past_grant_type_check(self):
        server_cfg = {
            "enabled": True,
            "auth_mode": "oauth",
            "oauth": {
                "grant_type": "authorization_code",
                "authorization_endpoint": "https://auth.example.com/authorize",
                "token_endpoint": "https://auth.example.com/token",
                "client_id": "cid",
            },
        }
        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers._get_mcp_server_config",
                return_value=server_cfg,
            ),
            patch("deep_agent.aegra.mcp_oauth_handlers.settings") as mock_settings,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.cache_set",
                return_value=True,
            ),
        ):
            mock_settings.oauth_callback_url = (
                "https://agent.example.com/mcp/oauth/callback"
            )
            mock_settings.agent_deployment_id = "test-agent"
            result = await handle_mcp_connect("user-1", "oauth-mcp")
            assert "authorize_url" in result
            assert "auth.example.com/authorize" in result["authorize_url"]

    async def test_non_oauth_auth_mode_rejects(self):
        server_cfg = {
            "enabled": True,
            "auth_mode": "sso",
        }
        with patch(
            "deep_agent.aegra.mcp_oauth_handlers._get_mcp_server_config",
            return_value=server_cfg,
        ):
            with pytest.raises(HTTPException) as exc:
                await handle_mcp_connect("user-1", "sso-mcp")
            assert exc.value.status_code == 400
            assert "does not use OAuth" in exc.value.detail


@pytest.mark.asyncio
class TestRegisterDcrClient:
    async def test_uses_requested_scopes(self):
        oauth_cfg = {
            "registration_endpoint": "https://auth.example.com/register",
            "scopes": ["read", "write"],
        }
        server_cfg = {"enabled": True}

        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {
            "client_id": "dcr-cid",
            "client_secret": "dcr-secret",
        }

        mock_ctx = AsyncMock(post=AsyncMock(return_value=mock_response))
        mock_store = MagicMock()
        mock_store.upsert_client = AsyncMock()

        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.httpx.AsyncClient",
            ) as mock_client_cls,
            patch("deep_agent.aegra.mcp_oauth_handlers.settings") as mock_settings,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.mcp_httpx_verify",
                return_value=True,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.McpTokenStore",
                return_value=mock_store,
            ),
        ):
            mock_settings.oauth_callback_url = (
                "https://agent.example.com/mcp/oauth/callback"
            )
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            cid, secret = await _register_dcr_client(
                "test-agent", "dcr-mcp", oauth_cfg, server_cfg
            )

        assert cid == "dcr-cid"
        assert secret == "dcr-secret"
        post_kwargs = mock_ctx.post.call_args
        body = post_kwargs.kwargs.get("json") or post_kwargs[1].get("json", {})
        assert body["scope"] == "read write"


def _mock_request() -> Request:
    """Create a minimal mock Starlette Request for OAuth callback tests."""
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/mcp/oauth/callback",
        "query_string": b"",
        "headers": [],
    }
    return Request(scope)


@pytest.mark.asyncio
class TestHandleMcpOauthCallback:
    async def test_missing_code_returns_error(self):
        response = await handle_mcp_oauth_callback(
            code=None, state="some-state", request=_mock_request()
        )
        assert response.status_code == 400
        assert b"Missing" in response.body

    async def test_missing_state_returns_error(self):
        response = await handle_mcp_oauth_callback(
            code="some-code", state=None, request=_mock_request()
        )
        assert response.status_code == 400
        assert b"Missing" in response.body

    async def test_successful_callback_returns_connected_html(self):
        state_payload = json.dumps(
            {
                "user_id": "user-1",
                "mcp_name": "oauth-mcp",
                "code_verifier": "test-verifier",
            }
        )

        server_cfg = {
            "enabled": True,
            "auth_mode": "oauth",
            "oauth": {
                "token_endpoint": "https://auth.example.com/token",
                "client_id": "cid",
            },
        }

        token_response = MagicMock()
        token_response.json.return_value = {
            "access_token": "new-access-token",
            "expires_in": 3600,
        }
        token_response.raise_for_status = MagicMock()

        mock_ctx = AsyncMock(post=AsyncMock(return_value=token_response))
        mock_store = MagicMock()
        mock_store.upsert_token = AsyncMock()

        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.cache_get",
                return_value=state_payload,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers._get_mcp_server_config",
                return_value=server_cfg,
            ),
            patch("deep_agent.aegra.mcp_oauth_handlers.settings") as mock_settings,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.httpx.AsyncClient",
            ) as mock_client_cls,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.mcp_httpx_verify",
                return_value=True,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.McpTokenStore",
                return_value=mock_store,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.resolve_oauth_client_secret",
                return_value="csecret",
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.get_mcp_credential_resolver",
            ) as mock_resolver,
            patch("deep_agent.aegra.mcp.invalidate_mcp_tool_cache"),
            patch("deep_agent.aegra.graph.invalidate_graph_cache"),
        ):
            mock_settings.oauth_callback_url = (
                "https://agent.example.com/mcp/oauth/callback"
            )
            mock_settings.agent_deployment_id = "test-agent"
            mock_settings.database_uri = "sqlite:///test.db"
            mock_settings.ui_origin = "https://ui.example.com"

            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resolver.return_value.invalidate_cache = MagicMock()

            response = await handle_mcp_oauth_callback(
                code="auth-code", state="valid-state", request=_mock_request()
            )

        assert response.status_code == 200
        assert b"Connected" in response.body
        assert b"mcp_oauth_done" in response.body

    async def test_ok_false_returns_error_with_message(self):
        state_payload = json.dumps(
            {
                "user_id": "user-1",
                "mcp_name": "oauth-mcp",
                "code_verifier": "test-verifier",
            }
        )
        server_cfg = {
            "enabled": True,
            "auth_mode": "oauth",
            "oauth": {
                "token_endpoint": "https://auth.example.com/token",
                "client_id": "cid",
            },
        }
        token_response = MagicMock()
        token_response.json.return_value = {"ok": False, "error": "invalid_code"}
        token_response.raise_for_status = MagicMock()

        mock_ctx = AsyncMock(post=AsyncMock(return_value=token_response))

        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.cache_get",
                return_value=state_payload,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers._get_mcp_server_config",
                return_value=server_cfg,
            ),
            patch("deep_agent.aegra.mcp_oauth_handlers.settings") as mock_settings,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.httpx.AsyncClient",
            ) as mock_client_cls,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.mcp_httpx_verify",
                return_value=True,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.McpTokenStore",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.resolve_oauth_client_secret",
                return_value="csecret",
            ),
        ):
            mock_settings.oauth_callback_url = (
                "https://agent.example.com/mcp/oauth/callback"
            )
            mock_settings.agent_deployment_id = "test-agent"
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            response = await handle_mcp_oauth_callback(
                code="auth-code", state="valid-state", request=_mock_request()
            )

        assert response.status_code == 502
        assert b"invalid_code" in response.body

    async def test_authed_user_access_token_fallback(self):
        state_payload = json.dumps(
            {
                "user_id": "user-1",
                "mcp_name": "oauth-mcp",
                "code_verifier": "test-verifier",
            }
        )
        server_cfg = {
            "enabled": True,
            "auth_mode": "oauth",
            "oauth": {
                "token_endpoint": "https://auth.example.com/token",
                "client_id": "cid",
            },
        }
        token_response = MagicMock()
        token_response.json.return_value = {
            "ok": True,
            "authed_user": {
                "access_token": "xoxp-nested-token",
                "scope": "chat:write",
            },
        }
        token_response.raise_for_status = MagicMock()

        mock_ctx = AsyncMock(post=AsyncMock(return_value=token_response))
        mock_store = MagicMock()
        mock_store.upsert_token = AsyncMock()

        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.cache_get",
                return_value=state_payload,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers._get_mcp_server_config",
                return_value=server_cfg,
            ),
            patch("deep_agent.aegra.mcp_oauth_handlers.settings") as mock_settings,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.httpx.AsyncClient",
            ) as mock_client_cls,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.mcp_httpx_verify",
                return_value=True,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.McpTokenStore",
                return_value=mock_store,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.resolve_oauth_client_secret",
                return_value="csecret",
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.get_mcp_credential_resolver",
            ) as mock_resolver,
            patch("deep_agent.aegra.mcp.invalidate_mcp_tool_cache"),
            patch("deep_agent.aegra.graph.invalidate_graph_cache"),
        ):
            mock_settings.oauth_callback_url = (
                "https://agent.example.com/mcp/oauth/callback"
            )
            mock_settings.agent_deployment_id = "test-agent"
            mock_settings.database_uri = "sqlite:///test.db"
            mock_settings.ui_origin = "https://ui.example.com"

            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resolver.return_value.invalidate_cache = MagicMock()

            response = await handle_mcp_oauth_callback(
                code="auth-code", state="valid-state", request=_mock_request()
            )

        assert response.status_code == 200
        assert b"Connected" in response.body
        mock_store.upsert_token.assert_awaited_once()
        call_kwargs = mock_store.upsert_token.call_args[1]
        assert call_kwargs["access_token"] == "xoxp-nested-token"

    async def test_missing_access_token_everywhere_returns_error(self):
        state_payload = json.dumps(
            {
                "user_id": "user-1",
                "mcp_name": "oauth-mcp",
                "code_verifier": "test-verifier",
            }
        )
        server_cfg = {
            "enabled": True,
            "auth_mode": "oauth",
            "oauth": {
                "token_endpoint": "https://auth.example.com/token",
                "client_id": "cid",
            },
        }
        token_response = MagicMock()
        token_response.json.return_value = {"token_type": "bearer", "expires_in": 3600}
        token_response.raise_for_status = MagicMock()

        mock_ctx = AsyncMock(post=AsyncMock(return_value=token_response))

        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.cache_get",
                return_value=state_payload,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers._get_mcp_server_config",
                return_value=server_cfg,
            ),
            patch("deep_agent.aegra.mcp_oauth_handlers.settings") as mock_settings,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.httpx.AsyncClient",
            ) as mock_client_cls,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.mcp_httpx_verify",
                return_value=True,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.McpTokenStore",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.resolve_oauth_client_secret",
                return_value="csecret",
            ),
        ):
            mock_settings.oauth_callback_url = (
                "https://agent.example.com/mcp/oauth/callback"
            )
            mock_settings.agent_deployment_id = "test-agent"
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            response = await handle_mcp_oauth_callback(
                code="auth-code", state="valid-state", request=_mock_request()
            )

        assert response.status_code == 502
        assert b"missing access_token" in response.body


class TestMcpOauthCallbackRoute:
    def test_callback_route_returns_error_without_params(self):
        client = TestClient(app)
        resp = client.get("/mcp/oauth/callback")
        assert resp.status_code == 400
        assert "Missing" in resp.text


class TestCallbackHtml:
    def test_includes_opener_origin_in_postmessage(self):
        html = _callback_html(
            mcp_name="test-mcp", opener_origin="https://ui.example.com"
        )
        assert '"https://ui.example.com"' in html
        assert "mcp_oauth_done" in html

    def test_skips_postmessage_when_no_origin(self):
        html = _callback_html(mcp_name="test-mcp", opener_origin=None)
        assert "postMessage" not in html
        assert "Connected" in html

    def test_error_html_returned(self):
        result = _callback_html(error="something broke")
        assert "something broke" in result
        assert "MCP OAuth Error" in result

    def test_error_html_escapes_tags(self):
        result = _callback_html(error='<script>alert("xss")</script>')
        assert "<script>" not in result
        assert "&lt;script&gt;" in result


@pytest.mark.asyncio
class TestValidateDcrClient:
    """Tests for _validate_dcr_client helper."""

    async def test_returns_true_on_302_redirect(self):
        from deep_agent.aegra.mcp_oauth_handlers import _validate_dcr_client

        mock_response = MagicMock()
        mock_response.status_code = 302
        mock_response.text = ""

        mock_ctx = AsyncMock(get=AsyncMock(return_value=mock_response))

        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.httpx.AsyncClient"
            ) as mock_client,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.mcp_httpx_verify",
                return_value=True,
            ),
            patch("deep_agent.aegra.mcp_oauth_handlers.settings") as mock_settings,
        ):
            mock_settings.oauth_callback_url = (
                "https://agent.example.com/mcp/oauth/callback"
            )
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await _validate_dcr_client(
                "valid-client-id",
                {"authorization_endpoint": "https://mcp.example.com/authorize"},
                {"enabled": True},
            )

        assert result is True

    async def test_returns_false_on_invalid_client_error(self):
        from deep_agent.aegra.mcp_oauth_handlers import _validate_dcr_client

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = (
            '{"error": "invalid_client", "error_description": "client_id expired"}'
        )

        mock_ctx = AsyncMock(get=AsyncMock(return_value=mock_response))

        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.httpx.AsyncClient"
            ) as mock_client,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.mcp_httpx_verify",
                return_value=True,
            ),
            patch("deep_agent.aegra.mcp_oauth_handlers.settings") as mock_settings,
        ):
            mock_settings.oauth_callback_url = (
                "https://agent.example.com/mcp/oauth/callback"
            )
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await _validate_dcr_client(
                "expired-client-id",
                {"authorization_endpoint": "https://mcp.example.com/authorize"},
                {"enabled": True},
            )

        assert result is False

    async def test_returns_true_when_no_authorization_endpoint(self):
        from deep_agent.aegra.mcp_oauth_handlers import _validate_dcr_client

        result = await _validate_dcr_client(
            "some-client-id",
            {},
            {"enabled": True},
        )
        assert result is True

    async def test_returns_true_on_network_error(self):
        from deep_agent.aegra.mcp_oauth_handlers import _validate_dcr_client

        mock_ctx = AsyncMock(get=AsyncMock(side_effect=Exception("connection refused")))

        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.httpx.AsyncClient"
            ) as mock_client,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.mcp_httpx_verify",
                return_value=True,
            ),
            patch("deep_agent.aegra.mcp_oauth_handlers.settings") as mock_settings,
        ):
            mock_settings.oauth_callback_url = (
                "https://agent.example.com/mcp/oauth/callback"
            )
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await _validate_dcr_client(
                "some-client-id",
                {"authorization_endpoint": "https://mcp.example.com/authorize"},
                {"enabled": True},
            )

        assert result is True

    async def test_returns_true_on_unknown_error_status(self):
        from deep_agent.aegra.mcp_oauth_handlers import _validate_dcr_client

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_ctx = AsyncMock(get=AsyncMock(return_value=mock_response))

        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.httpx.AsyncClient"
            ) as mock_client,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.mcp_httpx_verify",
                return_value=True,
            ),
            patch("deep_agent.aegra.mcp_oauth_handlers.settings") as mock_settings,
        ):
            mock_settings.oauth_callback_url = (
                "https://agent.example.com/mcp/oauth/callback"
            )
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await _validate_dcr_client(
                "some-client-id",
                {"authorization_endpoint": "https://mcp.example.com/authorize"},
                {"enabled": True},
            )

        assert result is True


@pytest.mark.asyncio
class TestHandleMcpConnectDcrValidation:
    """Tests for handle_mcp_connect DCR client_id validation and re-registration."""

    async def test_expired_client_triggers_reregistration(self):
        from deep_agent.aegra.mcp_token_store import McpOAuthClient

        existing_client = McpOAuthClient(
            agent_name="test-agent",
            mcp_name="dcr-mcp",
            client_id="expired-cid",
            client_secret="old-secret",
        )
        new_client = McpOAuthClient(
            agent_name="test-agent",
            mcp_name="dcr-mcp",
            client_id="fresh-cid",
            client_secret="new-secret",
        )

        server_cfg = {
            "enabled": True,
            "auth_mode": "dcr",
            "oauth": {
                "authorization_endpoint": "https://mcp.example.com/authorize",
                "token_endpoint": "https://mcp.example.com/token",
                "registration_endpoint": "https://mcp.example.com/register",
            },
        }

        mock_store = MagicMock()
        get_client_calls = [existing_client, new_client]
        mock_store.get_client = AsyncMock(side_effect=get_client_calls)
        mock_store.delete_client = AsyncMock()

        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers._get_mcp_server_config",
                return_value=server_cfg,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers._validate_dcr_client",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers._register_dcr_client",
                new=AsyncMock(return_value=("fresh-cid", "new-secret")),
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.McpTokenStore",
                return_value=mock_store,
            ),
            patch("deep_agent.aegra.mcp_oauth_handlers.settings") as mock_settings,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.cache_set",
                return_value=True,
            ),
        ):
            mock_settings.oauth_callback_url = (
                "https://agent.example.com/mcp/oauth/callback"
            )
            mock_settings.agent_deployment_id = "test-agent"

            result = await handle_mcp_connect("user-1", "dcr-mcp")

        assert "authorize_url" in result
        mock_store.delete_client.assert_awaited_once_with("test-agent", "dcr-mcp")

    async def test_valid_client_skips_reregistration(self):
        from deep_agent.aegra.mcp_token_store import McpOAuthClient

        existing_client = McpOAuthClient(
            agent_name="test-agent",
            mcp_name="dcr-mcp",
            client_id="valid-cid",
            client_secret="secret",
        )

        server_cfg = {
            "enabled": True,
            "auth_mode": "dcr",
            "oauth": {
                "authorization_endpoint": "https://mcp.example.com/authorize",
                "token_endpoint": "https://mcp.example.com/token",
                "registration_endpoint": "https://mcp.example.com/register",
            },
        }

        mock_store = MagicMock()
        mock_store.get_client = AsyncMock(return_value=existing_client)
        mock_store.delete_client = AsyncMock()

        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers._get_mcp_server_config",
                return_value=server_cfg,
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers._validate_dcr_client",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.McpTokenStore",
                return_value=mock_store,
            ),
            patch("deep_agent.aegra.mcp_oauth_handlers.settings") as mock_settings,
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.cache_set",
                return_value=True,
            ),
        ):
            mock_settings.oauth_callback_url = (
                "https://agent.example.com/mcp/oauth/callback"
            )
            mock_settings.agent_deployment_id = "test-agent"

            result = await handle_mcp_connect("user-1", "dcr-mcp")

        assert "authorize_url" in result
        assert "valid-cid" in result["authorize_url"]
        mock_store.delete_client.assert_not_awaited()
