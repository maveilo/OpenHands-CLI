"""Tests for get_default_critic function with custom endpoints."""

import os
import pytest
from unittest.mock import patch

from openhands.sdk import LLM
from openhands_cli.stores.agent_store import get_default_critic
from openhands_cli.stores.cli_settings import CliSettings, CriticSettings

# Allow short context windows for testing
os.environ["ALLOW_SHORT_CONTEXT_WINDOWS"] = "true"


class TestGetDefaultCriticCustomEndpoint:
    """Tests for get_default_critic with custom OpenAI-compatible endpoints."""

    def test_custom_endpoint_localhost_with_custom_model_name(self) -> None:
        """Should configure critic with custom model name for localhost endpoint."""
        llm = LLM(
            model="gpt-4",
            api_key="test-key",
            base_url="http://localhost:8080/v1",
            usage_id="agent",
        )

        # Set custom critic model name in settings
        mock_settings = CliSettings(
            critic=CriticSettings(
                enable_critic=True,
                model_name="openai/critic_model_name",
            )
        )

        with patch.object(CliSettings, "load", return_value=mock_settings):
            critic = get_default_critic(llm, enable_critic=True)

        assert critic is not None
        assert critic.server_url == "http://localhost:8080/v1"
        assert critic.api_key.get_secret_value() == "no"
        assert critic.model_name == "openai/critic_model_name"

    def test_custom_endpoint_remote_with_custom_model_name(self) -> None:
        """Should configure critic with custom model name for remote endpoint."""
        llm = LLM(
            model="gpt-4",
            api_key="test-key",
            base_url="https://api.example.com/v1",
            usage_id="agent",
        )

        # Set custom critic model name in settings
        mock_settings = CliSettings(
            critic=CriticSettings(
                enable_critic=True,
                model_name="custom/critic-model",
            )
        )

        with patch.object(CliSettings, "load", return_value=mock_settings):
            critic = get_default_critic(llm, enable_critic=True)

        assert critic is not None
        assert critic.server_url == "https://api.example.com/v1"
        assert critic.api_key.get_secret_value() == "test-key"
        assert critic.model_name == "custom/critic-model"

    def test_custom_endpoint_without_trailing_slash(self) -> None:
        """Should handle custom endpoint without trailing slash."""
        llm = LLM(
            model="gpt-4",
            api_key="test-key",
            base_url="http://localhost:8080/v1",
            usage_id="agent",
        )

        mock_settings = CliSettings(
            critic=CriticSettings(
                enable_critic=True,
                model_name="critic",
            )
        )

        with patch.object(CliSettings, "load", return_value=mock_settings):
            critic = get_default_critic(llm, enable_critic=True)

        assert critic is not None
        assert critic.server_url == "http://localhost:8080/v1"
        assert critic.api_key.get_secret_value() == "no"
        assert critic.model_name == "critic"

    def test_custom_endpoint_with_trailing_slash(self) -> None:
        """Should handle custom endpoint with trailing slash."""
        llm = LLM(
            model="gpt-4",
            api_key="test-key",
            base_url="http://localhost:8080/v1/",
            usage_id="agent",
        )

        mock_settings = CliSettings(
            critic=CriticSettings(
                enable_critic=True,
                model_name="critic",
            )
        )

        with patch.object(CliSettings, "load", return_value=mock_settings):
            critic = get_default_critic(llm, enable_critic=True)

        assert critic is not None
        assert critic.server_url == "http://localhost:8080/v1"
        assert critic.api_key.get_secret_value() == "no"
        assert critic.model_name == "critic"

    def test_all_hands_proxy_endpoint_unchanged(self) -> None:
        """Should keep All-Hands proxy endpoint behavior unchanged."""
        llm = LLM(
            model="gpt-4",
            api_key="test-key",
            base_url="https://llm-proxy.app.all-hands.dev/",
            usage_id="agent",
        )

        mock_settings = CliSettings(
            critic=CriticSettings(
                enable_critic=True,
                model_name="custom/critic-model",
            )
        )

        with patch.object(CliSettings, "load", return_value=mock_settings):
            critic = get_default_critic(llm, enable_critic=True)

        assert critic is not None
        assert critic.server_url == "https://llm-proxy.app.all-hands.dev/vllm"
        assert critic.api_key.get_secret_value() == "test-key"
        assert critic.model_name == "critic"  # Should use hardcoded "critic" for proxy

    def test_all_hands_proxy_without_api_key_returns_none(self) -> None:
        """Should return None when All-Hands proxy endpoint has no API key."""
        llm = LLM(
            model="gpt-4",
            api_key=None,
            base_url="https://llm-proxy.app.all-hands.dev/",
            usage_id="agent",
        )

        mock_settings = CliSettings(
            critic=CriticSettings(
                enable_critic=True,
                model_name="critic",
            )
        )

        with patch.object(CliSettings, "load", return_value=mock_settings):
            critic = get_default_critic(llm, enable_critic=True)

        assert critic is None

    def test_custom_endpoint_without_api_key_returns_none(self) -> None:
        """Should return None when custom endpoint has no API key."""
        llm = LLM(
            model="gpt-4",
            api_key=None,
            base_url="http://localhost:8080/v1",
            usage_id="agent",
        )

        mock_settings = CliSettings(
            critic=CriticSettings(
                enable_critic=True,
                model_name="critic",
            )
        )

        with patch.object(CliSettings, "load", return_value=mock_settings):
            critic = get_default_critic(llm, enable_critic=True)

        assert critic is None

    def test_custom_endpoint_without_base_url_returns_none(self) -> None:
        """Should return None when base_url is None."""
        llm = LLM(
            model="gpt-4",
            api_key="test-key",
            base_url=None,
            usage_id="agent",
        )

        mock_settings = CliSettings(
            critic=CriticSettings(
                enable_critic=True,
                model_name="critic",
            )
        )

        with patch.object(CliSettings, "load", return_value=mock_settings):
            critic = get_default_critic(llm, enable_critic=True)

        assert critic is None

    def test_critic_disabled_returns_none(self) -> None:
        """Should return None when critic is disabled."""
        llm = LLM(
            model="gpt-4",
            api_key="test-key",
            base_url="http://localhost:8080/v1",
            usage_id="agent",
        )

        mock_settings = CliSettings(
            critic=CriticSettings(
                enable_critic=False,
                model_name="critic",
            )
        )

        with patch.object(CliSettings, "load", return_value=mock_settings):
            critic = get_default_critic(llm, enable_critic=False)

        assert critic is None

    def test_custom_endpoint_uses_default_model_name(self) -> None:
        """Should use default "critic" model name when not specified in settings."""
        llm = LLM(
            model="gpt-4",
            api_key="test-key",
            base_url="http://localhost:8080/v1",
            usage_id="agent",
        )

        # Settings without model_name field (old format)
        mock_settings = CliSettings(
            critic=CriticSettings(
                enable_critic=True,
                # No model_name specified
            )
        )

        with patch.object(CliSettings, "load", return_value=mock_settings):
            critic = get_default_critic(llm, enable_critic=True)

        assert critic is not None
        assert critic.server_url == "http://localhost:8080/v1"
        assert critic.api_key.get_secret_value() == "no"
        assert critic.model_name == "critic"  # Should use default

    def test_custom_endpoint_with_empty_model_name(self) -> None:
        """Should use default "critic" model name when empty string is specified."""
        llm = LLM(
            model="gpt-4",
            api_key="test-key",
            base_url="http://localhost:8080/v1",
            usage_id="agent",
        )

        mock_settings = CliSettings(
            critic=CriticSettings(
                enable_critic=True,
                model_name="",
            )
        )

        with patch.object(CliSettings, "load", return_value=mock_settings):
            critic = get_default_critic(llm, enable_critic=True)

        assert critic is not None
        assert critic.server_url == "http://localhost:8080/v1"
        assert critic.api_key.get_secret_value() == "no"
        assert critic.model_name == "critic"  # Should use default

    def test_custom_endpoint_http_without_localhost(self) -> None:
        """Should use API key for non-local HTTP endpoints."""
        llm = LLM(
            model="gpt-4",
            api_key="test-key",
            base_url="http://example.com/v1",
            usage_id="agent",
        )

        mock_settings = CliSettings(
            critic=CriticSettings(
                enable_critic=True,
                model_name="critic",
            )
        )

        with patch.object(CliSettings, "load", return_value=mock_settings):
            critic = get_default_critic(llm, enable_critic=True)

        assert critic is not None
        assert critic.server_url == "http://example.com/v1"
        assert critic.api_key.get_secret_value() == "test-key"  # Should use API key
        assert critic.model_name == "critic"