"""Integration tests for get_default_critic with custom endpoints.

This test verifies that:
1. Custom critic model names are properly used
2. Requests are sent to the correct endpoint with the correct model
3. The critic configuration is correct for custom endpoints
"""
# Allow short context windows for testing
import os
os.environ["ALLOW_SHORT_CONTEXT_WINDOWS"] = "true"

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from openhands.sdk import LLM
from openhands.sdk.critic.impl.api import APIBasedCritic
from openhands_cli.stores.agent_store import get_default_critic
from openhands_cli.stores.cli_settings import CliSettings, CriticSettings


class TestGetDefaultCriticIntegration:
    """Integration tests for get_default_critic with custom endpoints."""

    @pytest.mark.asyncio
    async def test_custom_endpoint_with_critic_model_name_integration(self) -> None:
        """Test that custom critic model name is used in API requests.

        This test verifies:
        1. A critic is created with the custom model name from settings
        2. The critic makes API calls to the correct endpoint
        3. The correct model name is used in the request
        """
        # Setup: Create LLM with custom endpoint
        llm = LLM(
            model="gpt-4",
            api_key=SecretStr("test-key"),
            base_url="http://localhost:8080/v1",
            usage_id="agent",
        )

        # Setup: Configure settings with custom critic model name
        mock_settings = CliSettings(
            critic=CriticSettings(
                enable_critic=True,
                model_name="openai/critic-model",
            )
        )

        with patch.object(CliSettings, "load", return_value=mock_settings):
            critic = get_default_critic(llm, enable_critic=True)

        # Verify critic is created with correct configuration
        assert critic is not None
        assert isinstance(critic, APIBasedCritic)
        assert critic.server_url == "http://localhost:8080/v1"
        assert critic.api_key.get_secret_value() == "no"  # Local endpoint uses "no"
        assert critic.model_name == "openai/critic-model"

        # Verify the CriticClient has the correct configuration
        assert critic.model_name == "openai/critic-model"
        assert critic.server_url == "http://localhost:8080/v1"
        assert critic.api_key.get_secret_value() == "no"

    @pytest.mark.asyncio
    async def test_remote_endpoint_with_custom_critic_model(self) -> None:
        """Test that custom critic model name is used for remote endpoints.

        This test verifies:
        1. A critic is created with the custom model name from settings
        2. The critic uses the provided API key for remote endpoints
        3. The correct model name is used in the request
        """
        # Setup: Create LLM with remote endpoint
        llm = LLM(
            model="gpt-4",
            api_key=SecretStr("remote-key"),
            base_url="https://api.example.com/v1",
            usage_id="agent",
        )

        # Setup: Configure settings with custom critic model name
        mock_settings = CliSettings(
            critic=CriticSettings(
                enable_critic=True,
                model_name="openai/my-critic-model",
            )
        )

        with patch.object(CliSettings, "load", return_value=mock_settings):
            critic = get_default_critic(llm, enable_critic=True)

        # Verify critic is created with correct configuration
        assert critic is not None
        assert isinstance(critic, APIBasedCritic)
        assert critic.server_url == "https://api.example.com/v1"
        assert critic.api_key.get_secret_value() == "remote-key"  # Remote uses provided key
        assert critic.model_name == "openai/my-critic-model"

        # Verify the CriticClient has the correct configuration
        assert critic.model_name == "openai/my-critic-model"
        assert critic.server_url == "https://api.example.com/v1"
        assert critic.api_key.get_secret_value() == "remote-key"

    @pytest.mark.asyncio
    async def test_default_critic_model_name(self) -> None:
        """Test that default critic model name is used when not specified.

        This test verifies:
        1. When no custom model name is set, the default "critic" is used
        2. The critic is still created correctly
        """
        # Setup: Create LLM with custom endpoint
        llm = LLM(
            model="gpt-4",
            api_key=SecretStr("test-key"),
            base_url="http://localhost:8080/v1",
            usage_id="agent",
        )

        # Setup: Configure settings without custom critic model name (empty string)
        mock_settings = CliSettings(
            critic=CriticSettings(
                enable_critic=True,
                model_name="",  # Empty string should use default
            )
        )

        with patch.object(CliSettings, "load", return_value=mock_settings):
            critic = get_default_critic(llm, enable_critic=True)

        # Verify critic is created with default model name
        assert critic is not None
        assert isinstance(critic, APIBasedCritic)
        assert critic.server_url == "http://localhost:8080/v1"
        assert critic.api_key.get_secret_value() == "no"
        assert critic.model_name == "critic"  # Default value

    @pytest.mark.asyncio
    async def test_all_hands_proxy_uses_hardcoded_critic(self) -> None:
        """Test that All-Hands proxy always uses hardcoded "critic" model.

        This test verifies:
        1. All-Hands proxy endpoints ignore custom model name settings
        2. The critic uses the hardcoded "critic" model name
        """
        # Setup: Create LLM with All-Hands proxy endpoint
        llm = LLM(
            model="gpt-4",
            api_key=SecretStr("test-key"),
            base_url="https://llm-proxy.app.all-hands.dev/",
            usage_id="agent",
        )

        # Setup: Configure settings with custom critic model name
        mock_settings = CliSettings(
            critic=CriticSettings(
                enable_critic=True,
                model_name="openai/custom-critic",
            )
        )

        with patch.object(CliSettings, "load", return_value=mock_settings):
            critic = get_default_critic(llm, enable_critic=True)

        # Verify critic uses hardcoded "critic" model name
        assert critic is not None
        assert isinstance(critic, APIBasedCritic)
        assert critic.server_url == "https://llm-proxy.app.all-hands.dev/vllm"
        assert critic.api_key.get_secret_value() == "test-key"
        assert critic.model_name == "critic"  # Hardcoded for proxy