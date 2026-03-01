"""Tests for LLM providers.

Uses mocking to avoid calling real APIs in tests.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from rag_eval.llm import OpenAIProvider, AnthropicProvider, OllamaProvider
from rag_eval.llm.base import LLMResponse


class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    @patch("rag_eval.llm.openai_provider.OpenAI")
    def test_complete_basic(self, mock_openai_class: Mock) -> None:
        """Test basic completion."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is a test response"
        mock_response.model = "gpt-4o-mini"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20

        mock_client.chat.completions.create.return_value = mock_response

        # Test
        provider = OpenAIProvider(api_key="test_key", model="gpt-4o-mini")
        result = provider.complete("Hello, world!")

        # Verify
        assert isinstance(result, LLMResponse)
        assert result.content == "This is a test response"
        assert result.model == "gpt-4o-mini"
        assert result.provider == "openai"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 20

    @patch("rag_eval.llm.openai_provider.OpenAI")
    def test_complete_with_system_prompt(self, mock_openai_class: Mock) -> None:
        """Test completion with system prompt."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.model = "gpt-4o-mini"
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 25

        mock_client.chat.completions.create.return_value = mock_response

        # Test
        provider = OpenAIProvider(api_key="test_key")
        result = provider.complete("User prompt", system_prompt="System prompt")

        # Verify messages were passed correctly
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_provider_name(self) -> None:
        """Test provider name property."""
        with patch("rag_eval.llm.openai_provider.OpenAI"):
            provider = OpenAIProvider(api_key="test_key")
            assert provider.provider_name == "openai"


class TestAnthropicProvider:
    """Tests for Anthropic provider."""

    @patch("rag_eval.llm.anthropic_provider.Anthropic")
    def test_complete_basic(self, mock_anthropic_class: Mock) -> None:
        """Test basic completion."""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_content_block = Mock()
        mock_content_block.text = "This is a Claude response"

        mock_response = Mock()
        mock_response.content = [mock_content_block]
        mock_response.model = "claude-haiku-4-5"
        mock_response.usage.input_tokens = 12
        mock_response.usage.output_tokens = 22

        mock_client.messages.create.return_value = mock_response

        # Test
        provider = AnthropicProvider(api_key="test_key", model="claude-haiku-4-5")
        result = provider.complete("Hello, Claude!")

        # Verify
        assert isinstance(result, LLMResponse)
        assert result.content == "This is a Claude response"
        assert result.model == "claude-haiku-4-5"
        assert result.provider == "anthropic"
        assert result.prompt_tokens == 12
        assert result.completion_tokens == 22

    @patch("rag_eval.llm.anthropic_provider.Anthropic")
    def test_complete_with_system_prompt(self, mock_anthropic_class: Mock) -> None:
        """Test completion with system prompt."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_content_block = Mock()
        mock_content_block.text = "Response"

        mock_response = Mock()
        mock_response.content = [mock_content_block]
        mock_response.model = "claude-haiku-4-5"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20

        mock_client.messages.create.return_value = mock_response

        # Test
        provider = AnthropicProvider(api_key="test_key")
        result = provider.complete("User prompt", system_prompt="System prompt")

        # Verify system prompt was passed
        call_args = mock_client.messages.create.call_args
        assert "system" in call_args.kwargs
        assert call_args.kwargs["system"] == "System prompt"

    def test_provider_name(self) -> None:
        """Test provider name property."""
        with patch("rag_eval.llm.anthropic_provider.Anthropic"):
            provider = AnthropicProvider(api_key="test_key")
            assert provider.provider_name == "anthropic"


class TestOllamaProvider:
    """Tests for Ollama provider."""

    @patch("rag_eval.llm.ollama_provider.httpx.Client")
    def test_complete_basic(self, mock_client_class: Mock) -> None:
        """Test basic completion."""
        # Setup mock
        mock_client = Mock()
        mock_client_class.return_value.__enter__.return_value = mock_client

        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "This is an Ollama response",
            "prompt_eval_count": 8,
            "eval_count": 18,
        }

        mock_client.post.return_value = mock_response

        # Test
        provider = OllamaProvider(base_url="http://localhost:11434", model="llama3.2")
        result = provider.complete("Hello, Llama!")

        # Verify
        assert isinstance(result, LLMResponse)
        assert result.content == "This is an Ollama response"
        assert result.model == "llama3.2"
        assert result.provider == "ollama"
        assert result.prompt_tokens == 8
        assert result.completion_tokens == 18

    @patch("rag_eval.llm.ollama_provider.httpx.Client")
    def test_complete_with_system_prompt(self, mock_client_class: Mock) -> None:
        """Test completion with system prompt."""
        mock_client = Mock()
        mock_client_class.return_value.__enter__.return_value = mock_client

        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Response",
            "prompt_eval_count": 10,
            "eval_count": 20,
        }

        mock_client.post.return_value = mock_response

        # Test
        provider = OllamaProvider()
        result = provider.complete("User prompt", system_prompt="System prompt")

        # Verify the prompt was combined
        call_args = mock_client.post.call_args
        payload = call_args.kwargs["json"]
        assert "System prompt" in payload["prompt"]
        assert "User prompt" in payload["prompt"]

    def test_provider_name(self) -> None:
        """Test provider name property."""
        provider = OllamaProvider()
        assert provider.provider_name == "ollama"

    @patch("rag_eval.llm.ollama_provider.httpx.Client")
    def test_missing_token_counts(self, mock_client_class: Mock) -> None:
        """Test handling of missing token counts."""
        mock_client = Mock()
        mock_client_class.return_value.__enter__.return_value = mock_client

        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Response without token counts",
        }

        mock_client.post.return_value = mock_response

        # Test
        provider = OllamaProvider()
        result = provider.complete("Test")

        # Should default to 0 for missing token counts
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
