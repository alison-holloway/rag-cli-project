"""Tests for LLM client module."""

# Check if anthropic is available
import importlib.util
from unittest.mock import MagicMock, patch

import pytest

from src.llm_client import (
    ClaudeClient,
    LLMClient,
    LLMResponse,
    OllamaClient,
    get_llm_client,
)

HAS_ANTHROPIC = importlib.util.find_spec("anthropic") is not None


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_llm_response_creation(self):
        """Test creating an LLM response."""
        response = LLMResponse(
            content="Generated text",
            model="llama3.1:8b",
            provider="ollama",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

        assert response.content == "Generated text"
        assert response.model == "llama3.1:8b"
        assert response.provider == "ollama"
        assert response.token_count == 150

    def test_llm_response_without_tokens(self):
        """Test LLM response without token counts."""
        response = LLMResponse(
            content="Generated text",
            model="test-model",
            provider="test",
        )

        assert response.token_count is None


class TestOllamaClient:
    """Tests for OllamaClient class."""

    @pytest.fixture
    def mock_ollama_client(self):
        """Create a mock Ollama client."""
        with patch("src.llm_client.ollama.Client") as mock:
            client_instance = MagicMock()
            mock.return_value = client_instance
            yield client_instance

    def test_ollama_client_initialization(self, mock_ollama_client):
        """Test Ollama client initialization."""
        client = OllamaClient(model="test-model", base_url="http://localhost:11434")

        assert client.model == "test-model"
        assert client.base_url == "http://localhost:11434"

    def test_ollama_client_default_model(self, mock_ollama_client):
        """Test Ollama client uses config defaults."""
        client = OllamaClient()

        # Should have a model from config
        assert client.model is not None

    def test_ollama_generate(self, mock_ollama_client):
        """Test Ollama generate method."""
        mock_ollama_client.chat.return_value = {
            "message": {"content": "Test response"},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        client = OllamaClient()
        response = client.generate("Test prompt", system_prompt="Be helpful")

        assert isinstance(response, LLMResponse)
        assert response.content == "Test response"
        assert response.provider == "ollama"
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 5
        assert response.total_tokens == 15

    def test_ollama_generate_without_system_prompt(self, mock_ollama_client):
        """Test generation without system prompt."""
        mock_ollama_client.chat.return_value = {
            "message": {"content": "Response"},
        }

        client = OllamaClient()
        client.generate("Test prompt")

        # Should be called with just user message
        call_args = mock_ollama_client.chat.call_args
        messages = call_args[1]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_ollama_generate_with_system_prompt(self, mock_ollama_client):
        """Test generation with system prompt."""
        mock_ollama_client.chat.return_value = {
            "message": {"content": "Response"},
        }

        client = OllamaClient()
        client.generate("Test prompt", system_prompt="Be helpful")

        call_args = mock_ollama_client.chat.call_args
        messages = call_args[1]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_ollama_generate_with_temperature(self, mock_ollama_client):
        """Test generation with custom temperature."""
        mock_ollama_client.chat.return_value = {
            "message": {"content": "Response"},
        }

        client = OllamaClient()
        client.generate("Test prompt", temperature=0.5)

        call_args = mock_ollama_client.chat.call_args
        assert call_args[1]["options"]["temperature"] == 0.5

    def test_ollama_generate_error(self, mock_ollama_client):
        """Test handling of Ollama errors."""
        import ollama

        mock_ollama_client.chat.side_effect = ollama.ResponseError("API error")

        client = OllamaClient()
        with pytest.raises(RuntimeError) as exc_info:
            client.generate("Test prompt")

        assert "Ollama generation failed" in str(exc_info.value)

    def test_ollama_generate_stream(self, mock_ollama_client):
        """Test streaming generation."""
        mock_ollama_client.chat.return_value = iter([
            {"message": {"content": "Hello"}},
            {"message": {"content": " world"}},
            {"message": {"content": "!"}},
        ])

        client = OllamaClient()
        chunks = list(client.generate_stream("Test prompt"))

        assert chunks == ["Hello", " world", "!"]

    def test_ollama_is_available_true(self, mock_ollama_client):
        """Test is_available when Ollama is running."""
        mock_ollama_client.list.return_value = {
            "models": [
                {"name": "llama3.1:8b"},
                {"name": "mistral:7b"},
            ]
        }

        client = OllamaClient(model="llama3.1:8b")
        assert client.is_available() is True

    def test_ollama_is_available_model_not_found(self, mock_ollama_client):
        """Test is_available when model is not available."""
        mock_ollama_client.list.return_value = {
            "models": [
                {"name": "mistral:7b"},
            ]
        }

        client = OllamaClient(model="llama3.1:8b")
        assert client.is_available() is False

    def test_ollama_is_available_connection_error(self, mock_ollama_client):
        """Test is_available when Ollama is not running."""
        mock_ollama_client.list.side_effect = Exception("Connection refused")

        client = OllamaClient()
        assert client.is_available() is False

    def test_ollama_list_models(self, mock_ollama_client):
        """Test listing available models."""
        mock_ollama_client.list.return_value = {
            "models": [
                {"name": "llama3.1:8b"},
                {"name": "mistral:7b"},
            ]
        }

        client = OllamaClient()
        models = client.list_models()

        assert models == ["llama3.1:8b", "mistral:7b"]

    def test_ollama_list_models_error(self, mock_ollama_client):
        """Test list_models handles errors."""
        mock_ollama_client.list.side_effect = Exception("Error")

        client = OllamaClient()
        models = client.list_models()

        assert models == []

    def test_ollama_pull_model(self, mock_ollama_client):
        """Test pulling a model."""
        client = OllamaClient()
        result = client.pull_model("test-model")

        assert result is True
        mock_ollama_client.pull.assert_called_with("test-model")

    def test_ollama_pull_model_error(self, mock_ollama_client):
        """Test pull_model handles errors."""
        mock_ollama_client.pull.side_effect = Exception("Pull failed")

        client = OllamaClient()
        result = client.pull_model("test-model")

        assert result is False


class TestClaudeClient:
    """Tests for ClaudeClient class."""

    def test_claude_client_no_api_key(self):
        """Test Claude client without API key."""
        with patch("src.llm_client.get_settings") as mock_settings:
            mock_settings.return_value.llm.anthropic_api_key = None
            client = ClaudeClient()

            assert client._client is None
            assert client.is_available() is False

    @pytest.mark.skipif(not HAS_ANTHROPIC, reason="anthropic not installed")
    def test_claude_client_with_api_key(self):
        """Test Claude client with API key."""
        with patch("src.llm_client.get_settings") as mock_settings:
            mock_settings.return_value.llm.anthropic_api_key = "test-key"

            with patch("anthropic.Anthropic") as mock_anthropic:
                client = ClaudeClient(api_key="test-key")

                assert client._client is not None
                mock_anthropic.assert_called_with(api_key="test-key")

    def test_claude_generate_no_client(self):
        """Test generate raises error when client not available."""
        with patch("src.llm_client.get_settings") as mock_settings:
            mock_settings.return_value.llm.anthropic_api_key = None
            client = ClaudeClient()

            with pytest.raises(RuntimeError) as exc_info:
                client.generate("Test prompt")

            assert "Claude client not available" in str(exc_info.value)

    @pytest.mark.skipif(not HAS_ANTHROPIC, reason="anthropic not installed")
    def test_claude_generate(self):
        """Test Claude generate method."""
        with patch("src.llm_client.get_settings") as mock_settings:
            mock_settings.return_value.llm.anthropic_api_key = "test-key"
            mock_settings.return_value.llm.llm_temperature = 0.7
            mock_settings.return_value.llm.max_tokens = 1000

            with patch("anthropic.Anthropic") as mock_anthropic:
                mock_client = MagicMock()
                mock_anthropic.return_value = mock_client

                # Mock response
                mock_response = MagicMock()
                mock_response.content = [MagicMock(text="Test response")]
                mock_response.usage.input_tokens = 10
                mock_response.usage.output_tokens = 5
                mock_client.messages.create.return_value = mock_response

                client = ClaudeClient(api_key="test-key")
                response = client.generate("Test prompt", system_prompt="Be helpful")

                assert response.content == "Test response"
                assert response.provider == "claude"
                assert response.total_tokens == 15

    def test_claude_generate_stream_no_client(self):
        """Test streaming raises error when client not available."""
        with patch("src.llm_client.get_settings") as mock_settings:
            mock_settings.return_value.llm.anthropic_api_key = None
            client = ClaudeClient()

            with pytest.raises(RuntimeError):
                list(client.generate_stream("Test prompt"))


class TestLLMClient:
    """Tests for unified LLMClient class."""

    @patch("src.llm_client.OllamaClient")
    def test_llm_client_ollama_provider(self, mock_ollama_class):
        """Test LLMClient with Ollama provider."""
        mock_instance = MagicMock()
        mock_ollama_class.return_value = mock_instance

        client = LLMClient(provider="ollama")

        assert client.provider == "ollama"
        mock_ollama_class.assert_called_once()

    @patch("src.llm_client.ClaudeClient")
    def test_llm_client_claude_provider(self, mock_claude_class):
        """Test LLMClient with Claude provider."""
        mock_instance = MagicMock()
        mock_claude_class.return_value = mock_instance

        client = LLMClient(provider="claude")

        assert client.provider == "claude"
        mock_claude_class.assert_called_once()

    def test_llm_client_unknown_provider(self):
        """Test LLMClient with unknown provider."""
        with pytest.raises(ValueError) as exc_info:
            LLMClient(provider="unknown")

        assert "Unknown LLM provider" in str(exc_info.value)

    @patch("src.llm_client.OllamaClient")
    def test_llm_client_generate_delegates(self, mock_ollama_class):
        """Test that generate delegates to provider client."""
        mock_instance = MagicMock()
        mock_instance.generate.return_value = LLMResponse(
            content="Response",
            model="test",
            provider="ollama",
        )
        mock_ollama_class.return_value = mock_instance

        client = LLMClient(provider="ollama")
        client.generate("Test prompt", temperature=0.5)

        mock_instance.generate.assert_called_with(
            prompt="Test prompt",
            system_prompt=None,
            temperature=0.5,
            max_tokens=None,
        )

    @patch("src.llm_client.OllamaClient")
    def test_llm_client_generate_stream_delegates(self, mock_ollama_class):
        """Test that generate_stream delegates to provider client."""
        mock_instance = MagicMock()
        mock_instance.generate_stream.return_value = iter(["Hello", " ", "world"])
        mock_ollama_class.return_value = mock_instance

        client = LLMClient(provider="ollama")
        chunks = list(client.generate_stream("Test prompt"))

        assert chunks == ["Hello", " ", "world"]

    @patch("src.llm_client.OllamaClient")
    def test_llm_client_is_available_delegates(self, mock_ollama_class):
        """Test that is_available delegates to provider client."""
        mock_instance = MagicMock()
        mock_instance.is_available.return_value = True
        mock_ollama_class.return_value = mock_instance

        client = LLMClient(provider="ollama")
        assert client.is_available() is True


class TestGetLLMClient:
    """Tests for get_llm_client factory function."""

    @patch("src.llm_client.OllamaClient")
    def test_get_llm_client_default(self, mock_ollama_class):
        """Test get_llm_client with default provider."""
        mock_ollama_class.return_value = MagicMock()

        client = get_llm_client()

        # Default is Ollama
        assert client.provider == "ollama"

    @patch("src.llm_client.ClaudeClient")
    def test_get_llm_client_claude(self, mock_claude_class):
        """Test get_llm_client with Claude provider."""
        mock_claude_class.return_value = MagicMock()

        client = get_llm_client(provider="claude")

        assert client.provider == "claude"
