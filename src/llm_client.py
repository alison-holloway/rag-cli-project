"""LLM client module for RAG CLI.

Provides integration with Ollama (default, free) and optionally Claude.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import ollama

from .config import get_settings
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM."""

    content: str
    model: str
    provider: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    @property
    def token_count(self) -> int | None:
        """Get total token count if available."""
        return self.total_tokens


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Returns:
            LLMResponse with generated content.
        """
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Generate a streaming response from the LLM.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Yields:
            Text chunks as they are generated.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM service is available.

        Returns:
            True if the service is reachable.
        """
        pass


class OllamaClient(BaseLLMClient):
    """Client for Ollama local LLM service.

    Ollama provides free, local LLM inference. This is the default
    provider for RAG CLI.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ):
        """Initialize Ollama client.

        Args:
            model: Model name (e.g., 'llama3.1:8b'). Defaults to config.
            base_url: Ollama API URL. Defaults to config.
        """
        settings = get_settings()
        self.model = model or settings.llm.ollama_model
        self.base_url = base_url or settings.llm.ollama_base_url

        # Initialize Ollama client
        self._client = ollama.Client(host=self.base_url)

        logger.info(f"Ollama client initialized: model={self.model}")

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response using Ollama.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Maximum tokens in response.

        Returns:
            LLMResponse with generated content.
        """
        settings = get_settings()
        if temperature is None:
            temperature = settings.llm.llm_temperature
        max_tokens = max_tokens or settings.llm.max_tokens

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.debug(f"Generating response with Ollama ({self.model})")

        try:
            response = self._client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            )

            content = response["message"]["content"]

            # Extract token counts if available
            prompt_tokens = response.get("prompt_eval_count")
            completion_tokens = response.get("eval_count")
            total_tokens = None
            if prompt_tokens and completion_tokens:
                total_tokens = prompt_tokens + completion_tokens

            logger.debug(f"Generated {len(content)} characters")

            return LLMResponse(
                content=content,
                model=self.model,
                provider="ollama",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        except ollama.ResponseError as e:
            logger.error(f"Ollama error: {e}")
            raise RuntimeError(f"Ollama generation failed: {e}") from e

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Generate a streaming response using Ollama.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Yields:
            Text chunks as they are generated.
        """
        settings = get_settings()
        if temperature is None:
            temperature = settings.llm.llm_temperature
        max_tokens = max_tokens or settings.llm.max_tokens

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.debug(f"Streaming response with Ollama ({self.model})")

        try:
            stream = self._client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
                stream=True,
            )

            for chunk in stream:
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]

        except ollama.ResponseError as e:
            logger.error(f"Ollama streaming error: {e}")
            raise RuntimeError(f"Ollama streaming failed: {e}") from e

    def is_available(self) -> bool:
        """Check if Ollama service is available.

        Returns:
            True if Ollama is reachable and model is available.
        """
        try:
            # Try to list models to verify connection
            response = self._client.list()
            # Handle both object-based (new) and dict-based (old) responses
            if hasattr(response, "models"):
                model_names = [m.model for m in response.models]
            else:
                model_names = [m["name"] for m in response.get("models", [])]

            # Check if our model is available
            # Model names might include tags like 'llama3.1:8b'
            base_model = self.model.split(":")[0]
            available = any(base_model in name for name in model_names)

            if not available:
                logger.warning(
                    f"Model '{self.model}' not found. Available: {model_names}"
                )

            return available

        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False

    def list_models(self) -> list[str]:
        """List available Ollama models.

        Returns:
            List of model names.
        """
        try:
            response = self._client.list()
            # Handle both object-based (new) and dict-based (old) responses
            if hasattr(response, "models"):
                return [m.model for m in response.models]
            else:
                return [m["name"] for m in response.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def pull_model(self, model: str | None = None) -> bool:
        """Pull/download a model.

        Args:
            model: Model to pull. Defaults to configured model.

        Returns:
            True if successful.
        """
        model = model or self.model
        logger.info(f"Pulling model: {model}")

        try:
            self._client.pull(model)
            return True
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False


class ClaudeClient(BaseLLMClient):
    """Client for Anthropic Claude API.

    Optional paid provider for higher quality responses.
    Requires ANTHROPIC_API_KEY environment variable.
    """

    def __init__(
        self, api_key: str | None = None, model: str = "claude-sonnet-4-20250514"
    ):
        """Initialize Claude client.

        Args:
            api_key: Anthropic API key. Defaults to config/environment.
            model: Claude model to use.
        """
        settings = get_settings()
        self.api_key = api_key or settings.llm.anthropic_api_key
        self.model = model

        if not self.api_key:
            logger.warning("Claude API key not configured")
            self._client = None
        else:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.api_key)
                logger.info(f"Claude client initialized: model={self.model}")
            except ImportError:
                logger.warning(
                    "anthropic package not installed. "
                    "Install with: pip install anthropic"
                )
                self._client = None

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response using Claude.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Returns:
            LLMResponse with generated content.
        """
        if not self._client:
            raise RuntimeError(
                "Claude client not available. "
                "Set ANTHROPIC_API_KEY or install anthropic package."
            )

        settings = get_settings()
        if temperature is None:
            temperature = settings.llm.llm_temperature
        max_tokens = max_tokens or settings.llm.max_tokens

        logger.debug(f"Generating response with Claude ({self.model})")

        try:
            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }

            if system_prompt:
                kwargs["system"] = system_prompt

            if temperature is not None:
                kwargs["temperature"] = temperature

            response = self._client.messages.create(**kwargs)

            content = response.content[0].text

            return LLMResponse(
                content=content,
                model=self.model,
                provider="claude",
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            )

        except Exception as e:
            logger.error(f"Claude error: {e}")
            raise RuntimeError(f"Claude generation failed: {e}") from e

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Generate a streaming response using Claude.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Yields:
            Text chunks as they are generated.
        """
        if not self._client:
            raise RuntimeError("Claude client not available")

        settings = get_settings()
        if temperature is None:
            temperature = settings.llm.llm_temperature
        max_tokens = max_tokens or settings.llm.max_tokens

        logger.debug(f"Streaming response with Claude ({self.model})")

        try:
            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }

            if system_prompt:
                kwargs["system"] = system_prompt

            if temperature is not None:
                kwargs["temperature"] = temperature

            with self._client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error(f"Claude streaming error: {e}")
            raise RuntimeError(f"Claude streaming failed: {e}") from e

    def is_available(self) -> bool:
        """Check if Claude API is available.

        Returns:
            True if client is configured and API is reachable.
        """
        if not self._client:
            return False

        try:
            # Make a minimal API call to verify
            self._client.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True
        except Exception as e:
            logger.warning(f"Claude not available: {e}")
            return False


class LLMClient:
    """Unified LLM client that delegates to provider-specific clients.

    Supports automatic fallback and provider selection.
    """

    def __init__(self, provider: str | None = None):
        """Initialize LLM client.

        Args:
            provider: LLM provider ('ollama' or 'claude').
                     Defaults to config setting.
        """
        settings = get_settings()
        self.provider = provider or settings.llm.default_llm_provider

        # Initialize the appropriate client
        if self.provider == "ollama":
            self._client: BaseLLMClient = OllamaClient()
        elif self.provider == "claude":
            self._client = ClaudeClient()
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

        logger.info(f"LLM client using provider: {self.provider}")

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Returns:
            LLMResponse with generated content.
        """
        return self._client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Generate a streaming response.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Yields:
            Text chunks as they are generated.
        """
        yield from self._client.generate_stream(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def is_available(self) -> bool:
        """Check if the LLM service is available."""
        return self._client.is_available()


# Module-level factory
def get_llm_client(provider: str | None = None) -> LLMClient:
    """Get an LLM client for the specified provider.

    Args:
        provider: LLM provider. Defaults to config setting.

    Returns:
        Configured LLMClient instance.
    """
    return LLMClient(provider=provider)
