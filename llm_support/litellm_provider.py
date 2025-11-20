"""
LiteLLM Provider for unified multi-model LLM support.

This module provides a unified interface to multiple LLM providers
using LiteLLM, including Anthropic, OpenAI, Google, and Ollama.
"""

import os
import logging
import time
from typing import Dict, Any, Optional
from urllib import request as urllib_request, error as urllib_error

logger = logging.getLogger(__name__)

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    logger.warning("litellm not installed. Run: pip install litellm")


class LiteLLMProvider:
    """
    Unified LLM provider using LiteLLM.

    Supports:
    - Anthropic Claude models
    - OpenAI GPT models
    - Google Gemini models
    - Ollama local models
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: int = 3000,
        temperature: float = 0.0,
        timeout: int = 120,
        retry_attempts: int = 3,
        **kwargs
    ):
        """
        Initialize the LiteLLM provider.

        Args:
            model: Model identifier (e.g., "claude-sonnet-4-5-20250929", "gpt-4o", "ollama/llama3")
            api_key: API key for the provider
            api_base: Custom API base URL (for Ollama, etc.)
            max_tokens: Maximum tokens in response
            temperature: Temperature setting
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts
        """
        self.model = self._normalize_model_name(model)
        self.api_key = api_key
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.settings = kwargs

        # Set up environment variables for API keys
        self._setup_api_keys()

        # Ensure local models point to an Ollama endpoint
        if self.model.startswith("ollama"):
            self._configure_ollama_endpoint()

        logger.info(f"LiteLLM provider initialized with model: {self.model}")

    def _normalize_model_name(self, model: str) -> str:
        """
        Normalize model name to LiteLLM format.

        Auto-detects provider from model name prefix.
        LiteLLM format: "model" for cloud, "ollama/model" for local, "gemini/model" for Google

        Examples:
            "claude-sonnet-4-5-20250929" -> "claude-sonnet-4-5-20250929"
            "gpt-4o" -> "gpt-4o"
            "gemini-2.0-flash" -> "gemini/gemini-2.0-flash"
            "ollama/llama3" -> "ollama/llama3"
        """
        # Handle legacy format with colon (provider:model)
        if ":" in model and not model.startswith("ollama/"):
            provider, model_name = model.split(":", 1)
            provider = provider.lower()

            if provider in ["anthropic", "claude"]:
                return model_name
            elif provider == "openai":
                return model_name
            elif provider in ["google", "gemini"]:
                return f"gemini/{model_name}"
            elif provider == "ollama":
                return f"ollama/{model_name}"
            else:
                return model_name

        # Auto-detect provider from model name
        if model.startswith("ollama/"):
            return model  # Already in correct format
        # Claude and GPT models pass through as-is
        return model

    def _setup_api_keys(self):
        """Set up API keys in environment variables from settings or parameter."""
        # Import settings_manager to get API keys if not provided
        try:
            from models import settings_manager
        except ImportError:
            settings_manager = None

        # Set up API key based on model type
        if self.model.startswith("claude"):
            api_key = self.api_key or (settings_manager.anthropic_api_key if settings_manager else '')
            if api_key:
                self.api_key = api_key
                os.environ["ANTHROPIC_API_KEY"] = api_key
        elif self.model.startswith(("gpt", "o1", "o3", "o4")):
            api_key = self.api_key or (settings_manager.openai_api_key if settings_manager else '')
            if api_key:
                self.api_key = api_key
                os.environ["OPENAI_API_KEY"] = api_key

        # Set Ollama API base if provided
        if self.api_base and self.model.startswith("ollama"):
            os.environ["OLLAMA_API_BASE"] = self.api_base
            os.environ.setdefault("LITELLM_OLLAMA_API_BASE", self.api_base)
        elif self.model.startswith("ollama"):
            default_base = os.environ.get('OLLAMA_API_BASE') or 'http://localhost:11434'
            os.environ["OLLAMA_API_BASE"] = default_base
            os.environ.setdefault("LITELLM_OLLAMA_API_BASE", default_base)

    def _configure_ollama_endpoint(self):
        """Guarantee that LiteLLM receives a valid Ollama API base."""
        default_base = self.api_base or os.environ.get("OLLAMA_API_BASE") or "http://localhost:11434"
        # Normalize to avoid duplicate trailing slashes
        default_base = default_base.rstrip("/")
        self.api_base = default_base
        os.environ["OLLAMA_API_BASE"] = default_base
        os.environ.setdefault("LITELLM_OLLAMA_API_BASE", default_base)

    def is_available(self) -> bool:
        """Check if the provider is available."""
        if not LITELLM_AVAILABLE:
            return False

        # Check for required API keys
        if self.model.startswith("claude"):
            return bool(self.api_key)
        elif self.model.startswith(("gpt", "o1", "o3", "o4")):
            return bool(self.api_key)
        elif self.model.startswith("gemini"):
            return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or self.api_key)
        elif self.model.startswith("ollama"):
            # Ollama doesn't need API key, just needs the local server running
            return self._is_ollama_running()

        return True

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the LLM.

        Args:
            prompt: The prompt to send
            **kwargs: Override settings

        Returns:
            Generated text

        Raises:
            RuntimeError: If generation fails after all retries
        """
        if not LITELLM_AVAILABLE:
            raise RuntimeError("litellm not installed. Run: pip install litellm")

        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        temperature = kwargs.get('temperature', self.temperature)
        timeout = kwargs.get('timeout', self.timeout)

        # Adjust temperature for models that have restrictions
        adjusted_temperature = temperature
        if self.model.startswith("gpt-5") or self.model.startswith("o1") or self.model.startswith("o3"):
            # GPT-5 and reasoning models only support temperature=1
            adjusted_temperature = 1.0
            if temperature != 1.0:
                logger.debug(f"Adjusted temperature from {temperature} to 1.0 for {self.model}")

        # Prepare arguments
        llm_args = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": adjusted_temperature,
            "timeout": timeout
        }

        # Add API base if configured
        if self.api_base:
            llm_args["api_base"] = self.api_base

        # Apply Ollama JSON format enforcement by default
        extra_body = kwargs.get("extra_body")
        if self.model.startswith("ollama"):
            extra_body = dict(extra_body) if extra_body else {}
            extra_body.setdefault("format", "json")

        if extra_body:
            llm_args["extra_body"] = extra_body

        # Retry loop with exponential backoff
        last_error = None
        for attempt in range(self.retry_attempts + 1):
            try:
                logger.debug(f"LiteLLM call attempt {attempt + 1}/{self.retry_attempts + 1}")

                response = litellm.completion(**llm_args)

                # Extract response text
                content = response.choices[0].message.content
                if content:
                    return content.strip()
                else:
                    raise ValueError("Empty response from LLM")

            except Exception as e:
                last_error = e
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")

                if attempt < self.retry_attempts:
                    # Exponential backoff
                    sleep_time = 2 ** attempt
                    logger.debug(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)

        raise RuntimeError(f"LLM generation failed after {self.retry_attempts + 1} attempts: {last_error}")

    def _is_ollama_running(self) -> bool:
        """Ping the Ollama endpoint to ensure it is responding."""
        if not self.api_base:
            return False

        health_url = f"{self.api_base.rstrip('/')}/api/tags"
        try:
            with urllib_request.urlopen(health_url, timeout=3) as response:
                return response.status == 200
        except urllib_error.URLError as exc:
            logger.debug(f"Ollama endpoint unreachable at {health_url}: {exc}")
            return False
        except Exception as exc:
            logger.debug(f"Unexpected error checking Ollama endpoint: {exc}")
            return False

    def get_cost(self, response) -> float:
        """
        Get the cost of a response using LiteLLM's built-in function.

        Args:
            response: The completion response

        Returns:
            Cost in USD
        """
        if not LITELLM_AVAILABLE:
            return 0.0

        try:
            return litellm.completion_cost(completion_response=response)
        except Exception:
            return 0.0


class LiteLLMManager:
    """
    High-level manager for LLM operations with fallback support.

    Drop-in replacement for the original LLMManager.
    """

    def __init__(
        self,
        primary_model: str,
        fallback_model: str = None,
        api_key: Optional[str] = None,
        fallback_api_key: Optional[str] = None,
        **settings
    ):
        """
        Initialize the LLM manager.

        Args:
            primary_model: Primary model to use
            fallback_model: Fallback model if primary fails
            api_key: API key for primary model
            fallback_api_key: API key for fallback model
            **settings: Additional settings
        """
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.settings = settings

        # Create primary provider
        self.primary_provider = LiteLLMProvider(
            model=primary_model,
            api_key=api_key,
            **settings
        )

        # Create fallback provider if specified
        self.fallback_provider = None
        if fallback_model:
            self.fallback_provider = LiteLLMProvider(
                model=fallback_model,
                api_key=fallback_api_key or api_key,
                **settings
            )

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text with fallback support.

        Args:
            prompt: The prompt to send
            **kwargs: Override settings

        Returns:
            Generated text

        Raises:
            RuntimeError: If all providers fail
        """
        # Try primary provider
        if self.primary_provider.is_available():
            try:
                logger.info(f"Using primary model: {self.primary_model}")
                return self.primary_provider.generate(prompt, **kwargs)
            except Exception as e:
                logger.warning(f"Primary model failed: {e}")

        # Try fallback provider
        if self.fallback_provider and self.fallback_provider.is_available():
            try:
                logger.info(f"Using fallback model: {self.fallback_model}")
                return self.fallback_provider.generate(prompt, **kwargs)
            except Exception as e:
                logger.warning(f"Fallback model failed: {e}")

        raise RuntimeError("No LLM providers available")

    def get_available_providers(self) -> Dict[str, bool]:
        """Get status of all providers."""
        status = {
            "primary": self.primary_provider.is_available() if self.primary_provider else False
        }

        if self.fallback_provider:
            status["fallback"] = self.fallback_provider.is_available()

        return status


# Convenience function to check if LiteLLM is available
def check_litellm_available() -> bool:
    """Check if LiteLLM is installed and available."""
    return LITELLM_AVAILABLE


# Model name examples for reference
LITELLM_MODEL_EXAMPLES = {
    "Anthropic": [
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-20250514",
    ],
    "OpenAI": [
        "gpt-4o",
        "gpt-4o-mini",
        "o1-preview",
    ],
    "Google": [
        "gemini/gemini-2.0-flash",
        "gemini/gemini-1.5-pro",
    ],
    "Ollama": [
        "ollama/llama3",
        "ollama/mistral",
        "ollama/codellama",
    ]
}
