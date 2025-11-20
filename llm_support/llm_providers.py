"""
Multi-provider LLM system supporting Anthropic, OpenAI, Gemini, Groq, and HuggingFace models.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import warnings

from models import settings_manager

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.settings = kwargs
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the LLM."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available (API key, dependencies, etc.)."""
        pass


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""
    
    def __init__(self, model_name: str, api_key: str = None, **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or settings_manager.anthropic_api_key
        self.client = None
        
        if self.api_key:
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
            except ImportError:
                logging.error("anthropic package not installed. Run: pip install anthropic")
    
    def is_available(self) -> bool:
        return self.client is not None and self.api_key is not None
    
    def generate(self, prompt: str, **kwargs) -> str:
        if not self.is_available():
            raise RuntimeError("Anthropic client not available")
        
        max_tokens = kwargs.get('max_tokens', self.settings.get('max_tokens', 2000))
        temperature = kwargs.get('temperature', self.settings.get('temperature', 0))
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            logging.error(f"Anthropic API error: {e}")
            raise


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""
    
    def __init__(self, model_name: str, api_key: str = None, **kwargs):
        super().__init__(model_name, **kwargs)
        # Try to get API key from parameter or settings manager
        self.api_key = api_key or settings_manager.openai_api_key
        self.client = None
        
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                logging.error("openai package not installed. Run: pip install openai")
    
    def is_available(self) -> bool:
        return self.client is not None and self.api_key is not None
    
    def generate(self, prompt: str, **kwargs) -> str:
        if not self.is_available():
            raise RuntimeError("OpenAI client not available")
        
        max_tokens = kwargs.get('max_tokens', self.settings.get('max_tokens', 2000))
        temperature = kwargs.get('temperature', self.settings.get('temperature', 0))
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            raise





class LLMFactory:
    """Factory class to create LLM providers."""
    
    @staticmethod
    def create_provider(model_config: str, **kwargs) -> LLMProvider:
        """
        Create an LLM provider from model configuration string.
        
        Args:
            model_config: Format "provider:model_name" or just "model_name" for anthropic
            **kwargs: Additional settings
            
        Returns:
            LLMProvider instance
        """
        # Handle legacy format with colon
        if ":" in model_config:
            provider, model_name = model_config.split(":", 1)
            provider = provider.lower()

            if provider == "anthropic":
                return AnthropicProvider(model_name, **kwargs)
            elif provider == "openai":
                return OpenAIProvider(model_name, **kwargs)
            else:
                raise ValueError(f"Unknown provider: {provider}")

        # Auto-detect provider from model name
        model_lower = model_config.lower()
        if model_lower.startswith("claude"):
            return AnthropicProvider(model_config, **kwargs)
        elif model_lower.startswith(("gpt", "o1", "o3", "o4")):
            return OpenAIProvider(model_config, **kwargs)
        else:
            # Default to anthropic for backward compatibility
            return AnthropicProvider(model_config, **kwargs)


class LLMManager:
    """High-level manager for LLM operations with fallback support."""

    def __init__(self, primary_model: str, fallback_model: str = None, **settings):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.settings = settings

        # Try to use LiteLLM if available
        try:
            from .litellm_provider import LiteLLMManager, check_litellm_available
            if check_litellm_available():
                logging.info("Using LiteLLM for unified provider support")
                self._use_litellm = True
                self._litellm_manager = LiteLLMManager(
                    primary_model=primary_model,
                    fallback_model=fallback_model,
                    api_key=settings.get('api_key'),
                    fallback_api_key=settings.get('fallback_api_key'),
                    max_tokens=settings.get('max_tokens', 3000),
                    temperature=settings.get('temperature', 0.0),
                    timeout=settings.get('timeout', 120)
                )
                self.primary_provider = self._litellm_manager.primary_provider
                self.fallback_provider = self._litellm_manager.fallback_provider
                return
        except ImportError:
            pass

        # Fallback to original providers
        logging.info("Using original provider implementation")
        self._use_litellm = False
        self.primary_provider = LLMFactory.create_provider(primary_model, **settings)
        self.fallback_provider = None

        if fallback_model:
            self.fallback_provider = LLMFactory.create_provider(fallback_model, **settings)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text with fallback support."""

        # Use LiteLLM manager if available
        if hasattr(self, '_use_litellm') and self._use_litellm:
            return self._litellm_manager.generate(prompt, **kwargs)

        # Original implementation
        # Try primary provider
        if self.primary_provider.is_available():
            try:
                logging.info(f"Using primary model: {self.primary_model}")
                return self.primary_provider.generate(prompt, **kwargs)
            except Exception as e:
                logging.warning(f"Primary model failed: {e}")

        # Try fallback provider
        if self.fallback_provider and self.fallback_provider.is_available():
            try:
                logging.info(f"Using fallback model: {self.fallback_model}")
                return self.fallback_provider.generate(prompt, **kwargs)
            except Exception as e:
                logging.warning(f"Fallback model failed: {e}")

        # No providers available
        raise RuntimeError("No LLM providers available")

    def get_available_providers(self) -> Dict[str, bool]:
        """Get status of all providers."""
        if hasattr(self, '_use_litellm') and self._use_litellm:
            return self._litellm_manager.get_available_providers()

        status = {
            "primary": self.primary_provider.is_available() if self.primary_provider else False
        }

        if self.fallback_provider:
            status["fallback"] = self.fallback_provider.is_available()

        return status


# Installation helper
def check_dependencies():
    """Check and report on available dependencies."""
    dependencies = {
        "anthropic": "pip install anthropic",
        "openai": "pip install openai",
        "litellm": "pip install litellm"
    }
    
    available = {}
    for package, install_cmd in dependencies.items():
        try:
            __import__(package)
            available[package] = True
        except ImportError:
            available[package] = False
            print(f"❌ {package} not available - {install_cmd}")
    
    print(f"✅ Available providers: {[k for k, v in available.items() if v]}")
    return available


if __name__ == "__main__":
    # Test the system
    check_dependencies()
    
    # Example usage
    manager = LLMManager(
        primary_model="anthropic:claude-3-5-sonnet-20241022",
        fallback_model="anthropic:claude-3-5-haiku-20241022"
    )
    
    print(f"Provider status: {manager.get_available_providers()}")
