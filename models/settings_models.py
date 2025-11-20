"""
Settings models for StyleRev CMS configuration.

This module provides data classes for managing application and project-specific settings.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from pathlib import Path
import json


# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()


@dataclass
class PathSettings:
    """Path configuration settings."""

    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    rules_json: Path = field(default_factory=lambda: PROJECT_ROOT / "resources" / "guidelines.json")
    cache_dir: Path = field(default_factory=lambda: PROJECT_ROOT / ".cache" / "huggingface_cache")
    output_dir: str = "projects"


@dataclass
class ModelSettings:
    """LLM and RAG model configuration settings."""

    # LLM Configuration
    llm_model: str = "claude-sonnet-4-5-20250929"

    # RAG Configuration
    rag_model: str = "BAAI/bge-base-en-v1.5"
    top_k_rules: int = 10

    # Model Parameters
    max_tokens: int = 3000
    temperature: float = 0.0
    timeout: int = 120

    # HuggingFace Settings
    huggingface_device: str = "auto"
    enable_local_models: bool = True
    use_gpu_if_available: bool = True


@dataclass
class APIKeys:
    """API key configuration."""

    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    def mask_keys(self) -> Dict[str, str]:
        """Return masked version of API keys for display."""
        def mask_key(key: Optional[str]) -> str:
            if not key or len(key) < 8:
                return "Not set"
            return f"{key[:4]}...{key[-4:]}"

        return {
            'anthropic': mask_key(self.anthropic_api_key),
            'openai': mask_key(self.openai_api_key),
            'google': mask_key(self.google_api_key)
        }
           

@dataclass
class ProcessingSettings:
    """Processing and performance settings."""

    # Processing Options
    rebuild_db: bool = False
    parallel_processing: bool = False
    max_concurrent_corrections: int = 1

    # Performance Options
    cache_embeddings: bool = True
    embedding_batch_size: int = 32

    # Output Options
    generate_pdf: bool = True
    generate_corrected: bool = False
    auto_save: bool = True


@dataclass
class ApplicationSettings:
    """Complete application settings configuration."""

    model_settings: ModelSettings = field(default_factory=ModelSettings)
    api_keys: APIKeys = field(default_factory=APIKeys)
    processing_settings: ProcessingSettings = field(default_factory=ProcessingSettings)
    path_settings: PathSettings = field(default_factory=PathSettings)

    # Metadata
    settings_version: str = "1.0"
    is_project_specific: bool = False

    def to_dict(self, include_api_keys: bool = True) -> Dict[str, Any]:
        """
        Convert settings to dictionary.

        Args:
            include_api_keys: Whether to include API keys in output
        """
        result = {
            'model_settings': asdict(self.model_settings),
            'processing_settings': asdict(self.processing_settings),
            'settings_version': self.settings_version,
            'is_project_specific': self.is_project_specific
        }

        if include_api_keys:
            result['api_keys'] = asdict(self.api_keys)

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ApplicationSettings':
        """Create settings from dictionary."""
        # Handle backward compatibility for api_keys
        api_keys_data = data.get('api_keys', {})
        # Remove old keys that are no longer supported
        api_keys_data.pop('groq_api_key', None)
        api_keys_data.pop('huggingface_token', None)
        api_keys_data.pop('google_api_key', None)

        model_settings_data = data.get('model_settings', {}).copy()
        model_settings_data.pop('fallback_model', None)

        return cls(
            model_settings=ModelSettings(**model_settings_data),
            api_keys=APIKeys(**api_keys_data),
            processing_settings=ProcessingSettings(**data.get('processing_settings', {})),
            settings_version=data.get('settings_version', '1.0'),
            is_project_specific=data.get('is_project_specific', False)
        )

    def save_to_file(self, file_path: Path, include_api_keys: bool = True):
        """Save settings to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(include_api_keys=include_api_keys), f, indent=2)

    @classmethod
    def load_from_file(cls, file_path: Path) -> 'ApplicationSettings':
        """Load settings from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

# Provider types
LLM_PROVIDERS = ["Anthropic", "OpenAI", "Google", "Local (Ollama)"]

# Cloud API models organized by provider
CLOUD_MODELS = {
    "Anthropic": [
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-1-20250805",
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
    ],
    "OpenAI": [
        "gpt-5.1-2025-11-13",
        "gpt-5-2025-08-07",
        "gpt-5-pro-2025-10-06",
        "gpt-5-mini-2025-08-07",
        "gpt-5-nano-2025-08-07",
        "gpt-4.1-2025-04-14",
        "gpt-4.1-mini-2025-04-14",
        "gpt-4.1-nano-2025-04-14",
    ],
    "Google": [
        "gemini-3-pro-preview",
        "gemini-2.5-pro",
        "gemini-2.5-pro-preview-tts",
        "gemini-2.5-flash",
        "gemini-2.5-flash-preview-09-2025",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash-lite-preview-09-2025",
        "gemini-2.0-flash",
    ],
}

# Local Ollama models organized by family with size info (in GB)
OLLAMA_MODELS = {
    "Meta Llama": [
        ("llama3.1:405b", 229.0),
        ("llama3.1:70b", 40.0),
        ("llama3.1:8b", 4.7),
        ("llama3.2:3b", 2.0),
        ("llama3.2:1b", 1.3),
    ],
    "Alibaba Qwen": [
        ("qwen2.5:72b", 41.0),
        ("qwen2.5:32b", 18.0),
        ("qwen2.5:14b", 8.0),
        ("qwen2.5:7b", 4.4),
        ("qwen2.5:3b", 2.0),
        ("qwen2.5:1.5b-instruct", 1.0),
        ("qwen2.5:0.5b-instruct", 0.4),
    ],
    "DeepSeek Reasoning": [
        ("deepseek-r1:671b", 400.0),
        ("deepseek-r1:70b", 42.0),
        ("deepseek-r1:32b", 20.0),
        ("deepseek-r1:14b", 9.0),
        ("deepseek-r1:8b", 4.9),
        ("deepseek-r1:7b", 4.7),
        ("deepseek-r1:1.5b", 1.1),
    ],
    "Google Gemma": [
        ("gemma2:27b", 16.0),
        ("gemma2:9b", 5.4),
        ("gemma2:2b", 1.6),
    ],
    "Microsoft Phi": [
        ("phi3:14b", 7.9),
        ("phi3:3.8b-instruct", 2.2),
        ("phi3:3.8b", 2.2),
    ],
    "Mistral AI": [
        ("mixtral:8x22b", 80.0),
        ("mixtral:8x7b", 26.0),
        ("mistral-large:latest", 69.0),
        ("mistral-nemo:latest", 7.1),
        ("mistral:7b-instruct-v0.3", 4.1),
        ("mistral:7b-instruct-v0.2", 4.1),
        ("mistral:7b-instruct", 4.1),
        ("mistral:7b", 4.1),
    ],
}

# Flat list of all Ollama model names for convenience
OLLAMA_MODEL_LIST = [model for family in OLLAMA_MODELS.values() for model, _ in family]

# Get model size info
def get_ollama_model_size(model_name: str) -> float:
    """Get the size in GB for an Ollama model."""
    for family in OLLAMA_MODELS.values():
        for model, size in family:
            if model == model_name:
                return size
    return 0.0

# Legacy flat list for backward compatibility
LLM_MODEL_OPTIONS = (
    CLOUD_MODELS["Anthropic"] +
    CLOUD_MODELS["OpenAI"] +
    CLOUD_MODELS["Google"] +
    [f"ollama/{model}" for model in OLLAMA_MODEL_LIST]
)

RAG_MODEL_OPTIONS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-base-en-v1.5",
    "allenai/specter",
]

DEVICE_OPTIONS = ["auto", "cpu", "cuda", "mps"]


class SettingsManager:
    """
    Global settings manager for the application.

    Provides a centralized way to access and update settings
    without needing to import config.py.
    """

    _instance = None
    _settings: ApplicationSettings = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._settings = ApplicationSettings()
        return cls._instance

    @property
    def settings(self) -> ApplicationSettings:
        return self._settings

    def update_settings(self, new_settings: ApplicationSettings):
        """Update the global settings."""
        self._settings = new_settings

    # Convenience properties
    @property
    def llm_model(self) -> str:
        return self._settings.model_settings.llm_model

    @property
    def rag_model(self) -> str:
        return self._settings.model_settings.rag_model

    @property
    def anthropic_api_key(self) -> str:
        return self._settings.api_keys.anthropic_api_key or ''

    @property
    def openai_api_key(self) -> str:
        return self._settings.api_keys.openai_api_key or ''

    @property
    def rules_json(self) -> Path:
        return self._settings.path_settings.rules_json

    @property
    def cache_dir(self) -> Path:
        return self._settings.path_settings.cache_dir

    @property
    def max_tokens(self) -> int:
        return self._settings.model_settings.max_tokens

    @property
    def temperature(self) -> float:
        return self._settings.model_settings.temperature

    @property
    def timeout(self) -> int:
        return self._settings.model_settings.timeout

    @property
    def top_k(self) -> int:
        return self._settings.model_settings.top_k_rules

    @property
    def device(self) -> str:
        return self._settings.model_settings.huggingface_device


# Global settings manager instance
settings_manager = SettingsManager()
