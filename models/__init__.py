"""
Data models for the StyleRev CMS Expert Annotation System.

This module provides the core data structures used throughout the application,
following the Model-View architecture pattern.
"""

from .annotation_models import (
    Suggestion,
    ExpertAnnotation,
    Paragraph,
    Project,
    ProjectMetadata
)

from .enums import (
    ExpertAction,
    IssueType,
    IssueSeverity,
    ParagraphType,
    Status
)

from .settings_models import (
    ModelSettings,
    APIKeys,
    ProcessingSettings,
    PathSettings,
    ApplicationSettings,
    LLM_MODEL_OPTIONS,
    RAG_MODEL_OPTIONS,
    DEVICE_OPTIONS,
    LLM_PROVIDERS,
    CLOUD_MODELS,
    OLLAMA_MODELS,
    OLLAMA_MODEL_LIST,
    get_ollama_model_size,
    SettingsManager,
    settings_manager,
    PROJECT_ROOT
)

from .rule_models import (
    Rule,
    RulePriority
)

from .rulebook_models import (
    Rulebook
)

__all__ = [
    'Suggestion',
    'ExpertAnnotation',
    'Paragraph',
    'Project',
    'ProjectMetadata',
    'ExpertAction',
    'IssueType',
    'IssueSeverity',
    'ParagraphType',
    'Status',
    'ModelSettings',
    'APIKeys',
    'ProcessingSettings',
    'PathSettings',
    'ApplicationSettings',
    'LLM_MODEL_OPTIONS',
    'RAG_MODEL_OPTIONS',
    'DEVICE_OPTIONS',
    'LLM_PROVIDERS',
    'CLOUD_MODELS',
    'OLLAMA_MODELS',
    'OLLAMA_MODEL_LIST',
    'get_ollama_model_size',
    'SettingsManager',
    'settings_manager',
    'PROJECT_ROOT',
    'Rule',
    'RulePriority',
    'Rulebook'
]