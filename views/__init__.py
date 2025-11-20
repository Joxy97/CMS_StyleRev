"""
View components for the StyleRev CMS Expert Annotation System.

This module contains all GUI components following the Model-View architecture.
"""

from .paragraph_list import ParagraphListView
from .annotation_panel import AnnotationPanel
from .settings_dialog import SettingsDialog

__all__ = [
    'ParagraphListView',
    'AnnotationPanel',
    'SettingsDialog'
]