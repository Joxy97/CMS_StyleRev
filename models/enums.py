"""
Enumerations for the Expert Annotation System.
"""

from enum import Enum, auto


class ExpertAction(Enum):
    """Actions an expert can take on suggestions."""
    PENDING = "pending"
    ACCEPT = "accept"
    REJECT = "reject"
    MODIFY = "modify"
    HUMAN_SUGGESTED = "human_suggested"  # For expert-created suggestions (frozen/accepted by default)
    IMPLEMENTED = "implemented"  # Applied to the paragraph and locked


class IssueType(Enum):
    """Categories of issues that can be identified."""
    GRAMMAR = "grammar"
    CMS_STYLE = "cms_style"
    PHYSICS_NOTATION = "physics_notation"
    CLARITY = "clarity"
    CITATION = "citation"
    FORMATTING = "formatting"
    OTHER = "other"


class IssueSeverity(Enum):
    """Severity levels for issues."""
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class ParagraphType(Enum):
    """Types of paragraphs in academic papers."""
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    METHODOLOGY = "methodology"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    BODY = "body"
    MATHEMATICAL = "mathematical"
    LITERATURE_REVIEW = "literature_review"


class Status(Enum):
    """Status of processing for paragraphs."""
    NOT_PROCESSED = "Not Processed"
    CHANGED = "Changed"
    DONE = "Done"
    ERROR = "Error"
