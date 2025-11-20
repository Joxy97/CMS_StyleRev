"""
Core data models for the Expert Annotation System.

These models represent the domain objects used throughout the application,
following clean architecture principles with immutable data structures
where possible.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import uuid

from .enums import ExpertAction, IssueType, IssueSeverity, ParagraphType, Status
from .rulebook_models import Rulebook


@dataclass
class Suggestion:
    """
    Represents a suggestion for text correction (AI or human-generated).

    This model now handles both AI suggestions and expert-identified issues,
    with expert_action tracking the current status of the suggestion.
    """
    id: str
    original: str
    suggested: str
    rule_title: str
    rule_content: str
    confidence: float
    source: str = "ai"  # "ai" or "human"
    expert_action: ExpertAction = ExpertAction.PENDING
    modified_text: Optional[str] = None  # For MODIFY action
    expert_notes: Optional[str] = None
    issue_type: Optional[IssueType] = None  # For human suggestions
    severity: Optional[IssueSeverity] = None  # For human suggestions
    timestamp: datetime = field(default_factory=datetime.now)
    implemented_start: Optional[int] = None
    implemented_end: Optional[int] = None
    implemented_replacement: Optional[str] = None
    pre_implementation_action: Optional[ExpertAction] = None
    implemented_original_occurrence_index: Optional[int] = None
    implemented_replacement_occurrence_index: Optional[int] = None

    @classmethod
    def create(cls, original: str, suggested: str, rule_title: str,
               rule_content: str, confidence: float, source: str = "ai") -> 'Suggestion':
        """Factory method to create a new suggestion with generated ID."""
        return cls(
            id=str(uuid.uuid4()),
            original=original,
            suggested=suggested,
            rule_title=rule_title,
            rule_content=rule_content,
            confidence=confidence,
            source=source
        )

    @classmethod
    def create_human_suggestion(cls, original_text: str, corrected_text: str,
                               issue_type: IssueType, severity: IssueSeverity,
                               notes: Optional[str] = None) -> 'Suggestion':
        """Factory method to create a human-identified suggestion."""
        return cls(
            id=str(uuid.uuid4()),
            original=original_text,
            suggested=corrected_text,
            rule_title=f"Expert Identified: {issue_type.value.replace('_', ' ').title()}",
            rule_content=notes or f"Expert-identified {issue_type.value.replace('_', ' ')} issue (severity: {severity.value})",
            confidence=1.0,  # Human suggestions have 100% confidence
            source="human",
            expert_action=ExpertAction.HUMAN_SUGGESTED,  # Human suggestions are frozen/accepted by default
            issue_type=issue_type,
            severity=severity,
            expert_notes=notes
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'original': self.original,
            'suggested': self.suggested,
            'rule_title': self.rule_title,
            'rule_content': self.rule_content,
            'confidence': self.confidence,
            'source': self.source,
            'expert_action': self.expert_action.value,
            'modified_text': self.modified_text,
            'expert_notes': self.expert_notes,
            'issue_type': self.issue_type.value if self.issue_type else None,
            'severity': self.severity.value if self.severity else None,
            'timestamp': self.timestamp.isoformat(),
            'implemented_start': self.implemented_start,
            'implemented_end': self.implemented_end,
            'implemented_replacement': self.implemented_replacement,
            'pre_implementation_action': self.pre_implementation_action.value if self.pre_implementation_action else None,
            'implemented_original_occurrence_index': self.implemented_original_occurrence_index,
            'implemented_replacement_occurrence_index': self.implemented_replacement_occurrence_index
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Suggestion':
        """Create from dictionary (JSON deserialization)."""
        return cls(
            id=data['id'],
            original=data['original'],
            suggested=data['suggested'],
            rule_title=data['rule_title'],
            rule_content=data['rule_content'],
            confidence=data['confidence'],
            source=data.get('source', 'ai'),
            expert_action=ExpertAction(data.get('expert_action', 'pending')),
            modified_text=data.get('modified_text'),
            expert_notes=data.get('expert_notes'),
            issue_type=IssueType(data['issue_type']) if data.get('issue_type') else None,
            severity=IssueSeverity(data['severity']) if data.get('severity') else None,
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now(),
            implemented_start=data.get('implemented_start'),
            implemented_end=data.get('implemented_end'),
            implemented_replacement=data.get('implemented_replacement'),
            pre_implementation_action=ExpertAction(data['pre_implementation_action']) if data.get('pre_implementation_action') else None,
            implemented_original_occurrence_index=data.get('implemented_original_occurrence_index'),
            implemented_replacement_occurrence_index=data.get('implemented_replacement_occurrence_index')
        )


@dataclass
class ExpertAnnotation:
    """
    Represents an expert's decision on an AI suggestion.

    This is mutable as experts can change their minds during annotation.
    """
    suggestion_id: str
    expert_action: ExpertAction
    modified_text: Optional[str] = None
    expert_notes: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'suggestion_id': self.suggestion_id,
            'expert_action': self.expert_action.value,
            'modified_text': self.modified_text,
            'expert_notes': self.expert_notes,
            'timestamp': self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExpertAnnotation':
        """Create from dictionary (JSON deserialization)."""
        return cls(
            suggestion_id=data['suggestion_id'],
            expert_action=ExpertAction(data['expert_action']),
            modified_text=data.get('modified_text'),
            expert_notes=data.get('expert_notes'),
            timestamp=datetime.fromisoformat(data['timestamp'])
        )


@dataclass
class Paragraph:
    """
    Represents a paragraph from the document with its associated suggestions and annotations.

    This is the main entity that experts work with during annotation.
    """
    id: str
    text: str
    section_title: str
    section_number: int
    subsection_title: Optional[str] = None
    subsection_number: Optional[int] = None
    paragraph_type: ParagraphType = ParagraphType.BODY
    status: Status = Status.NOT_PROCESSED

    # Collections (mutable)
    suggestions: List[Suggestion] = field(default_factory=list)
    expert_annotations: List[ExpertAnnotation] = field(default_factory=list)
    implemented_suggestion_ids: List[str] = field(default_factory=list)

    @classmethod
    def create(cls, text: str, section_title: str, section_number: int,
               subsection_title: Optional[str] = None,
               subsection_number: Optional[int] = None,
               paragraph_type: ParagraphType = ParagraphType.BODY) -> 'Paragraph':
        """Factory method to create a new paragraph with generated ID."""
        return cls(
            id=str(uuid.uuid4()),
            text=text,
            section_title=section_title,
            section_number=section_number,
            subsection_title=subsection_title,
            subsection_number=subsection_number,
            paragraph_type=paragraph_type
        )

    def add_suggestion(self, suggestion: Suggestion) -> None:
        """Add an AI suggestion to this paragraph."""
        self.suggestions.append(suggestion)

    def add_expert_annotation(self, annotation: ExpertAnnotation) -> None:
        """Add an expert annotation to this paragraph."""
        # Remove any existing annotation for the same suggestion
        self.expert_annotations = [
            a for a in self.expert_annotations
            if a.suggestion_id != annotation.suggestion_id
        ]
        self.expert_annotations.append(annotation)


    def get_annotation_for_suggestion(self, suggestion_id: str) -> Optional[ExpertAnnotation]:
        """Get the expert annotation for a specific suggestion."""
        for annotation in self.expert_annotations:
            if annotation.suggestion_id == suggestion_id:
                return annotation
        return None

    def get_final_text(self) -> str:
        """
        Generate the final text with all accepted changes applied.

        This applies all expert decisions in the correct order.
        """
        final_text = self.text

        # Apply accepted suggestions (both AI and human)
        for suggestion in self.suggestions:
            if suggestion.expert_action == ExpertAction.ACCEPT or suggestion.expert_action == ExpertAction.HUMAN_SUGGESTED:
                final_text = final_text.replace(suggestion.original, suggestion.suggested)
            elif suggestion.expert_action == ExpertAction.MODIFY and suggestion.modified_text:
                final_text = final_text.replace(suggestion.original, suggestion.modified_text)
            else:
                # Check for explicit annotation (for backwards compatibility)
                annotation = self.get_annotation_for_suggestion(suggestion.id)
                if annotation:
                    if annotation.expert_action == ExpertAction.ACCEPT:
                        final_text = final_text.replace(suggestion.original, suggestion.suggested)
                    elif annotation.expert_action == ExpertAction.MODIFY and annotation.modified_text:
                        final_text = final_text.replace(suggestion.original, annotation.modified_text)

        return final_text

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'text': self.text,
            'section_title': self.section_title,
            'section_number': self.section_number,
            'subsection_title': self.subsection_title,
            'subsection_number': self.subsection_number,
            'paragraph_type': self.paragraph_type.value,
            'status': self.status.value,
            'suggestions': [s.to_dict() for s in self.suggestions],
            'expert_annotations': [a.to_dict() for a in self.expert_annotations],
            'implemented_suggestion_ids': self.implemented_suggestion_ids
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Paragraph':
        """Create from dictionary (JSON deserialization)."""
        # Handle backward compatibility for ai_status -> status
        status_value = data.get('status') or data.get('ai_status', 'Not Processed')
        # Map old values to new values
        status_mapping = {
            'not_processed': 'Not Processed',
            'processing': 'Changed',  # Old "processing" becomes "Changed"
            'completed': 'Not Processed',  # Old "completed" becomes "Not Processed" (user needs to review)
            'error': 'Error'
        }
        if status_value in status_mapping:
            status_value = status_mapping[status_value]

        paragraph = cls(
            id=data['id'],
            text=data['text'],
            section_title=data['section_title'],
            section_number=data['section_number'],
            subsection_title=data.get('subsection_title'),
            subsection_number=data.get('subsection_number'),
            paragraph_type=ParagraphType(data.get('paragraph_type', 'body')),
            status=Status(status_value)
        )

        # Load suggestions
        paragraph.suggestions = [
            Suggestion.from_dict(s) for s in data.get('suggestions', [])
        ]

        # Load expert annotations
        paragraph.expert_annotations = [
            ExpertAnnotation.from_dict(a) for a in data.get('expert_annotations', [])
        ]

        # Track implemented suggestions (optional field)
        paragraph.implemented_suggestion_ids = data.get('implemented_suggestion_ids', [])
        if not paragraph.implemented_suggestion_ids:
            paragraph.implemented_suggestion_ids = [
                s.id for s in paragraph.suggestions
                if s.expert_action == ExpertAction.IMPLEMENTED
            ]

        # Backward compatibility: Convert old expert_additions to human suggestions
        if 'expert_additions' in data and data['expert_additions']:
            for addition_data in data['expert_additions']:
                # Convert old ExpertAddition to new Suggestion format
                human_suggestion = cls._convert_expert_addition_to_suggestion(addition_data)
                paragraph.suggestions.append(human_suggestion)

        return paragraph

    @staticmethod
    def _convert_expert_addition_to_suggestion(addition_data: Dict[str, Any]) -> 'Suggestion':
        """Convert old ExpertAddition data to new Suggestion format (for backward compatibility)."""
        from datetime import datetime

        issue_type = IssueType(addition_data['issue_type'])
        severity = IssueSeverity(addition_data['severity'])

        return Suggestion(
            id=addition_data['id'],
            original=addition_data['original_text'],
            suggested=addition_data['corrected_text'],
            rule_title=f"Expert Identified: {issue_type.value.replace('_', ' ').title()}",
            rule_content=addition_data.get('notes') or f"Expert-identified {issue_type.value.replace('_', ' ')} issue (severity: {severity.value})",
            confidence=1.0,
            source="human",
            expert_action=ExpertAction.HUMAN_SUGGESTED,
            modified_text=None,
            expert_notes=addition_data.get('notes'),
            issue_type=issue_type,
            severity=severity,
            timestamp=datetime.fromisoformat(addition_data['timestamp']) if 'timestamp' in addition_data else datetime.now()
        )


@dataclass
class ProjectMetadata:
    """
    Metadata about a project.
    """
    name: str
    created_at: datetime
    last_modified: datetime
    latex_file_path: str
    total_paragraphs: int = 0
    processed_paragraphs: int = 0
    expert_reviewed_paragraphs: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'created_at': self.created_at.isoformat(),
            'last_modified': self.last_modified.isoformat(),
            'latex_file_path': self.latex_file_path,
            'total_paragraphs': self.total_paragraphs,
            'processed_paragraphs': self.processed_paragraphs,
            'expert_reviewed_paragraphs': self.expert_reviewed_paragraphs
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectMetadata':
        """Create from dictionary (JSON deserialization)."""
        return cls(
            name=data['name'],
            created_at=datetime.fromisoformat(data['created_at']),
            last_modified=datetime.fromisoformat(data['last_modified']),
            latex_file_path=data['latex_file_path'],
            total_paragraphs=data.get('total_paragraphs', 0),
            processed_paragraphs=data.get('processed_paragraphs', 0),
            expert_reviewed_paragraphs=data.get('expert_reviewed_paragraphs', 0)
        )


@dataclass
class Project:
    """
    Root aggregate representing a complete annotation project.

    This contains all the data for a single document annotation session.
    """
    id: str
    metadata: ProjectMetadata
    paragraphs: List[Paragraph] = field(default_factory=list)
    rulebook: Optional[Rulebook] = None

    @classmethod
    def create(cls, name: str, latex_file_path: str) -> 'Project':
        """Factory method to create a new project with generated ID."""
        now = datetime.now()
        return cls(
            id=str(uuid.uuid4()),
            metadata=ProjectMetadata(
                name=name,
                created_at=now,
                last_modified=now,
                latex_file_path=latex_file_path
            ),
            rulebook=Rulebook()
        )

    def add_paragraph(self, paragraph: Paragraph) -> None:
        """Add a paragraph to this project."""
        self.paragraphs.append(paragraph)
        self.metadata.total_paragraphs = len(self.paragraphs)
        self.metadata.last_modified = datetime.now()

    def get_paragraph_by_id(self, paragraph_id: str) -> Optional[Paragraph]:
        """Get a paragraph by its ID."""
        for paragraph in self.paragraphs:
            if paragraph.id == paragraph_id:
                return paragraph
        return None

    def update_progress_stats(self) -> None:
        """Update the project metadata with current progress statistics."""
        self.metadata.processed_paragraphs = sum(
            1 for p in self.paragraphs
            if p.status == Status.DONE
        )

        self.metadata.expert_reviewed_paragraphs = sum(
            1 for p in self.paragraphs
            if p.expert_annotations or any(s.source == 'human' for s in p.suggestions)
        )

        self.metadata.last_modified = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'id': self.id,
            'metadata': self.metadata.to_dict(),
            'paragraphs': [p.to_dict() for p in self.paragraphs]
        }

        # Include rulebook if present
        if self.rulebook is not None:
            result['rulebook'] = self.rulebook.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Project':
        """Create from dictionary (JSON deserialization)."""
        project = cls(
            id=data['id'],
            metadata=ProjectMetadata.from_dict(data['metadata'])
        )

        # Load paragraphs
        project.paragraphs = [
            Paragraph.from_dict(p) for p in data.get('paragraphs', [])
        ]

        # Load rulebook if present
        if 'rulebook' in data and data['rulebook']:
            project.rulebook = Rulebook.from_dict(data['rulebook'])
        else:
            project.rulebook = Rulebook()

        return project
