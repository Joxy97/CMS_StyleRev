"""
Paragraph Export Service.

Responsible for validating suggestion states and exporting the raw paragraph
texts from a project into a single `.tex` file for downstream use.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import List, Optional, Union

from models import Paragraph, Project, ExpertAction, PROJECT_ROOT

logger = logging.getLogger(__name__)


@dataclass
class ParagraphExportIssue:
    """Describes why a suggestion prevents export."""
    paragraph_id: str
    paragraph_label: str
    suggestion_id: str
    expert_action: ExpertAction


@dataclass
class ParagraphExportValidationResult:
    """Tracks outstanding suggestion issues before exporting."""
    pending_issues: List[ParagraphExportIssue] = field(default_factory=list)
    reviewed_unimplemented_issues: List[ParagraphExportIssue] = field(default_factory=list)

    @property
    def has_pending_issues(self) -> bool:
        return bool(self.pending_issues)

    @property
    def has_reviewed_unimplemented(self) -> bool:
        return bool(self.reviewed_unimplemented_issues)

    @property
    def has_issues(self) -> bool:
        return self.has_pending_issues or self.has_reviewed_unimplemented


class ParagraphExportServiceError(Exception):
    """Raised when validation fails before exporting."""

    def __init__(self, message: str, validation_result: Optional[ParagraphExportValidationResult] = None):
        super().__init__(message)
        self.validation_result = validation_result


class ParagraphExportService:
    """Exports concatenated paragraph.text values once all suggestions are resolved."""

    _REVIEWED_REQUIRES_IMPLEMENTATION = {
        ExpertAction.ACCEPT,
        ExpertAction.MODIFY,
        ExpertAction.HUMAN_SUGGESTED
    }

    def __init__(self, base_export_dir: Optional[Union[str, Path]] = None):
        self.base_export_dir = Path(base_export_dir) if base_export_dir is not None else PROJECT_ROOT / "exports"
        self.base_export_dir.mkdir(parents=True, exist_ok=True)

    def validate_export_ready(self, project: Project) -> ParagraphExportValidationResult:
        """
        Inspect every suggestion to ensure there are no pending or reviewed-but-unimplemented items.
        """
        result = ParagraphExportValidationResult()

        for paragraph in project.paragraphs:
            label = self._build_paragraph_label(paragraph)
            for suggestion in paragraph.suggestions:
                if suggestion.expert_action == ExpertAction.PENDING:
                    result.pending_issues.append(
                        ParagraphExportIssue(
                            paragraph_id=paragraph.id,
                            paragraph_label=label,
                            suggestion_id=suggestion.id,
                            expert_action=suggestion.expert_action
                        )
                    )
                    continue

                if suggestion.expert_action in self._REVIEWED_REQUIRES_IMPLEMENTATION:
                    if suggestion.id not in paragraph.implemented_suggestion_ids:
                        result.reviewed_unimplemented_issues.append(
                            ParagraphExportIssue(
                                paragraph_id=paragraph.id,
                                paragraph_label=label,
                                suggestion_id=suggestion.id,
                                expert_action=suggestion.expert_action
                            )
                        )

        return result

    def export_paragraphs(self, project: Project, output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Export all paragraph.text entries into a single .tex file.

        The service refuses to export when pending suggestions exist or actions requiring
        implementation have not yet been applied.
        """
        validation = self.validate_export_ready(project)
        if validation.has_issues:
            raise ParagraphExportServiceError(
                "Export blocked because there are outstanding suggestions.",
                validation_result=validation
            )

        destination = self._determine_output_path(project, output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        paragraphs = [para.text.strip() for para in project.paragraphs if para.text.strip()]
        export_text = "\n\n".join(paragraphs)
        if export_text and not export_text.endswith("\n"):
            export_text += "\n"

        destination.write_text(export_text, encoding="utf-8")
        logger.info("Exported %d paragraphs to %s", len(paragraphs), destination)
        return destination

    def _determine_output_path(self, project: Project, output_path: Optional[Union[str, Path]]) -> Path:
        if output_path:
            return Path(output_path)

        safe_name = self._sanitize_name(project.metadata.name or "project")
        filename = f"{safe_name}_{project.id[:8]}_paragraphs.tex"
        return self.base_export_dir / filename

    @staticmethod
    def _sanitize_name(name: str) -> str:
        unsafe_chars = '<>:"/\\|?*'
        sanitized = ''.join('_' if c in unsafe_chars else c for c in name)
        return sanitized[:50]

    @staticmethod
    def _build_paragraph_label(paragraph: Paragraph) -> str:
        label = paragraph.section_title or "section"
        if paragraph.subsection_title:
            label = f"{label} > {paragraph.subsection_title}"
        return label
