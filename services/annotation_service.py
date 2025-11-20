"""
Annotation Service.

This service handles the business logic for expert annotations,
including AI processing integration and expert decision management.
"""

import logging
from typing import List, Optional, Dict, Any, Callable

from models import (
    Project, Paragraph, Suggestion, ExpertAnnotation,
    ExpertAction, IssueType, IssueSeverity, Status,
    settings_manager
)
from services.single_stage_corrector import CMSSingleStageCorrector

logger = logging.getLogger(__name__)


class AnnotationServiceError(Exception):
    """Custom exception for annotation service errors."""
    pass


class AnnotationService:
    """
    Service for managing the annotation workflow.

    This service coordinates between AI processing and expert annotations,
    managing the complete workflow from initial AI suggestions to final
    expert-reviewed text.
    """

    def __init__(self, llm_model: str = None, rag_model: str = None):
        """
        Initialize the annotation service.

        Args:
            llm_model: LLM model identifier
            rag_model: RAG model identifier
        """
        self.llm_model = llm_model
        self.rag_model = rag_model
        self._corrector = None

    @property
    def corrector(self) -> CMSSingleStageCorrector:
        """Lazy-loaded corrector instance."""
        if self._corrector is None:
            self._corrector = CMSSingleStageCorrector(
                llm_model=self.llm_model,
                rag_model=self.rag_model
            )
        return self._corrector

    def update_models(self, llm_model: str = None, rag_model: str = None):
        """
        Update model settings and reset the corrector.

        This should be called when user changes settings to ensure
        the new models are used.

        Args:
            llm_model: New LLM model identifier
            rag_model: New RAG model identifier
        """
        if llm_model:
            self.llm_model = llm_model
        if rag_model:
            self.rag_model = rag_model

        # Reset corrector so it will be recreated with new settings
        self._corrector = None
        logger.info(f"Model settings updated - LLM: {self.llm_model}, RAG: {self.rag_model}")

    def process_paragraph_with_ai(self, paragraph: Paragraph, top_k: Optional[int] = None,
                                   rulebook=None) -> bool:
        """
        Process a single paragraph with AI to generate suggestions.

        Args:
            paragraph: The paragraph to process
            top_k: Number of top rules to consider (defaults to project settings)
            rulebook: Optional project-specific rulebook to use

        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            logger.info(f"Processing paragraph {paragraph.id} with AI")

            if top_k is None:
                top_k = settings_manager.top_k

            # Set the project rulebook if provided
            if rulebook is not None:
                self.corrector.set_rulebook(rulebook)

            # Use the existing corrector to get suggestions
            result = self.corrector.correct_paragraph(paragraph.text, top_k=top_k)

            # Clear existing AI suggestions only
            paragraph.suggestions = [s for s in paragraph.suggestions if s.source == 'human']

            # Convert result to our suggestion format
            if result.get('changed', False):
                edits = result.get('edits', [])
                for edit in edits:
                    try:
                        confidence = float(edit.get('confidence', 0.8))
                    except (TypeError, ValueError):
                        confidence = 0.8
                    confidence = max(0.0, min(confidence, 1.0))
                    suggestion = Suggestion.create(
                        original=edit.get('original', ''),
                        suggested=edit.get('corrected', ''),
                        rule_title=edit.get('rule_title', 'Unknown Rule'),
                        rule_content=edit.get('rule_content', ''),
                        confidence=confidence
                    )
                    paragraph.add_suggestion(suggestion)

            # Set status to CHANGED if suggestions were added
            if len(paragraph.suggestions) > 0:
                paragraph.status = Status.CHANGED
            logger.info(f"Generated {len([s for s in paragraph.suggestions if s.source == 'ai'])} suggestions for paragraph {paragraph.id}")
            return True

        except Exception as e:
            logger.error(f"Failed to process paragraph {paragraph.id}: {e}")
            paragraph.status = Status.ERROR
            return False

    def process_all_paragraphs_with_ai(self, project: Project, top_k: Optional[int] = None,
                                       progress_callback: Optional[Callable[[Paragraph, bool], None]] = None) -> Dict[str, int]:
        """
        Process all paragraphs in a project with AI.

        Args:
            project: The project to process
            top_k: Number of top rules to consider (defaults to project settings)

        Returns:
            Dict[str, int]: Statistics about the processing
        """
        stats = {
            'total': len(project.paragraphs),
            'processed': 0,
            'failed': 0,
            'suggestions_generated': 0
        }

        if top_k is None:
            top_k = settings_manager.top_k

        logger.info(f"Processing {stats['total']} paragraphs with AI")

        # Get the project's rulebook
        rulebook = project.rulebook if project.rulebook and project.rulebook.has_rules() else None

        for paragraph in project.paragraphs:
            success = self.process_paragraph_with_ai(paragraph, top_k, rulebook=rulebook)
            if success:
                stats['processed'] += 1
                stats['suggestions_generated'] += len(paragraph.suggestions)
            else:
                stats['failed'] += 1

            if progress_callback:
                try:
                    progress_callback(paragraph, success)
                except Exception as callback_error:
                    logger.warning(f"Progress callback failed: {callback_error}")

        logger.info(f"AI processing complete: {stats}")
        return stats

    def apply_expert_annotation(self, paragraph: Paragraph, suggestion_id: str,
                                action: ExpertAction, modified_text: Optional[str] = None,
                                notes: Optional[str] = None) -> bool:
        """
        Apply an expert annotation to a suggestion.

        Args:
            paragraph: The paragraph containing the suggestion
            suggestion_id: The ID of the suggestion to annotate
            action: The expert's action (accept, reject, modify)
            modified_text: Modified text if action is MODIFY
            notes: Optional expert notes

        Returns:
            bool: True if annotation was applied successfully
        """
        try:
            # Find the suggestion
            suggestion = None
            for s in paragraph.suggestions:
                if s.id == suggestion_id:
                    suggestion = s
                    break

            if not suggestion:
                logger.warning(f"Suggestion {suggestion_id} not found in paragraph {paragraph.id}")
                return False

            # Validate the annotation
            if action == ExpertAction.MODIFY and not modified_text:
                logger.warning("Modified text is required when action is MODIFY")
                return False

            # Update the suggestion directly
            suggestion.expert_action = action
            suggestion.modified_text = modified_text
            suggestion.expert_notes = notes

            # Also create the annotation for backwards compatibility and tracking
            annotation = ExpertAnnotation(
                suggestion_id=suggestion_id,
                expert_action=action,
                modified_text=modified_text,
                expert_notes=notes
            )
            paragraph.add_expert_annotation(annotation)

            logger.debug(f"Applied expert annotation: {action.value} for suggestion {suggestion_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to apply expert annotation: {e}")
            return False

    def add_expert_issue(self, paragraph: Paragraph, original_text: str, corrected_text: str,
                         issue_type: IssueType, severity: IssueSeverity,
                         notes: Optional[str] = None) -> bool:
        """
        Add an expert-identified issue as a human suggestion.

        Args:
            paragraph: The paragraph to add the issue to
            original_text: The original problematic text
            corrected_text: The corrected text
            issue_type: The type of issue
            severity: The severity of the issue
            notes: Optional expert notes

        Returns:
            bool: True if issue was added successfully
        """
        try:
            # Create a human suggestion instead of expert addition
            from models import Suggestion
            human_suggestion = Suggestion.create_human_suggestion(
                original_text=original_text,
                corrected_text=corrected_text,
                issue_type=issue_type,
                severity=severity,
                notes=notes
            )
            paragraph.add_suggestion(human_suggestion)

            logger.info(f"Added expert issue to paragraph {paragraph.id}: {issue_type.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to add expert issue: {e}")
            return False

    def get_annotation_statistics(self, project: Project) -> Dict[str, Any]:
        """
        Get comprehensive statistics about annotations in the project.

        Args:
            project: The project to analyze

        Returns:
            Dict[str, Any]: Detailed statistics
        """
        stats = {
            'total_paragraphs': len(project.paragraphs),
            'ai_processed_paragraphs': 0,
            'expert_reviewed_paragraphs': 0,
            'total_suggestions': 0,
            'ai_suggestions': 0,
            'human_suggestions': 0,
            'suggestions_by_action': {
                'pending': 0,
                'accepted': 0,
                'rejected': 0,
                'modified': 0,
                'human_suggested': 0,
                'implemented': 0
            },
            'human_suggestions_by_type': {},
            'human_suggestions_by_severity': {},
            'status_distribution': {
                'Not Processed': 0,
                'Changed': 0,
                'Done': 0,
                'Error': 0
            }
        }

        for paragraph in project.paragraphs:
            # Status statistics
            if paragraph.status == Status.DONE:
                stats['ai_processed_paragraphs'] += 1
            stats['status_distribution'][paragraph.status.value] += 1

            # Expert review statistics
            if paragraph.expert_annotations or any(s.source == 'human' for s in paragraph.suggestions):
                stats['expert_reviewed_paragraphs'] += 1

            # Suggestion statistics
            stats['total_suggestions'] += len(paragraph.suggestions)
            for suggestion in paragraph.suggestions:
                # Count by source
                if suggestion.source == 'human':
                    stats['human_suggestions'] += 1
                else:
                    stats['ai_suggestions'] += 1

                # Count by action
                stats['suggestions_by_action'][suggestion.expert_action.value] += 1

                # Human suggestion details
                if suggestion.source == 'human' and suggestion.issue_type and suggestion.severity:
                    # By type
                    type_key = suggestion.issue_type.value
                    stats['human_suggestions_by_type'][type_key] = \
                        stats['human_suggestions_by_type'].get(type_key, 0) + 1

                    # By severity
                    severity_key = suggestion.severity.value
                    stats['human_suggestions_by_severity'][severity_key] = \
                        stats['human_suggestions_by_severity'].get(severity_key, 0) + 1

        return stats

    def get_paragraph_completion_status(self, paragraph: Paragraph) -> Dict[str, Any]:
        """
        Get the completion status of a specific paragraph.

        Args:
            paragraph: The paragraph to analyze

        Returns:
            Dict[str, Any]: Completion status information
        """
        pending_suggestions = sum(
            1 for s in paragraph.suggestions
            if s.expert_action == ExpertAction.PENDING
        )

        human_suggestions = sum(
            1 for s in paragraph.suggestions
            if s.source == 'human'
        )

        implemented_suggestions = sum(
            1 for s in paragraph.suggestions
            if s.expert_action == ExpertAction.IMPLEMENTED
        )

        return {
            'ai_processed': paragraph.status == Status.DONE,
            'has_suggestions': len(paragraph.suggestions) > 0,
            'total_suggestions': len(paragraph.suggestions),
            'pending_suggestions': pending_suggestions,
            'completed_suggestions': len(paragraph.suggestions) - pending_suggestions,
            'implemented_suggestions': implemented_suggestions,
            'has_human_suggestions': human_suggestions > 0,
            'human_suggestions_count': human_suggestions,
            'is_fully_reviewed': (
                paragraph.status == Status.DONE and
                pending_suggestions == 0
            )
        }
