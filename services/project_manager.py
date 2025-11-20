"""
Project Management Service.

This service handles all file operations related to projects, including:
- Creating and loading projects
- Saving and loading project data
- Managing project directory structure
- Exporting final documents
"""

import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from models import Project, Paragraph, ParagraphType, Status, ApplicationSettings
from parsers.physics_latex_parser import PhysicsLaTeXParser

logger = logging.getLogger(__name__)


class ProjectManagerError(Exception):
    """Custom exception for project management errors."""
    pass


class ProjectManager:
    """
    Manages project lifecycle and file operations.

    This class is responsible for:
    - Creating project directory structure
    - Saving/loading project data to/from JSON
    - Integrating with LaTeX parser
    - Generating final output files
    """

    def __init__(self, projects_root: str = "projects"):
        """
        Initialize the project manager.

        Args:
            projects_root: Root directory for all projects
        """
        self.projects_root = Path(projects_root)
        self.projects_root.mkdir(exist_ok=True)
        self.latex_parser = PhysicsLaTeXParser()

    def create_project_from_latex(self, latex_file_path: str, project_name: Optional[str] = None) -> Project:
        """
        Create a new project from a LaTeX file.

        Args:
            latex_file_path: Path to the source LaTeX file
            project_name: Optional name for the project

        Returns:
            Project: The newly created project

        Raises:
            ProjectManagerError: If project creation fails
        """
        try:
            latex_path = Path(latex_file_path)
            if not latex_path.exists():
                raise ProjectManagerError(f"LaTeX file not found: {latex_file_path}")

            # Generate project name if not provided
            if not project_name:
                timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
                project_name = f"{latex_path.stem}_{timestamp}"

            # Create project
            project = Project.create(project_name, str(latex_path.absolute()))

            # Create project directory structure
            project_dir = self._create_project_directory(project.id, project_name)

            # Copy source LaTeX file
            source_dir = project_dir / "source"
            shutil.copy2(latex_path, source_dir / latex_path.name)

            # Parse LaTeX file to extract paragraphs
            logger.info(f"Parsing LaTeX file: {latex_file_path}")
            parsed_data = self.latex_parser.parse_latex_file(latex_file_path)

            # Convert parsed paragraphs to our data model
            for para_data in parsed_data.get('paragraphs', []):
                paragraph = self._convert_parsed_paragraph(para_data)
                project.add_paragraph(paragraph)

            # Update project statistics
            project.update_progress_stats()

            # Save initial project data
            self.save_project(project)

            logger.info(f"Created project '{project_name}' with {len(project.paragraphs)} paragraphs")
            return project

        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            raise ProjectManagerError(f"Failed to create project: {e}")

    def load_project(self, project_id: str) -> Optional[Project]:
        """
        Load an existing project.

        Args:
            project_id: The unique identifier of the project

        Returns:
            Project: The loaded project, or None if not found
        """
        try:
            project_file = self._get_project_file_path(project_id)
            if not project_file.exists():
                logger.warning(f"Project file not found: {project_file}")
                return None

            with open(project_file, 'r', encoding='utf-8') as f:
                project_data = json.load(f)

            project = Project.from_dict(project_data)
            logger.info(f"Loaded project '{project.metadata.name}' with {len(project.paragraphs)} paragraphs")
            return project

        except Exception as e:
            logger.error(f"Failed to load project {project_id}: {e}")
            return None

    def save_project(self, project: Project) -> bool:
        """
        Save project data to JSON file.

        Args:
            project: The project to save

        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            project.update_progress_stats()
            project_file = self._get_project_file_path(project.id)

            with open(project_file, 'w', encoding='utf-8') as f:
                json.dump(project.to_dict(), f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved project '{project.metadata.name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to save project {project.id}: {e}")
            return False

    def list_projects(self) -> List[Dict[str, Any]]:
        """
        List all available projects.

        Returns:
            List[Dict]: List of project metadata
        """
        projects = []
        try:
            for project_dir in self.projects_root.iterdir():
                if project_dir.is_dir():
                    project_file = project_dir / "project.json"
                    if project_file.exists():
                        try:
                            with open(project_file, 'r', encoding='utf-8') as f:
                                project_data = json.load(f)
                            projects.append({
                                'id': project_data['id'],
                                'name': project_data['metadata']['name'],
                                'created_at': project_data['metadata']['created_at'],
                                'last_modified': project_data['metadata']['last_modified'],
                                'total_paragraphs': project_data['metadata']['total_paragraphs'],
                                'processed_paragraphs': project_data['metadata']['processed_paragraphs'],
                                'expert_reviewed_paragraphs': project_data['metadata']['expert_reviewed_paragraphs']
                            })
                        except Exception as e:
                            logger.warning(f"Failed to read project metadata from {project_file}: {e}")

        except Exception as e:
            logger.error(f"Failed to list projects: {e}")

        return sorted(projects, key=lambda x: x['last_modified'], reverse=True)

    def delete_project(self, project_id: str) -> bool:
        """
        Delete a project and all its files.

        Args:
            project_id: The unique identifier of the project

        Returns:
            bool: True if deleted successfully, False otherwise
        """
        try:
            project_dir = self._get_project_directory_path(project_id)
            if project_dir.exists():
                shutil.rmtree(project_dir)
                logger.info(f"Deleted project {project_id}")
                return True
            else:
                logger.warning(f"Project directory not found: {project_dir}")
                return False

        except Exception as e:
            logger.error(f"Failed to delete project {project_id}: {e}")
            return False

    def export_final_latex(self, project: Project) -> Optional[str]:
        """
        Generate final LaTeX document with all expert changes applied.

        Args:
            project: The project to export

        Returns:
            Optional[str]: Path to the generated LaTeX file, or None if failed
        """
        try:
            # Load original LaTeX content
            original_latex_path = Path(project.metadata.latex_file_path)
            if not original_latex_path.exists():
                raise ProjectManagerError(f"Original LaTeX file not found: {original_latex_path}")

            with open(original_latex_path, 'r', encoding='utf-8') as f:
                latex_content = f.read()

            # Apply all expert changes to get final paragraphs
            final_paragraphs = {}
            for paragraph in project.paragraphs:
                final_paragraphs[paragraph.text] = paragraph.get_final_text()

            # Replace paragraphs in LaTeX content
            final_latex_content = latex_content
            for original, final in final_paragraphs.items():
                if original != final:  # Only replace if there are changes
                    final_latex_content = final_latex_content.replace(original, final)

            # Save final LaTeX file
            project_dir = self._get_project_directory_path(project.id)
            final_latex_path = project_dir / "revised.tex"

            with open(final_latex_path, 'w', encoding='utf-8') as f:
                f.write(final_latex_content)

            logger.info(f"Generated final LaTeX: {final_latex_path}")
            return str(final_latex_path)

        except Exception as e:
            logger.error(f"Failed to export final LaTeX for project {project.id}: {e}")
            return None

    def _create_project_directory(self, project_id: str, project_name: str) -> Path:
        """
        Create the directory structure for a new project.

        Args:
            project_id: The unique identifier of the project
            project_name: The human-readable name of the project

        Returns:
            Path: The project directory path
        """
        # Create project directory with sanitized name
        safe_name = self._sanitize_filename(project_name)
        project_dir = self.projects_root / f"{safe_name}_{project_id[:8]}"
        project_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (project_dir / "source").mkdir(exist_ok=True)

        return project_dir

    def _get_project_directory_path(self, project_id: str) -> Path:
        """Get the directory path for a project."""
        # Find directory that ends with the project ID prefix
        for dir_path in self.projects_root.iterdir():
            if dir_path.is_dir() and dir_path.name.endswith(project_id[:8]):
                return dir_path
        # Fallback: use just the project ID
        return self.projects_root / project_id

    def _get_project_file_path(self, project_id: str) -> Path:
        """Get the path to the project JSON file."""
        project_dir = self._get_project_directory_path(project_id)
        return project_dir / "project.json"

    def _convert_parsed_paragraph(self, para_data: Dict[str, Any]) -> Paragraph:
        """
        Convert parsed paragraph data to our Paragraph model.

        Args:
            para_data: Paragraph data from the LaTeX parser

        Returns:
            Paragraph: The converted paragraph
        """
        # Map paragraph types
        paragraph_type = ParagraphType.BODY
        if para_data.get('paragraph_type'):
            type_mapping = {
                'abstract': ParagraphType.ABSTRACT,
                'introduction': ParagraphType.INTRODUCTION,
                'methodology': ParagraphType.METHODOLOGY,
                'results': ParagraphType.RESULTS,
                'conclusion': ParagraphType.CONCLUSION,
                'mathematical': ParagraphType.MATHEMATICAL,
                'literature_review': ParagraphType.LITERATURE_REVIEW
            }
            paragraph_type = type_mapping.get(para_data['paragraph_type'], ParagraphType.BODY)

        return Paragraph.create(
            text=para_data['text'],
            section_title=para_data['section_title'],
            section_number=para_data['section_number'],
            subsection_title=para_data.get('subsection_title'),
            subsection_number=para_data.get('subsection_number'),
            paragraph_type=paragraph_type
        )

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a filename to be safe for the filesystem.

        Args:
            filename: The original filename

        Returns:
            str: The sanitized filename
        """
        # Remove or replace unsafe characters
        unsafe_chars = '<>:"/\\|?*'
        safe_filename = filename
        for char in unsafe_chars:
            safe_filename = safe_filename.replace(char, '_')

        # Limit length
        return safe_filename[:50]

    # Settings Management Methods

    def save_project_settings(self, project_id: str, settings: ApplicationSettings) -> bool:
        """
        Save project-specific settings.

        Args:
            project_id: The project ID
            settings: Settings to save

        Returns:
            bool: True if saved successfully
        """
        try:
            project_dir = self._get_project_directory(project_id)
            settings_file = project_dir / "settings.json"

            # Mark as project-specific
            settings.is_project_specific = True

            # Save without API keys for security (they should use global)
            settings.save_to_file(settings_file, include_api_keys=False)

            logger.info(f"Saved project settings for project {project_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save project settings: {e}")
            return False

    def load_project_settings(self, project_id: str) -> Optional[ApplicationSettings]:
        """
        Load project-specific settings.

        Args:
            project_id: The project ID

        Returns:
            Optional[ApplicationSettings]: Project settings if they exist, None otherwise
        """
        try:
            project_dir = self._get_project_directory(project_id)
            settings_file = project_dir / "settings.json"

            if settings_file.exists():
                settings = ApplicationSettings.load_from_file(settings_file)
                logger.info(f"Loaded project settings for project {project_id}")
                return settings
            else:
                logger.debug(f"No project-specific settings found for project {project_id}")
                return None

        except Exception as e:
            logger.error(f"Failed to load project settings: {e}")
            return None

    def save_global_settings(self, settings: ApplicationSettings) -> bool:
        """
        Save global application settings.

        Args:
            settings: Settings to save

        Returns:
            bool: True if saved successfully
        """
        try:
            global_settings_file = self.projects_root / "global_settings.json"

            # Mark as global
            settings.is_project_specific = False

            # Save with API keys
            settings.save_to_file(global_settings_file, include_api_keys=True)

            logger.info("Saved global settings")
            return True

        except Exception as e:
            logger.error(f"Failed to save global settings: {e}")
            return False

    def load_global_settings(self) -> Optional[ApplicationSettings]:
        """
        Load global application settings.

        Returns:
            Optional[ApplicationSettings]: Global settings if they exist, None otherwise
        """
        try:
            global_settings_file = self.projects_root / "global_settings.json"

            if global_settings_file.exists():
                settings = ApplicationSettings.load_from_file(global_settings_file)
                logger.info("Loaded global settings")
                return settings
            else:
                logger.debug("No global settings found")
                return None

        except Exception as e:
            logger.error(f"Failed to load global settings: {e}")
            return None

    # .CMS File Format Methods

    def save_project_to_cms(self, project: Project, file_path: str, settings: Optional[ApplicationSettings] = None) -> bool:
        """
        Save project to a .cms file.

        Args:
            project: The project to save
            file_path: Path to save the .cms file
            settings: Optional project settings to save with the project

        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Update progress stats before saving
            project.update_progress_stats()

            # Create CMS file content
            cms_data = {
                'version': '1.1',  # Updated for rulebook support
                'project': project.to_dict(),
                'settings': settings.to_dict() if settings else None
            }

            # Ensure file has .cms extension
            if not file_path.endswith('.cms'):
                file_path += '.cms'

            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(cms_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved project to .cms file: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save project to .cms file: {e}")
            return False

    def load_project_from_cms(self, file_path: str) -> tuple[Optional[Project], Optional[ApplicationSettings]]:
        """
        Load project from a .cms file.

        Args:
            file_path: Path to the .cms file

        Returns:
            tuple: (Project, Settings) or (None, None) if failed
        """
        try:
            if not Path(file_path).exists():
                logger.warning(f"CMS file not found: {file_path}")
                return None, None

            with open(file_path, 'r', encoding='utf-8') as f:
                cms_data = json.load(f)

            # Load project
            project = Project.from_dict(cms_data['project'])

            # Load settings if present
            settings = None
            if cms_data.get('settings'):
                settings = ApplicationSettings.from_dict(cms_data['settings'])

            logger.info(f"Loaded project from .cms file: {file_path}")
            return project, settings

        except Exception as e:
            logger.error(f"Failed to load project from .cms file: {e}")
            return None, None