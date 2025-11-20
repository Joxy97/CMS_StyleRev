"""
Project Selector Component.

This component handles project loading, creation, and selection.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Callable, Optional, Dict, Any
import logging

from services import ProjectManager
from models import Project

logger = logging.getLogger(__name__)


class ProjectSelector(ttk.Frame):
    """
    GUI component for project selection and management.

    This component provides:
    - Load LaTeX file to create new project
    - Select existing project from dropdown
    - Display project metadata
    - Delete projects
    """

    def __init__(self, parent, on_project_loaded: Callable[[Project], None]):
        """
        Initialize the project selector.

        Args:
            parent: The parent tkinter widget
            on_project_loaded: Callback function called when a project is loaded
        """
        super().__init__(parent)
        self.on_project_loaded = on_project_loaded
        self.project_manager = ProjectManager()
        self.current_project: Optional[Project] = None

        self._create_widgets()
        self._refresh_project_list()

    def _create_widgets(self):
        """Create and layout the GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.X, padx=5, pady=5)

        # Load LaTeX section
        load_frame = ttk.LabelFrame(main_frame, text="Create New Project")
        load_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Button(
            load_frame,
            text="Load LaTeX File",
            command=self._load_latex_file
        ).pack(side=tk.LEFT, padx=5, pady=5)

        # Existing projects section
        existing_frame = ttk.LabelFrame(main_frame, text="Existing Projects")
        existing_frame.pack(fill=tk.X, pady=(5, 0))

        # Project selection
        select_frame = ttk.Frame(existing_frame)
        select_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(select_frame, text="Project:").pack(side=tk.LEFT)

        self.project_var = tk.StringVar()
        self.project_combo = ttk.Combobox(
            select_frame,
            textvariable=self.project_var,
            state="readonly",
            width=40
        )
        self.project_combo.pack(side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True)
        self.project_combo.bind('<<ComboboxSelected>>', self._on_project_selected)

        # Action buttons
        button_frame = ttk.Frame(select_frame)
        button_frame.pack(side=tk.RIGHT, padx=(5, 0))

        ttk.Button(
            button_frame,
            text="Load",
            command=self._load_selected_project
        ).pack(side=tk.LEFT)

        ttk.Button(
            button_frame,
            text="Refresh",
            command=self._refresh_project_list
        ).pack(side=tk.LEFT, padx=(2, 0))

        ttk.Button(
            button_frame,
            text="Delete",
            command=self._delete_selected_project
        ).pack(side=tk.LEFT, padx=(2, 0))

        # Project info section
        info_frame = ttk.LabelFrame(self, text="Project Information")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create text widget for project info
        self.info_text = tk.Text(
            info_frame,
            height=4,
            state=tk.DISABLED,
            wrap=tk.WORD,
            font=('Consolas', 9)
        )
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Initial info
        self._update_project_info()

    def _load_latex_file(self):
        """Handle loading a LaTeX file to create a new project."""
        try:
            filename = filedialog.askopenfilename(
                title="Select LaTeX File",
                filetypes=[
                    ("LaTeX files", "*.tex"),
                    ("All files", "*.*")
                ]
            )

            if not filename:
                return

            # Show progress
            self._show_status("Creating project from LaTeX file...")

            # Create project
            project = self.project_manager.create_project_from_latex(filename)

            if project:
                logger.info(f"Created project: {project.metadata.name}")
                self.current_project = project
                self._refresh_project_list()
                self._update_project_info(project)
                self.on_project_loaded(project)
                self._show_status("Project created successfully!")
            else:
                messagebox.showerror("Error", "Failed to create project from LaTeX file")

        except Exception as e:
            logger.error(f"Failed to load LaTeX file: {e}")
            messagebox.showerror("Error", f"Failed to load LaTeX file: {str(e)}")
        finally:
            self._show_status("")

    def _refresh_project_list(self):
        """Refresh the list of available projects."""
        try:
            projects = self.project_manager.list_projects()

            # Update combobox values
            project_names = [f"{p['name']} ({p['id'][:8]}...)" for p in projects]
            self.project_combo['values'] = project_names

            # Store project data for reference
            self.project_list = projects

            logger.debug(f"Refreshed project list: {len(projects)} projects")

        except Exception as e:
            logger.error(f"Failed to refresh project list: {e}")
            messagebox.showerror("Error", f"Failed to refresh project list: {str(e)}")

    def _on_project_selected(self, event):
        """Handle project selection from combobox."""
        selected_index = self.project_combo.current()
        if 0 <= selected_index < len(self.project_list):
            project_info = self.project_list[selected_index]
            self._update_project_info_from_metadata(project_info)

    def _load_selected_project(self):
        """Load the currently selected project."""
        try:
            selected_index = self.project_combo.current()
            if selected_index < 0 or selected_index >= len(self.project_list):
                messagebox.showwarning("Warning", "Please select a project first")
                return

            project_info = self.project_list[selected_index]
            project_id = project_info['id']

            self._show_status("Loading project...")

            project = self.project_manager.load_project(project_id)
            if project:
                logger.info(f"Loaded project: {project.metadata.name}")
                self.current_project = project
                self._update_project_info(project)
                self.on_project_loaded(project)
                self._show_status("Project loaded successfully!")
            else:
                messagebox.showerror("Error", "Failed to load selected project")

        except Exception as e:
            logger.error(f"Failed to load project: {e}")
            messagebox.showerror("Error", f"Failed to load project: {str(e)}")
        finally:
            self._show_status("")

    def _delete_selected_project(self):
        """Delete the currently selected project."""
        try:
            selected_index = self.project_combo.current()
            if selected_index < 0 or selected_index >= len(self.project_list):
                messagebox.showwarning("Warning", "Please select a project first")
                return

            project_info = self.project_list[selected_index]
            project_name = project_info['name']

            # Confirm deletion
            if not messagebox.askyesno(
                "Confirm Delete",
                f"Are you sure you want to delete project '{project_name}'?\n\n"
                "This action cannot be undone."
            ):
                return

            project_id = project_info['id']
            if self.project_manager.delete_project(project_id):
                logger.info(f"Deleted project: {project_name}")
                self._refresh_project_list()
                self._update_project_info()
                if self.current_project and self.current_project.id == project_id:
                    self.current_project = None
                messagebox.showinfo("Success", f"Project '{project_name}' deleted successfully")
            else:
                messagebox.showerror("Error", f"Failed to delete project '{project_name}'")

        except Exception as e:
            logger.error(f"Failed to delete project: {e}")
            messagebox.showerror("Error", f"Failed to delete project: {str(e)}")

    def _update_project_info(self, project: Optional[Project] = None):
        """Update the project information display."""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)

        if project:
            info_text = f"""Project: {project.metadata.name}
Created: {project.metadata.created_at.strftime('%Y-%m-%d %H:%M:%S')}
Last Modified: {project.metadata.last_modified.strftime('%Y-%m-%d %H:%M:%S')}
Total Paragraphs: {project.metadata.total_paragraphs}
Processed: {project.metadata.processed_paragraphs}
Expert Reviewed: {project.metadata.expert_reviewed_paragraphs}
Source: {project.metadata.latex_file_path}"""
        else:
            info_text = "No project selected"

        self.info_text.insert(tk.END, info_text)
        self.info_text.config(state=tk.DISABLED)

    def _update_project_info_from_metadata(self, project_info: Dict[str, Any]):
        """Update project info display from metadata dictionary."""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)

        info_text = f"""Project: {project_info['name']}
Created: {project_info['created_at'][:19].replace('T', ' ')}
Last Modified: {project_info['last_modified'][:19].replace('T', ' ')}
Total Paragraphs: {project_info['total_paragraphs']}
Processed: {project_info['processed_paragraphs']}
Expert Reviewed: {project_info['expert_reviewed_paragraphs']}"""

        self.info_text.insert(tk.END, info_text)
        self.info_text.config(state=tk.DISABLED)

    def _show_status(self, message: str):
        """Show status message (placeholder for status bar integration)."""
        if hasattr(self.master, 'show_status'):
            self.master.show_status(message)
        else:
            logger.info(f"Status: {message}")

    def save_current_project(self):
        """Save the current project if one is loaded."""
        if self.current_project:
            if self.project_manager.save_project(self.current_project):
                self._update_project_info(self.current_project)
                return True
            else:
                messagebox.showerror("Error", "Failed to save project")
                return False
        return True

    def get_current_project(self) -> Optional[Project]:
        """Get the currently loaded project."""
        return self.current_project