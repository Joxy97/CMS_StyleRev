"""
StyleRev CMS - Expert Annotation System (Main Application)

This is the main GUI application that integrates the Model-View architecture
for expert annotation of CMS style corrections.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import logging
import threading
import os
import sys
from typing import Optional, List, Dict

from models import Project, Paragraph, ExpertAction, IssueType, IssueSeverity, ApplicationSettings, settings_manager, PROJECT_ROOT
from services import ProjectManager, AnnotationService
from services.paragraph_export_service import ParagraphExportService, ParagraphExportServiceError
from services.paragraph_export_service import ParagraphExportService, ParagraphExportServiceError
from views import ParagraphListView, AnnotationPanel, SettingsDialog
from views.rulebook_view import RulebookEditorWindow
from tkinter import filedialog
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExpertAnnotationApp:
    """
    Main application class for the Expert Annotation System.

    This application follows the Model-View architecture and provides:
    - Project management (create, load, save)
    - Document parsing and paragraph extraction
    - AI-powered suggestion generation
    - Expert annotation interface
    - Export capabilities
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("StyleRev CMS - Expert Annotation System")
        self.root.geometry("1400x900")

        # Set application icon
        icon_path = PROJECT_ROOT / "resources" / "CMS_Icon.ico"
        if icon_path.exists():
            self.root.iconbitmap(str(icon_path))

        # Initialize services
        self.project_manager = ProjectManager()
        self.annotation_service = AnnotationService(
            llm_model=settings_manager.llm_model,
            rag_model=settings_manager.rag_model
        )
        self.paragraph_export_service = ParagraphExportService()

        # Current state
        self.current_project: Optional[Project] = None
        self.current_paragraph: Optional[Paragraph] = None
        self.current_settings: ApplicationSettings = None
        self.current_file_path: Optional[str] = None  # Track currently open .cms file
        self.log_visible = tk.BooleanVar(value=True)  # Track log panel visibility
        self._process_all_thread: Optional[threading.Thread] = None
        self._process_all_pause_event: Optional[threading.Event] = None

        # Load default settings
        self._load_default_settings()

        # Create GUI
        self._create_widgets()
        self._create_menu()

        # Configure logging to show in status bar
        self._setup_logging_handler()

        logger.info("Expert Annotation System initialized")

    def _create_widgets(self):
        """Create and layout the main GUI widgets."""
        # Create vertical paned window to allow resizing of log area
        vertical_paned = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        vertical_paned.pack(fill=tk.BOTH, expand=True)

        # Main content area
        main_content = ttk.Frame(vertical_paned)
        vertical_paned.add(main_content, weight=10)

        # Main container with paned windows for resizable layout
        main_paned = ttk.PanedWindow(main_content, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        # Left panel for project management and paragraph list
        left_panel = ttk.Frame(main_paned)
        main_paned.add(left_panel, weight=1)

        # Right panel for annotation interface
        right_panel = ttk.Frame(main_paned)
        main_paned.add(right_panel, weight=2)

        # Configure left panel layout
        self._create_left_panel(left_panel)

        # Configure right panel layout
        self._create_right_panel(right_panel)

        # Log panel at bottom (stretchable)
        self._create_log_panel(vertical_paned)

        # Status bar at very bottom (always visible)
        self._create_status_bar()

    def _create_left_panel(self, parent):
        """Create the left panel with paragraph list."""
        # Paragraph list
        self.paragraph_list = ParagraphListView(
            parent,
            on_paragraph_selected=self._on_paragraph_selected
        )
        self.paragraph_list.pack(fill=tk.BOTH, expand=True)

        self.paragraph_list.on_pause_requested = self._pause_process_all
        self.paragraph_list.on_resume_requested = self._resume_process_all

        # Connect process all method
        self.paragraph_list.master.process_all_paragraphs = self._process_all_paragraphs

    def _create_right_panel(self, parent):
        """Create the right panel with annotation interface."""
        self.annotation_panel = AnnotationPanel(
            parent,
            on_annotation_changed=self._on_annotation_changed
        )
        self.annotation_panel.pack(fill=tk.BOTH, expand=True)

        # Connect annotation methods
        self.annotation_panel.master.process_paragraph_with_ai = self._process_paragraph_with_ai
        self.annotation_panel.master.apply_expert_annotation = self._apply_expert_annotation
        self.annotation_panel.master.add_expert_issue = self._add_expert_issue

    def _create_log_panel(self, parent_paned):
        """Create the stretchable log panel at the bottom."""
        self.log_frame = ttk.Frame(parent_paned)
        self.vertical_paned = parent_paned
        parent_paned.add(self.log_frame, weight=1)

        # Header with title
        header_frame = ttk.Frame(self.log_frame)
        header_frame.pack(fill=tk.X, padx=2, pady=2)

        ttk.Label(header_frame, text="Log", font=('TkDefaultFont', 9, 'bold')).pack(side=tk.LEFT)

        # Scrollable log text area
        self.log_text = scrolledtext.ScrolledText(
            self.log_frame,
            height=5,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=('Consolas', 9),
            bg='#F8F8F8'
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=(0, 2))

        # Configure tag for different log levels
        self.log_text.tag_configure('INFO', foreground='black')
        self.log_text.tag_configure('WARNING', foreground='orange')
        self.log_text.tag_configure('ERROR', foreground='red')

    def _create_status_bar(self):
        """Create the persistent status bar."""
        self.status_bar = ttk.Frame(self.root, padding=(6, 2))
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = ttk.Label(
            self.status_bar,
            text="Ready",
            anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.model_info_label = ttk.Label(
            self.status_bar,
            text="",
            anchor=tk.E,
            foreground='gray'
        )
        self.model_info_label.pack(side=tk.RIGHT)
        self._update_model_info_display()

    def _create_menu(self):
        """Create the application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Project...", command=self._new_project, accelerator="Ctrl+N")
        file_menu.add_command(label="Open Project...", command=self._open_project, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Save", command=self._save_project, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As...", command=self._save_project_as, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(
            label="Export Paragraphs to LaTeX...",
            command=self._export_paragraphs
        )
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_closing)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Rulebook Editor...", command=self._open_rulebook_editor)
        edit_menu.add_separator()
        edit_menu.add_command(label="Project Settings...", command=self._open_project_settings)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Process All Paragraphs", command=self._process_all_paragraphs)
        tools_menu.add_command(label="Project Statistics...", command=self._show_statistics)
        if sys.platform == 'win32':
            tools_menu.add_separator()
            tools_menu.add_command(label="Register .cms File Type...", command=self._register_file_type)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(label="Show Log", variable=self.log_visible, command=self._toggle_log)
        view_menu.add_separator()
        view_menu.add_command(label="Refresh All", command=self._refresh_all)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About...", command=self._show_about)

        # Keyboard shortcuts
        self.root.bind('<Control-n>', lambda e: self._new_project())
        self.root.bind('<Control-o>', lambda e: self._open_project())
        self.root.bind('<Control-s>', lambda e: self._save_project())
        self.root.bind('<Control-Shift-S>', lambda e: self._save_project_as())

    def _setup_logging_handler(self):
        """Set up logging to display messages in status bar."""
        class StatusBarHandler(logging.Handler):
            def __init__(self, status_callback):
                super().__init__()
                self.status_callback = status_callback

            def emit(self, record):
                if record.levelno >= logging.INFO:
                    msg = self.format(record)
                    # Schedule status update in main thread
                    self.status_callback(msg)

        # Add handler for INFO level and above
        status_handler = StatusBarHandler(self._show_status_threadsafe)
        status_handler.setLevel(logging.INFO)
        status_handler.setFormatter(logging.Formatter('%(message)s'))
        logging.getLogger().addHandler(status_handler)

    def _show_status_threadsafe(self, message: str):
        """Show status message in a thread-safe way."""
        self.root.after(0, lambda: self._append_to_log(message))

    def show_status(self, message: str):
        """Show status message."""
        if hasattr(self, 'status_label') and self.status_label is not None:
            self.status_label.config(text=message)
        self._append_to_log(message)
        self.root.update_idletasks()

    def _append_to_log(self, message: str):
        """Append a message to the log text widget."""
        if not hasattr(self, 'log_text'):
            return

        # Determine log level for coloring
        tag = 'INFO'
        if 'error' in message.lower() or 'failed' in message.lower():
            tag = 'ERROR'
        elif 'warning' in message.lower():
            tag = 'WARNING'

        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + '\n', tag)
        self.log_text.see(tk.END)  # Auto-scroll to bottom
        self.log_text.config(state=tk.DISABLED)

    def _toggle_log(self):
        """Toggle the visibility of the log panel."""
        if self.log_visible.get():
            # Show log panel
            self.vertical_paned.add(self.log_frame, weight=1)
        else:
            # Hide log panel
            self.vertical_paned.forget(self.log_frame)

    def _update_model_info_display(self):
        """Update the model info display in status bar."""
        if self.current_settings:
            llm = self.current_settings.model_settings.llm_model
            rag = self.current_settings.model_settings.rag_model
        else:
            llm = settings_manager.llm_model
            rag = settings_manager.rag_model

        # Shorten model names for display
        llm_short = llm.split('/')[-1] if '/' in llm else llm
        rag_short = rag.split('/')[-1] if '/' in rag else rag

        self.model_info_label.config(text=f"LLM: {llm_short} | RAG: {rag_short}")

    # Event Handlers (see _on_project_loaded in Settings Management section)

    def _on_paragraph_selected(self, paragraph: Paragraph):
        """Handle paragraph selection."""
        logger.debug(f"Selected paragraph: {paragraph.id}")
        self.current_paragraph = paragraph
        self.annotation_panel.load_paragraph(paragraph)
        self.show_status(f"Selected paragraph from {paragraph.section_title}")

    def _on_annotation_changed(self, paragraph: Paragraph):
        """Handle annotation changes."""
        logger.debug(f"Annotations changed for paragraph: {paragraph.id}")
        self.paragraph_list.refresh_paragraph(paragraph)

        # Mark project as modified
        if self.current_project and self.current_file_path:
            self.root.title(f"StyleRev CMS - {self.current_project.metadata.name} *")

        # Auto-save on annotation changes
        self._save_project()

    # AI Processing Methods

    def _process_paragraph_with_ai(self, paragraph: Paragraph):
        """Process a single paragraph with AI in a background thread."""
        if not self._ensure_api_keys_available():
            return

        # Create loading dialog
        loading_window = tk.Toplevel(self.root)
        loading_window.title("Processing")
        loading_window.geometry("300x120")
        loading_window.transient(self.root)
        loading_window.grab_set()
        loading_window.resizable(False, False)

        # Center on parent
        loading_window.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (loading_window.winfo_width() // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (loading_window.winfo_height() // 2)
        loading_window.geometry(f"+{x}+{y}")

        ttk.Label(
            loading_window,
            text="Processing paragraph with AI...",
            font=('TkDefaultFont', 10, 'bold')
        ).pack(pady=(20, 10))

        # Bouncing progress bar
        progress_bar = ttk.Progressbar(loading_window, mode='indeterminate', length=250)
        progress_bar.pack(pady=10)
        progress_bar.start(10)

        def process_in_background():
            try:
                self.show_status(f"Processing paragraph with AI...")
                # Pass the project's rulebook if available
                rulebook = self.current_project.rulebook if self.current_project else None
                success = self.annotation_service.process_paragraph_with_ai(
                    paragraph, rulebook=rulebook
                )

                if success:
                    # Update UI in main thread
                    self.root.after(0, lambda: self._on_paragraph_processed_with_dialog(paragraph, loading_window))
                else:
                    self.root.after(0, lambda: self._on_processing_failed(loading_window, "AI processing failed"))
            except Exception as e:
                logger.error(f"AI processing error: {e}")
                self.root.after(0, lambda: self._on_processing_failed(loading_window, f"AI processing error: {e}"))

        thread = threading.Thread(target=process_in_background)
        thread.daemon = True
        thread.start()

    def _on_paragraph_processed_with_dialog(self, paragraph: Paragraph, loading_window):
        """Handle paragraph processing completion with dialog cleanup."""
        loading_window.destroy()
        self._on_paragraph_processed(paragraph)

    def _on_processing_failed(self, loading_window, message: str):
        """Handle processing failure with dialog cleanup."""
        loading_window.destroy()
        self.show_status(message)

    def _process_all_paragraphs(self):
        """Process all paragraphs with AI in a background thread with progress controls."""
        if not self.current_project:
            messagebox.showwarning("Warning", "No project loaded")
            return
        if self._process_all_thread and self._process_all_thread.is_alive():
            messagebox.showinfo("AI Processing", "Paragraph processing is already running.")
            return
        if not self._ensure_api_keys_available():
            return

        paragraphs = list(self.current_project.paragraphs)
        total_paragraphs = len(paragraphs)
        if total_paragraphs == 0:
            messagebox.showinfo("AI Processing", "No paragraphs to process.")
            return

        stats = {
            'total': total_paragraphs,
            'processed': 0,
            'failed': 0,
            'suggestions_generated': 0
        }

        self._process_all_pause_event = threading.Event()
        self._process_all_pause_event.set()
        self.paragraph_list.set_process_state(True, total=total_paragraphs, completed=0)
        self.show_status("Processing all paragraphs with AI...")

        def process_in_background():
            try:
                rulebook = self.current_project.rulebook if self.current_project else None

                for index, paragraph in enumerate(paragraphs):
                    if not self._process_all_pause_event:
                        break
                    self._process_all_pause_event.wait()

                    success = self.annotation_service.process_paragraph_with_ai(
                        paragraph,
                        rulebook=rulebook
                    )
                    if success:
                        stats['processed'] += 1
                        stats['suggestions_generated'] += len(paragraph.suggestions)
                    else:
                        stats['failed'] += 1

                    completed = index + 1
                    self.root.after(0, lambda par=paragraph, ok=success: self._on_batch_paragraph_progress(par, ok))
                    self.root.after(0, lambda comp=completed: self.paragraph_list.update_processing_progress(comp, total_paragraphs))

                self.root.after(0, lambda: self._on_process_all_complete(stats))

            except Exception as e:
                logger.error(f"Batch AI processing error: {e}")
                self.root.after(0, lambda: self._on_process_all_complete(stats, error=e))

        self._process_all_thread = threading.Thread(target=process_in_background, daemon=True)
        self._process_all_thread.start()

    def _pause_process_all(self):
        """Pause batch processing if running."""
        if self._process_all_pause_event and self._process_all_thread and self._process_all_thread.is_alive():
            self._process_all_pause_event.clear()
            self.paragraph_list.set_process_paused(True)
            self.show_status("Paragraph processing paused")

    def _resume_process_all(self):
        """Resume paused batch processing."""
        if self._process_all_pause_event and self._process_all_thread and self._process_all_thread.is_alive():
            self._process_all_pause_event.set()
            self.paragraph_list.set_process_paused(False)
            self.show_status("Paragraph processing resumed")

    def _on_process_all_complete(self, stats: Dict[str, int], error: Optional[Exception] = None):
        """Handle completion (or failure) of batch processing."""
        self.paragraph_list.set_process_state(False)
        self.paragraph_list.set_process_paused(False)
        self._process_all_thread = None
        self._process_all_pause_event = None

        self.paragraph_list._refresh_display()
        self.annotation_panel._refresh_display()

        if error:
            self.show_status(f"AI processing error: {error}")
            messagebox.showerror("AI Processing Error", f"An error occurred during processing:\n{error}")
            return

        self.show_status(f"AI processing complete: {stats['processed']}/{stats['total']} paragraphs processed")
        messagebox.showinfo(
            "AI Processing Complete",
            f"Processed: {stats['processed']}/{stats['total']} paragraphs\n"
            f"Generated: {stats['suggestions_generated']} suggestions\n"
            f"Failed: {stats['failed']} paragraphs"
        )

    def _on_paragraph_processed(self, paragraph: Paragraph):
        """Handle paragraph processing completion."""
        self.paragraph_list.refresh_paragraph(paragraph)
        if self.current_paragraph == paragraph:
            self.annotation_panel._refresh_display()
        self.show_status(f"Generated {len(paragraph.suggestions)} suggestions for paragraph")

    def _on_batch_paragraph_progress(self, paragraph: Paragraph, success: bool):
        """Update UI incrementally while processing all paragraphs."""
        self.paragraph_list.refresh_paragraph(paragraph)
        if self.current_paragraph == paragraph:
            self.annotation_panel._refresh_display()
        if success:
            self.show_status(f"Generated {len(paragraph.suggestions)} suggestions for paragraph {paragraph.id}")
        else:
            self.show_status(f"Failed to process paragraph {paragraph.id}")

    # Expert Annotation Methods

    def _apply_expert_annotation(self, paragraph: Paragraph, suggestion_id: str,
                                action: ExpertAction, modified_text: Optional[str] = None):
        """Apply an expert annotation."""
        success = self.annotation_service.apply_expert_annotation(
            paragraph, suggestion_id, action, modified_text
        )

        if success:
            logger.info(f"Applied annotation: {action.value} for suggestion {suggestion_id}")
        else:
            messagebox.showerror("Error", "Failed to apply annotation")

    def _add_expert_issue(self, paragraph: Paragraph, original_text: str, corrected_text: str,
                         issue_type: IssueType, severity: IssueSeverity, notes: Optional[str] = None):
        """Add an expert-identified issue."""
        success = self.annotation_service.add_expert_issue(
            paragraph, original_text, corrected_text, issue_type, severity, notes
        )

        if success:
            logger.info(f"Added expert issue: {issue_type.value} ({severity.value})")
        else:
            messagebox.showerror("Error", "Failed to add expert issue")

    # Menu Command Handlers

    def _new_project(self):
        """Create a new project from a LaTeX file."""
        # Ask user to select a LaTeX file
        latex_file = filedialog.askopenfilename(
            title="Select LaTeX File",
            filetypes=[("LaTeX files", "*.tex"), ("All files", "*.*")]
        )

        if not latex_file:
            return

        try:
            # Create project from LaTeX
            self.show_status("Creating new project from LaTeX file...")
            project = self.project_manager.create_project_from_latex(latex_file)

            # Load default settings for the new project
            if self.current_settings:
                project_settings = self.current_settings
            else:
                project_settings = ApplicationSettings()
                project_settings.is_project_specific = True

            # Load the project
            self._load_project_into_ui(project, project_settings)

            # Reset current file path (unsaved new project)
            self.current_file_path = None

            # Update title
            self.root.title(f"StyleRev CMS - {project.metadata.name} *")

            messagebox.showinfo(
                "Project Created",
                f"Created project with {len(project.paragraphs)} paragraphs.\n\n"
                "Please save the project to a .cms file."
            )

        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            messagebox.showerror("Error", f"Failed to create project:\n{str(e)}")

    def _open_project(self):
        """Open an existing .cms project file."""
        file_path = filedialog.askopenfilename(
            title="Open Project",
            filetypes=[("CMS Project files", "*.cms"), ("All files", "*.*")]
        )

        if not file_path:
            return

        # Create loading dialog
        loading_window = tk.Toplevel(self.root)
        loading_window.title("Loading Project")
        loading_window.geometry("300x120")
        loading_window.transient(self.root)
        loading_window.grab_set()
        loading_window.resizable(False, False)

        # Center on parent
        loading_window.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (loading_window.winfo_width() // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (loading_window.winfo_height() // 2)
        loading_window.geometry(f"+{x}+{y}")

        ttk.Label(
            loading_window,
            text="Loading project...",
            font=('TkDefaultFont', 10, 'bold')
        ).pack(pady=(20, 10))

        # Bouncing progress bar
        progress_bar = ttk.Progressbar(loading_window, mode='indeterminate', length=250)
        progress_bar.pack(pady=10)
        progress_bar.start(10)

        def load_in_background():
            try:
                self.show_status(f"Loading project from {file_path}...")
                project, settings = self.project_manager.load_project_from_cms(file_path)

                if project is None:
                    self.root.after(0, lambda: self._on_project_load_failed(loading_window, "Failed to load project"))
                    return

                # Use loaded settings or default
                if settings is None:
                    settings = self.current_settings or ApplicationSettings()
                    settings.is_project_specific = True

                # Update UI in main thread
                self.root.after(0, lambda: self._on_project_loaded(loading_window, project, settings, file_path))

            except Exception as e:
                logger.error(f"Failed to open project: {e}")
                self.root.after(0, lambda: self._on_project_load_failed(loading_window, f"Failed to open project:\n{str(e)}"))

        thread = threading.Thread(target=load_in_background)
        thread.daemon = True
        thread.start()

    def _on_project_loaded(self, loading_window, project, settings, file_path):
        """Handle successful project loading."""
        loading_window.destroy()

        # Load the project
        self._load_project_into_ui(project, settings)

        # Save current file path
        self.current_file_path = file_path

        # Update title
        self.root.title(f"StyleRev CMS - {project.metadata.name}")

        self.show_status(f"Loaded project: {project.metadata.name}")

    def _on_project_load_failed(self, loading_window, message: str):
        """Handle project loading failure."""
        loading_window.destroy()
        messagebox.showerror("Error", message)

    def _save_project(self):
        """Save the current project."""
        if not self.current_project:
            messagebox.showwarning("Warning", "No project to save")
            return False

        # If we have a current file path, save to it
        if self.current_file_path:
            return self._save_to_file(self.current_file_path)
        else:
            # Otherwise, prompt for Save As
            return self._save_project_as()

    def _save_project_as(self):
        """Save the current project to a new file."""
        if not self.current_project:
            messagebox.showwarning("Warning", "No project to save")
            return False

        # Ask user where to save
        file_path = filedialog.asksaveasfilename(
            title="Save Project As",
            defaultextension=".cms",
            filetypes=[("CMS Project files", "*.cms"), ("All files", "*.*")]
        )

        if not file_path:
            return False

        # Save to the new file
        if self._save_to_file(file_path):
            self.current_file_path = file_path
            self.root.title(f"StyleRev CMS - {self.current_project.metadata.name}")
            return True
        return False

    def _save_to_file(self, file_path: str) -> bool:
        """Save project to a specific file path."""
        try:
            success = self.project_manager.save_project_to_cms(
                self.current_project,
                file_path,
                self.current_settings
            )

            if success:
                self.show_status(f"Project saved to {file_path}")
                # Update title to remove asterisk
                self.root.title(f"StyleRev CMS - {self.current_project.metadata.name}")
                return True
            else:
                messagebox.showerror("Error", "Failed to save project")
                return False

        except Exception as e:
            logger.error(f"Failed to save project: {e}")
            messagebox.showerror("Error", f"Failed to save project:\n{str(e)}")
            return False

    def _export_paragraphs(self):
        """Export the current paragraphs into a reconstructed LaTeX file."""
        if not self.current_project:
            messagebox.showwarning("Warning", "No project loaded")
            return

        file_path = filedialog.asksaveasfilename(
            title="Export Paragraphs to LaTeX",
            defaultextension=".tex",
            filetypes=[("LaTeX files", "*.tex"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            self.show_status("Exporting paragraphs...")
            destination = self.paragraph_export_service.export_paragraphs(
                self.current_project,
                output_path=file_path
            )
            messagebox.showinfo(
                "Export Successful",
                f"Paragraph export saved to:\n{destination}"
            )
            self.show_status("Paragraph export completed successfully")
        except ParagraphExportServiceError as err:
            validation = err.validation_result
            message = self._build_export_blocker_message(validation) if validation else "Export blocked due to outstanding suggestions."
            messagebox.showwarning("Export Blocked", message)
            self.show_status("Export blocked: outstanding suggestions exist")
        except Exception as e:
            logger.error(f"Export error: {e}")
            messagebox.showerror("Error", f"Export failed: {str(e)}")
            self.show_status("Paragraph export failed")

    def _build_export_blocker_message(self, validation):
        lines = ["Resolve the following suggestions before exporting:"]
        if validation.pending_issues:
            lines.append(f"- {len(validation.pending_issues)} pending suggestion(s) awaiting review")
            pending_labels = sorted({issue.paragraph_label for issue in validation.pending_issues})
            if pending_labels:
                lines.append(f"  Sections: {', '.join(pending_labels[:3])}")
        if validation.reviewed_unimplemented_issues:
            lines.append(f"- {len(validation.reviewed_unimplemented_issues)} reviewed suggestion(s) not implemented yet")
            reviewed_labels = sorted({issue.paragraph_label for issue in validation.reviewed_unimplemented_issues})
            if reviewed_labels:
                lines.append(f"  Sections: {', '.join(reviewed_labels[:3])}")
        return "\n".join(lines) if len(lines) > 1 else "Export blocked due to outstanding suggestions."

    def _show_statistics(self):
        """Show project statistics dialog."""
        if not self.current_project:
            messagebox.showwarning("Warning", "No project loaded")
            return

        stats = self.annotation_service.get_annotation_statistics(self.current_project)

        stats_text = f"""Project Statistics

Total Paragraphs: {stats['total_paragraphs']}
AI Processed: {stats['ai_processed_paragraphs']}
Expert Reviewed: {stats['expert_reviewed_paragraphs']}

AI Suggestions:
- Total: {stats['total_suggestions']}
- Pending: {stats['suggestions_by_action']['pending']}
- Accepted: {stats['suggestions_by_action']['accepted']}
- Rejected: {stats['suggestions_by_action']['rejected']}
- Modified: {stats['suggestions_by_action']['modified']}

Expert Additions: {stats['expert_additions']}

Status Distribution:
- Not Processed: {stats['status_distribution']['Not Processed']}
- Changed: {stats['status_distribution']['Changed']}
- Done: {stats['status_distribution']['Done']}
- Error: {stats['status_distribution']['Error']}"""

        messagebox.showinfo("Project Statistics", stats_text)

    def _refresh_all(self):
        """Refresh all views."""
        if self.current_project:
            self.paragraph_list._refresh_display()
            self.annotation_panel._refresh_display()
        self.project_selector._refresh_project_list()
        self.show_status("Views refreshed")

    def _show_about(self):
        """Show about dialog."""
        about_text = """StyleRev CMS - Expert Annotation System

A tool for expert review and annotation of AI-generated
style corrections for CMS physics papers.

Features:
• LaTeX document parsing with physics notation preservation
• AI-powered style suggestion generation
• Expert annotation interface
• Project management and export capabilities

Version: 1.0.0 (Development)
Architecture: Model-View pattern"""

        messagebox.showinfo("About StyleRev CMS", about_text)

    def _register_file_type(self):
        """Register .cms file type with Windows (requires admin privileges)."""
        if sys.platform != 'win32':
            messagebox.showinfo("Info", "File type registration is only available on Windows.")
            return

        try:
            import winreg

            icon_path = str(PROJECT_ROOT / "resources" / "CMS_Icon.ico")
            app_path = sys.executable

            # Confirm with user
            result = messagebox.askyesno(
                "Register File Type",
                "This will register .cms files with this application and set the custom icon.\n\n"
                "This may require administrator privileges.\n\n"
                "Do you want to continue?"
            )

            if not result:
                return

            # Register .cms extension
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\.cms") as key:
                winreg.SetValue(key, "", winreg.REG_SZ, "StyleRevCMS.Project")

            # Register file type
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\StyleRevCMS.Project") as key:
                winreg.SetValue(key, "", winreg.REG_SZ, "StyleRev CMS Project")

            # Set icon
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\StyleRevCMS.Project\DefaultIcon") as key:
                winreg.SetValue(key, "", winreg.REG_SZ, icon_path)

            # Set open command
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\StyleRevCMS.Project\shell\open\command") as key:
                winreg.SetValue(key, "", winreg.REG_SZ, f'"{app_path}" "%1"')

            messagebox.showinfo(
                "Success",
                ".cms file type registered successfully!\n\n"
                "You may need to restart Windows Explorer or log out/in for changes to take effect."
            )
            self.show_status(".cms file type registered successfully")

        except PermissionError:
            messagebox.showerror(
                "Permission Error",
                "Failed to register file type. Please run the application as administrator."
            )
        except Exception as e:
            logger.error(f"Failed to register file type: {e}")
            messagebox.showerror("Error", f"Failed to register file type:\n{str(e)}")

    def _on_closing(self):
        """Handle application closing."""
        if self._save_project():
            logger.info("Expert Annotation System closing")
            self.root.quit()
            self.root.destroy()

    # Settings Management Methods

    def _load_default_settings(self):
        """Load default settings on startup."""
        # Try to load global (default) settings first
        global_settings = self.project_manager.load_global_settings()

        if global_settings:
            self.current_settings = global_settings
            # Update global settings manager
            settings_manager.update_settings(global_settings)
            logger.info("Loaded global default settings")
        else:
            # Use default settings
            self.current_settings = ApplicationSettings()
            settings_manager.update_settings(self.current_settings)
            logger.info("Using default settings")

    def _detect_model_provider(self, model_name: Optional[str]) -> Optional[str]:
        """Determine the provider name based on the configured model string."""
        if not model_name:
            return None

        provider = None
        if ":" in model_name:
            provider = model_name.split(":", 1)[0].lower()
        else:
            model_lower = model_name.lower()
            if model_lower.startswith("claude"):
                provider = "anthropic"
            elif model_lower.startswith(("gpt", "o1", "o3", "o4")):
                provider = "openai"
            elif model_lower.startswith("anthropic"):
                provider = "anthropic"
            elif model_lower.startswith("openai"):
                provider = "openai"

        if provider in ("anthropic", "claude"):
            return "anthropic"
        if provider == "openai":
            return "openai"
        return None

    def _get_missing_api_keys(self) -> List[str]:
        """Return human-readable descriptions of API keys that are required but missing."""
        settings = self.current_settings or ApplicationSettings()
        provider = self._detect_model_provider(settings.model_settings.llm_model)

        missing = []
        if provider == "anthropic" and not settings_manager.anthropic_api_key:
            missing.append("Anthropic API key")
        if provider == "openai" and not settings_manager.openai_api_key:
            missing.append("OpenAI API key")
        return missing

    def _ensure_api_keys_available(self, prompt_user: bool = True) -> bool:
        """Ensure required API keys are configured, optionally prompting the user."""
        missing = self._get_missing_api_keys()
        if not missing:
            return True

        if prompt_user:
            missing_text = "\n - ".join(missing)
            prompt = (
                "The selected language models require the following credentials:\n"
                f" - {missing_text}\n\n"
                "Open Project Settings now to configure them?"
            )
            if messagebox.askyesno("API Keys Required", prompt):
                self._open_project_settings()

        self.show_status("API keys must be set in Project Settings before running AI corrections.")
        return False

    def _open_rulebook_editor(self):
        """Open the rulebook editor for the current project."""
        if not self.current_project:
            messagebox.showwarning("No Project", "Please load a project first to edit the rulebook.")
            return

        def on_rulebook_save(rulebook):
            """Handle rulebook changes."""
            self.current_project.rulebook = rulebook
            # Mark project as modified
            if self.current_file_path:
                self.root.title(f"StyleRev CMS - {self.current_project.metadata.name} *")
            # Auto-save
            self._save_project()
            self.show_status(f"Rulebook updated with {rulebook.get_rule_count()} rules")

        # Open rulebook editor window with current RAG model
        rag_model = self.current_settings.model_settings.rag_model if self.current_settings else settings_manager.rag_model
        editor = RulebookEditorWindow(
            self.root,
            self.current_project.rulebook,
            on_save=on_rulebook_save,
            rag_model=rag_model
        )

    def _open_project_settings(self):
        """Open the project-specific settings dialog."""
        if not self.current_project:
            messagebox.showwarning("No Project", "Please load a project first to configure project settings.")
            return

        # Use current settings as base
        current_settings = self.current_settings or ApplicationSettings()

        def on_save(new_settings: ApplicationSettings):
            # Check if RAG model changed
            old_rag = current_settings.model_settings.rag_model
            new_rag = new_settings.model_settings.rag_model
            rag_model_changed = old_rag != new_rag

            # Save project settings (embedded in current project)
            self.current_settings = new_settings
            settings_manager.update_settings(new_settings)
            # Update annotation service with new model settings
            self.annotation_service.update_models(
                llm_model=new_settings.model_settings.llm_model,
                rag_model=new_settings.model_settings.rag_model
            )

            # Update model info display
            self._update_model_info_display()

            # Re-embed rulebook if RAG model changed
            if rag_model_changed and self.current_project.rulebook.get_rule_count() > 0:
                if messagebox.askyesno(
                    "Re-embed Rulebook",
                    f"RAG model changed from '{old_rag}' to '{new_rag}'.\n\n"
                    f"The rulebook ({self.current_project.rulebook.get_rule_count()} rules) needs to be re-embedded "
                    f"with the new model for semantic search to work correctly.\n\n"
                    "Re-embed now?"
                ):
                    self._reembed_rulebook()

            self.show_status("Project settings saved successfully")
            # Mark project as modified
            if self.current_file_path:
                self.root.title(f"StyleRev CMS - {self.current_project.metadata.name} *")

        def on_save_as_default(new_settings: ApplicationSettings):
            # Save as default settings for new projects
            if self.project_manager.save_global_settings(new_settings):
                self.show_status("Settings saved as default successfully")
            else:
                messagebox.showerror("Error", "Failed to save default settings")

        dialog = SettingsDialog(
            self.root,
            current_settings=current_settings,
            on_save=on_save,
            on_save_as_default=on_save_as_default
        )

    def _load_project_into_ui(self, project: Project, settings: ApplicationSettings):
        """Load a project into the UI."""
        logger.info(f"Loading project: {project.metadata.name}")
        self.current_project = project
        self.current_settings = settings

        # Update global settings manager so services read the latest values
        settings_manager.update_settings(settings)

        # Update annotation service with project's model settings
        self.annotation_service.update_models(
            llm_model=settings.model_settings.llm_model,
            rag_model=settings.model_settings.rag_model
        )

        # Load project into views
        self.paragraph_list.load_project(project)
        self.annotation_panel.clear_paragraph()

        # Update model info display
        self._update_model_info_display()

        self.show_status(f"Loaded project: {project.metadata.name}")
        logger.info(f"Applied project settings")
        self._ensure_api_keys_available()

    def _reembed_rulebook(self):
        """Re-embed all rules in the rulebook with the current RAG model."""
        if not self.current_project or not self.current_project.rulebook:
            return

        rulebook = self.current_project.rulebook
        rule_count = rulebook.get_rule_count()

        if rule_count == 0:
            return

        # Create progress dialog
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Re-embedding Rulebook")
        progress_window.geometry("400x150")
        progress_window.transient(self.root)
        progress_window.grab_set()

        ttk.Label(
            progress_window,
            text=f"Re-embedding {rule_count} rules...",
            font=('', 10, 'bold')
        ).pack(pady=(20, 10))

        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100, length=300)
        progress_bar.pack(pady=10)

        status_label = ttk.Label(progress_window, text="Preparing...")
        status_label.pack(pady=5)

        def progress_callback(current, total, message):
            progress_var.set((current / total) * 100 if total > 0 else 0)
            status_label.config(text=message)
            progress_window.update_idletasks()

        def reembed():
            try:
                # Clear existing embeddings
                for rule in rulebook.rules:
                    rule._rule_embedding = None
                    rule._context_embedding = None

                # Re-generate embeddings
                from models.rule_models import Rule, EMBEDDING_AVAILABLE
                if EMBEDDING_AVAILABLE:
                    Rule.generate_embeddings_batch(rulebook.rules, progress_callback=progress_callback)

                self.root.after(0, lambda: self._on_reembed_complete(progress_window, True, ""))
            except Exception as e:
                self.root.after(0, lambda: self._on_reembed_complete(progress_window, False, str(e)))

        thread = threading.Thread(target=reembed, daemon=True)
        thread.start()

    def _on_reembed_complete(self, progress_window, success, error_msg):
        """Handle completion of re-embedding."""
        progress_window.destroy()

        if success:
            self._save_project()
            messagebox.showinfo("Success", "Rulebook re-embedded successfully!")
            self.show_status(f"Rulebook re-embedded with {self.current_project.rulebook.get_rule_count()} rules")
        else:
            messagebox.showerror("Error", f"Failed to re-embed rulebook: {error_msg}")

    def run(self):
        """Run the application."""
        # Set up window closing handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Set minimum window size
        self.root.minsize(1000, 600)

        # Start the GUI event loop
        logger.info("Starting Expert Annotation System...")
        self.show_status("Ready - Load a LaTeX file to begin")
        self.root.mainloop()


def main():
    """Main entry point."""
    try:
        app = ExpertAnnotationApp()
        app.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
        messagebox.showerror("Fatal Error", f"Application failed to start: {str(e)}")


if __name__ == "__main__":
    main()
