"""
Paragraph List View Component.

This component displays all paragraphs in a project and allows selection.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, List, Dict
import logging

from models import Project, Paragraph, Status, ExpertAction

logger = logging.getLogger(__name__)


class ParagraphListView(ttk.Frame):
    """
    GUI component for displaying and selecting paragraphs.

    This component provides:
    - Scrollable list of all paragraphs
    - Visual indicators for AI processing and expert review status
    - Paragraph selection
    - Basic statistics display
    """

    def __init__(self, parent, on_paragraph_selected: Callable[[Paragraph], None]):
        """
        Initialize the paragraph list view.

        Args:
            parent: The parent tkinter widget
            on_paragraph_selected: Callback function called when a paragraph is selected
        """
        super().__init__(parent)
        self.on_paragraph_selected = on_paragraph_selected
        self.current_project: Optional[Project] = None
        self.paragraphs: List[Paragraph] = []

        self._selected_paragraph_id: Optional[str] = None
        self._item_by_paragraph_id: Dict[str, str] = {}
        self.on_pause_requested: Callable[[], None] = lambda: None
        self.on_resume_requested: Callable[[], None] = lambda: None
        self._create_widgets()

    def _create_widgets(self):
        """Create and layout the GUI widgets."""
        # Header with title and stats
        header_frame = ttk.Frame(self)
        header_frame.pack(fill=tk.X, padx=5, pady=(5, 0))

        ttk.Label(
            header_frame,
            text="Document Paragraphs",
            font=('TkDefaultFont', 10, 'bold')
        ).pack(side=tk.LEFT)

        self.stats_label = ttk.Label(header_frame, text="")
        self.stats_label.pack(side=tk.RIGHT)

        # Main list frame with scrollbar
        list_frame = ttk.Frame(self)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create treeview for paragraph list
        columns = ('para_num', 'section', 'type', 'status', 'suggestions', 'action')
        self.tree = ttk.Treeview(
            list_frame,
            columns=columns,
            show='headings',  # Don't show tree column
            selectmode='browse'
        )

        # Configure columns with minimal widths for cleaner look
        self.tree.heading('para_num', text='#', anchor=tk.CENTER)
        self.tree.heading('section', text='Section', anchor=tk.W)
        self.tree.heading('type', text='Type', anchor=tk.W)
        self.tree.heading('status', text='Status', anchor=tk.CENTER)
        self.tree.heading('suggestions', text='Reviewed', anchor=tk.CENTER)
        self.tree.heading('action', text='Action', anchor=tk.CENTER)

        # Set column widths - auto-fit to content
        self.tree.column('para_num', width=50, minwidth=40, stretch=False)
        self.tree.column('section', width=250, minwidth=150)
        self.tree.column('type', width=120, minwidth=80)
        self.tree.column('status', width=100, minwidth=80, stretch=False)
        self.tree.column('suggestions', width=80, minwidth=60, stretch=False)
        self.tree.column('action', width=100, minwidth=80, stretch=False)

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=v_scrollbar.set)

        h_scrollbar = ttk.Scrollbar(list_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(xscrollcommand=h_scrollbar.set)
        self.tree.tag_configure('current_selection', background='#D6E4FF')

        # Pack treeview and scrollbars
        self.tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')

        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        # Bind selection event
        self.tree.bind('<<TreeviewSelect>>', self._on_tree_select)

        # Bind double-click event for marking paragraphs as done
        self.tree.bind('<Double-Button-1>', self._on_double_click)

        # Action buttons frame
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        self.process_all_button = ttk.Button(
            button_frame,
            text="AI Process All Paragraphs",
            command=self._process_all_paragraphs,
            state=tk.DISABLED
        )
        self.process_all_button.pack(side=tk.LEFT)

        ttk.Button(
            button_frame,
            text="Refresh",
            command=self._refresh_display
        ).pack(side=tk.LEFT, padx=(5, 0))

        # Progress and control frame
        progress_frame = ttk.Frame(self)
        progress_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        self.processing_label = ttk.Label(progress_frame, text="Idle")
        self.processing_label.pack(side=tk.LEFT, padx=(0, 5))

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            length=180
        )
        self.progress_bar.pack(side=tk.LEFT, padx=(0, 5))

        self.pause_button = ttk.Button(
            progress_frame,
            text="Pause",
            command=self._pause_processing,
            state=tk.DISABLED
        )
        self.pause_button.pack(side=tk.LEFT)

        self.resume_button = ttk.Button(
            progress_frame,
            text="Resume",
            command=self._resume_processing,
            state=tk.DISABLED
        )
        self.resume_button.pack(side=tk.LEFT, padx=(5, 0))

    def load_project(self, project: Project):
        """
        Load paragraphs from a project.

        Args:
            project: The project to load
        """
        self.current_project = project
        self.paragraphs = project.paragraphs
        self._refresh_display()
        self.process_all_button.config(state=tk.NORMAL)

    def clear_project(self):
        """Clear the current project and paragraph list."""
        self.current_project = None
        self.paragraphs = []
        self._refresh_display()
        self.process_all_button.config(state=tk.DISABLED)
        self._selected_paragraph_id = None

    def _refresh_display(self):
        """Refresh the paragraph list display."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        if not self.paragraphs:
            self._update_stats()
            self._apply_selection_highlight()
            return

        self._item_by_paragraph_id = {}
        # Add paragraphs to tree
        for i, paragraph in enumerate(self.paragraphs):
            # Status indicators
            status_icon = self._get_status_icon(paragraph.status)
            status_text = f"{status_icon} {paragraph.status.value}"

            # Suggestion count with review status
            suggestion_count = len(paragraph.suggestions)
            # Count reviewed suggestions: any suggestion that is NOT pending
            reviewed_count = sum(1 for s in paragraph.suggestions if s.expert_action != ExpertAction.PENDING)
            if suggestion_count > 0:
                suggestions_text = f"{reviewed_count}/{suggestion_count}"
            else:
                suggestions_text = "0"

            # Section display
            section_text = f"{paragraph.section_number}. {paragraph.section_title}"
            if paragraph.subsection_title:
                section_text += f" > {paragraph.subsection_title}"

            # Action text
            action_text = "" if paragraph.status == Status.DONE else "[Mark Done]"

            # Insert item with para_num as first column
            item_id = self.tree.insert(
                '',
                tk.END,
                values=(
                    f"{i+1}",  # Paragraph number
                    section_text,
                    paragraph.paragraph_type.value,
                    status_text,
                    suggestions_text,
                    action_text
                ),
                tags=(paragraph.id,)
            )
            self._item_by_paragraph_id[paragraph.id] = item_id

        self._update_stats()
        if self._selected_paragraph_id and self._selected_paragraph_id in self._item_by_paragraph_id:
            target_item = self._item_by_paragraph_id[self._selected_paragraph_id]
            self.tree.selection_set(target_item)
            self.tree.see(target_item)
        else:
            self._apply_selection_highlight()

    def _get_status_icon(self, status: Status) -> str:
        """Get icon for paragraph status."""
        icons = {
            Status.NOT_PROCESSED: "○",
            Status.CHANGED: "⚠",
            Status.DONE: "✓",
            Status.ERROR: "X"
        }
        return icons.get(status, "○")

    def _update_stats(self):
        """Update the statistics display."""
        if not self.paragraphs:
            self.stats_label.config(text="No paragraphs loaded")
            return

        total = len(self.paragraphs)
        done = sum(1 for p in self.paragraphs if p.status == Status.DONE)
        changed = sum(1 for p in self.paragraphs if p.status == Status.CHANGED)
        reviewed = sum(1 for p in self.paragraphs
                      if p.expert_annotations or any(s.source == 'human' for s in p.suggestions))

        stats_text = f"Total: {total} | Done: {done} | Changed: {changed} | Reviewed: {reviewed}"
        self.stats_label.config(text=stats_text)

    def _on_tree_select(self, event):
        """Handle paragraph selection."""
        selection = self.tree.selection()
        if selection:
            item = selection[0]
            # Get paragraph index from item position
            item_index = self.tree.index(item)
            if 0 <= item_index < len(self.paragraphs):
                selected_paragraph = self.paragraphs[item_index]
                self._selected_paragraph_id = selected_paragraph.id
                self._apply_selection_highlight()
                logger.debug(f"Selected paragraph {item_index}: {selected_paragraph.id}")
                self.on_paragraph_selected(selected_paragraph)

    def _on_double_click(self, event):
        """Handle double-click on tree items to mark paragraphs as done."""
        # Identify which row and column was clicked
        region = self.tree.identify('region', event.x, event.y)
        if region != 'cell':
            return

        column = self.tree.identify_column(event.x)
        item = self.tree.identify_row(event.y)

        if not item:
            return

        # Check if the action column was clicked (column #6)
        if column == '#6':  # Action column
            item_index = self.tree.index(item)
            if 0 <= item_index < len(self.paragraphs):
                paragraph = self.paragraphs[item_index]
                # Toggle status: if DONE, set to CHANGED; otherwise set to DONE
                if paragraph.status == Status.DONE:
                    paragraph.status = Status.CHANGED
                    logger.info(f"Unmarked paragraph {item_index} as done")
                else:
                    # Check if all suggestions are reviewed before marking as done
                    total_suggestions = len(paragraph.suggestions)
                    if total_suggestions > 0:
                        reviewed_suggestions = sum(1 for s in paragraph.suggestions if s.expert_action != ExpertAction.PENDING)
                        if reviewed_suggestions < total_suggestions:
                            # Not all suggestions reviewed, show warning
                            from tkinter import messagebox
                            messagebox.showwarning(
                                "Cannot Mark as Done",
                                f"Cannot mark paragraph as Done.\n\n"
                                f"All suggestions must be reviewed first.\n"
                                f"Currently reviewed: {reviewed_suggestions}/{total_suggestions}",
                                parent=self
                            )
                            return

                    paragraph.status = Status.DONE
                    logger.info(f"Marked paragraph {item_index} as done")

                # Refresh display to show updated status
                self._refresh_display()

    def _process_all_paragraphs(self):
        """Handle processing all paragraphs with AI."""
        if hasattr(self.master, 'process_all_paragraphs'):
            self.master.process_all_paragraphs()
        else:
            logger.warning("No process_all_paragraphs method found on master")

    def _pause_processing(self):
        """Request pause for the batch processing."""
        self.on_pause_requested()

    def _resume_processing(self):
        """Request resume for the batch processing."""
        self.on_resume_requested()

    def set_process_state(self, running: bool, total: int = 0, completed: int = 0):
        """Update UI for processing state."""
        self.process_all_button.config(state=tk.DISABLED if running else tk.NORMAL)
        if running:
            self.pause_button.config(state=tk.NORMAL)
            self.resume_button.config(state=tk.DISABLED)
        else:
            self.pause_button.config(state=tk.DISABLED)
            self.resume_button.config(state=tk.DISABLED)
        if running:
            self.processing_label.config(text=f"Processing 0/{total}")
            self.update_processing_progress(completed, total)
        else:
            self.progress_var.set(0.0)
            self.processing_label.config(text="Idle")
            self.pause_button.config(state=tk.DISABLED)
            self.resume_button.config(state=tk.DISABLED)

    def update_processing_progress(self, completed: int, total: int):
        """Update progress bar and label."""
        if total <= 0:
            self.progress_var.set(0.0)
            self.processing_label.config(text="Idle")
            return
        percent = min(100.0, (completed / total) * 100)
        self.progress_var.set(percent)
        self.processing_label.config(text=f"Processing {completed}/{total}")

    def set_process_paused(self, paused: bool):
        """Adjust controls when paused/resumed."""
        if paused:
            self.pause_button.config(state=tk.DISABLED)
            self.resume_button.config(state=tk.NORMAL)
            self.processing_label.config(text="Paused")
        else:
            self.pause_button.config(state=tk.NORMAL)
            self.resume_button.config(state=tk.DISABLED)
            # label will be updated by update_processing_progress

    def refresh_paragraph(self, paragraph: Paragraph):
        """
        Refresh the display of a specific paragraph.

        Args:
            paragraph: The paragraph that was updated
        """
        # Find the paragraph in our list and refresh the entire display
        # This is a simple implementation - could be optimized to update just one item
        if paragraph in self.paragraphs:
            self._refresh_display()

    def get_selected_paragraph(self) -> Optional[Paragraph]:
        """
        Get the currently selected paragraph.

        Returns:
            Optional[Paragraph]: The selected paragraph, or None if no selection
        """
        selection = self.tree.selection()
        if selection:
            item_index = self.tree.index(selection[0])
            if 0 <= item_index < len(self.paragraphs):
                return self.paragraphs[item_index]
        return None

    def _apply_selection_highlight(self):
        """Visually highlight the currently selected paragraph."""
        for item in self.tree.get_children():
            tags = list(self.tree.item(item, 'tags'))
            if 'current_selection' in tags:
                tags.remove('current_selection')
                self.tree.item(item, tags=tuple(tags))

        if self._selected_paragraph_id and self._selected_paragraph_id in self._item_by_paragraph_id:
            item_id = self._item_by_paragraph_id[self._selected_paragraph_id]
            tags = list(self.tree.item(item_id, 'tags'))
            if 'current_selection' not in tags:
                tags.append('current_selection')
                self.tree.item(item_id, tags=tuple(tags))
