"""
Annotation Panel Component with Card-Based UI.

This component displays suggestions as color-coded cards that can be sorted and interacted with.
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from typing import Callable, Optional, List
import logging

from models import Paragraph, Suggestion, ExpertAction, IssueType, IssueSeverity, Status

logger = logging.getLogger(__name__)


# Color scheme for different statuses
STATUS_COLORS = {
    ExpertAction.PENDING:'#F0F0F0',   # Grey (default)
    ExpertAction.ACCEPT: '#C8E6C9',   # Light green
    ExpertAction.REJECT: '#FFCDD2',   # Light red
    ExpertAction.MODIFY: '#FFF9C4',   # Yellow
    ExpertAction.HUMAN_SUGGESTED: '#BBDEFB',  # Blue (for expert suggestions)
    ExpertAction.IMPLEMENTED: '#81C784'  # Deeper green for applied suggestions
}


class SuggestionCard(tk.Frame):
    """
    A widget representing a single suggestion as a card.

    Cards show the suggestion details and provide action buttons.
    Each card has its own background color based on status.
    """

    def __init__(self, parent, suggestion: Suggestion, on_action: Callable[[Suggestion, ExpertAction, Optional[str]], None],
                 on_click: Optional[Callable[[Suggestion], None]] = None,
                 on_undo: Optional[Callable[[Suggestion], None]] = None):
        """
        Initialize the suggestion card.

        Args:
            parent: Parent widget
            suggestion: The suggestion to display
            on_action: Callback when action button is clicked (suggestion, action, modified_text)
            on_click: Callback when card is clicked (suggestion)
        """
        super().__init__(parent, relief=tk.RAISED, borderwidth=2, cursor='hand2')
        self.suggestion = suggestion
        self.on_action = on_action
        self.on_click = on_click
        self.on_undo = on_undo

        # Store widget references for color updates
        self.container = None
        self.header_frame = None
        self.rule_frame = None
        self.text_frame = None
        self.buttons_frame = None
        self.meta_frame = None

        # Store text widget references for updates
        self.suggested_label = None
        self.suggested_text = None

        # Store button references for enabling/disabling
        self.accept_btn = None
        self.reject_btn = None

        self._create_widgets()
        self._update_colors()
        self._bind_click_events()

    def _create_widgets(self):
        """Create the card widgets."""
        # Main container with padding - using tk.Frame for background color
        self.container = tk.Frame(self)
        self.container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header: Rule/Type and Source Badge
        self.header_frame = tk.Frame(self.container)
        self.header_frame.pack(fill=tk.X, pady=(0, 5))

        title_label = tk.Label(
            self.header_frame,
            text=self.suggestion.rule_title,
            font=('TkDefaultFont', 10, 'bold')
        )
        title_label.pack(side=tk.LEFT)

        # Source badge
        if self.suggestion.source == 'human':
            badge = tk.Label(
                self.header_frame,
                text="EXPERT",
                bg='#0D47A1',
                fg='white',
                font=('TkDefaultFont', 8, 'bold'),
                padx=5,
                pady=2
            )
            badge.pack(side=tk.RIGHT, padx=(5, 0))
        else:
            # Confidence for AI suggestions
            confidence_text = f"{self.suggestion.confidence:.0%}"
            confidence_label = tk.Label(
                self.header_frame,
                text=f"Confidence: {confidence_text}",
                font=('TkDefaultFont', 9)
            )
            confidence_label.pack(side=tk.RIGHT)

        # Rule content / Description
        self.rule_frame = tk.Frame(self.container)
        self.rule_frame.pack(fill=tk.X, pady=(0, 10))

        rule_text = tk.Text(
            self.rule_frame,
            height=2,
            wrap=tk.WORD,
            font=('TkDefaultFont', 9),
            relief=tk.FLAT,
            bg='#F0F0F0'
        )
        rule_text.insert('1.0', self.suggestion.rule_content)
        rule_text.config(state=tk.DISABLED)
        rule_text.pack(fill=tk.X)

        # Original and Suggested text
        self.text_frame = tk.Frame(self.container)
        self.text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Original (red text on light background)
        original_label = tk.Label(self.text_frame, text="Original:", font=('TkDefaultFont', 9, 'bold'))
        original_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 2))

        original_text = tk.Text(
            self.text_frame,
            height=2,
            wrap=tk.WORD,
            font=('TkDefaultFont', 9),
            bg='#F0F0F0',
            fg='#B71C1C'
        )
        original_text.insert('1.0', self.suggestion.original)
        original_text.config(state=tk.DISABLED)
        original_text.grid(row=1, column=0, sticky=tk.EW, pady=(0, 5))

        # Suggested (green text on light background)
        # Determine label text based on whether suggestion was modified
        suggested_label_text = "Suggested (modified):" if self.suggestion.modified_text else "Suggested:"
        self.suggested_label = tk.Label(self.text_frame, text=suggested_label_text, font=('TkDefaultFont', 9, 'bold'))
        self.suggested_label.grid(row=2, column=0, sticky=tk.W, pady=(0, 2))

        self.suggested_text = tk.Text(
            self.text_frame,
            height=2,
            wrap=tk.WORD,
            font=('TkDefaultFont', 9),
            bg='#F0F0F0',
            fg='#1B5E20'
        )
        # Show modified text if available, otherwise show original suggested text
        display_text = self.suggestion.modified_text if self.suggestion.modified_text else self.suggestion.suggested
        self.suggested_text.insert('1.0', display_text)
        self.suggested_text.config(state=tk.DISABLED)
        self.suggested_text.grid(row=3, column=0, sticky=tk.EW)

        self.text_frame.columnconfigure(0, weight=1)

        # Action buttons (only for AI suggestions or pending human suggestions)
        if self.suggestion.expert_action == ExpertAction.IMPLEMENTED:
            self.meta_frame = tk.Frame(self.container)
            self.meta_frame.pack(fill=tk.X, pady=(10, 0))

            implemented_label = tk.Label(
                self.meta_frame,
                text="Implemented (locked)",
                font=('TkDefaultFont', 9, 'bold'),
                fg='#1B5E20'
            )
            implemented_label.pack(side=tk.LEFT)

            if self.on_undo:
                undo_btn = ttk.Button(
                    self.meta_frame,
                    text="↩ Undo",
                    command=self._handle_undo,
                    width=10
                )
                undo_btn.pack(side=tk.RIGHT, padx=(5, 0))

            delete_btn = ttk.Button(
                self.meta_frame,
                text="🗑 Delete",
                command=self._handle_delete,
                width=10
            )
            delete_btn.pack(side=tk.RIGHT)

        elif self.suggestion.expert_action != ExpertAction.HUMAN_SUGGESTED:
            self.buttons_frame = tk.Frame(self.container)
            self.buttons_frame.pack(fill=tk.X, pady=(10, 0))

            # Accept button
            self.accept_btn = ttk.Button(
                self.buttons_frame,
                text="✓ Accept",
                command=lambda: self._handle_action(ExpertAction.ACCEPT),
                width=12
            )
            self.accept_btn.pack(side=tk.LEFT, padx=(0, 5))

            # Reject button
            self.reject_btn = ttk.Button(
                self.buttons_frame,
                text="✗ Reject",
                command=lambda: self._handle_action(ExpertAction.REJECT),
                width=12
            )
            self.reject_btn.pack(side=tk.LEFT, padx=(0, 5))

            # Disable Accept/Reject if suggestion has been modified
            if self.suggestion.expert_action == ExpertAction.MODIFY:
                self.accept_btn.config(state=tk.DISABLED)
                self.reject_btn.config(state=tk.DISABLED)

            # Modify button
            modify_btn = ttk.Button(
                self.buttons_frame,
                text="✎ Modify",
                command=self._handle_modify,
                width=12
            )
            modify_btn.pack(side=tk.LEFT, padx=(0, 5))

            # Reset button (always show for non-human suggestions)
            reset_btn = ttk.Button(
                self.buttons_frame,
                text="↺ Reset",
                command=lambda: self._handle_action(ExpertAction.PENDING),
                width=12
            )
            reset_btn.pack(side=tk.LEFT, padx=(0, 5))

            # Delete button (for all suggestions)
            delete_btn = ttk.Button(
                self.buttons_frame,
                text="🗑 Delete",
                command=self._handle_delete,
                width=10
            )
            delete_btn.pack(side=tk.LEFT)
        else:
            # For human suggestions, show metadata and delete button
            self.meta_frame = tk.Frame(self.container)
            self.meta_frame.pack(fill=tk.X, pady=(10, 0))

            if self.suggestion.severity:
                severity_label = tk.Label(
                    self.meta_frame,
                    text=f"Severity: {self.suggestion.severity.value.title()}",
                    font=('TkDefaultFont', 9)
                )
                severity_label.pack(side=tk.LEFT, padx=(0, 10))

            if self.suggestion.issue_type:
                type_label = tk.Label(
                    self.meta_frame,
                    text=f"Type: {self.suggestion.issue_type.value.replace('_', ' ').title()}",
                    font=('TkDefaultFont', 9)
                )
                type_label.pack(side=tk.LEFT)

            # Delete button for human suggestions
            delete_btn = ttk.Button(
                self.meta_frame,
                text="🗑 Delete",
                command=self._handle_delete,
                width=10
            )
            delete_btn.pack(side=tk.RIGHT)

    def _handle_action(self, action: ExpertAction):
        """Handle action button click."""
        self.suggestion.expert_action = action
        # Clear modified text when resetting to PENDING
        if action == ExpertAction.PENDING:
            self.suggestion.modified_text = None
            self._update_suggested_text()
            # Re-enable Accept/Reject buttons when resetting
            if self.accept_btn:
                self.accept_btn.config(state=tk.NORMAL)
            if self.reject_btn:
                self.reject_btn.config(state=tk.NORMAL)
        self._update_colors()
        self.on_action(self.suggestion, action, None)

    def _handle_modify(self):
        """Handle modify button click."""
        # Get modified text from user
        modified = simpledialog.askstring(
            "Modify Suggestion",
            "Enter your modification:",
            initialvalue=self.suggestion.suggested,
            parent=self
        )

        if modified and modified != self.suggestion.suggested:
            self.suggestion.expert_action = ExpertAction.MODIFY
            self.suggestion.modified_text = modified
            self._update_colors()
            self._update_suggested_text()
            # Disable Accept/Reject buttons when modified
            if self.accept_btn:
                self.accept_btn.config(state=tk.DISABLED)
            if self.reject_btn:
                self.reject_btn.config(state=tk.DISABLED)
            self.on_action(self.suggestion, ExpertAction.MODIFY, modified)

    def _handle_delete(self):
        """Handle delete button click for suggestions."""
        # Confirm deletion
        suggestion_type = "expert-identified issue" if self.suggestion.source == 'human' else "AI suggestion"
        result = messagebox.askyesno(
            "Delete Suggestion",
            f"Are you sure you want to delete this {suggestion_type}?",
            parent=self
        )

        if result:
            # Signal deletion by passing None as action
            self.on_action(self.suggestion, None, None)
    def _handle_undo(self):
        """Request undoing an implemented suggestion."""
        if self.on_undo:
            self.on_undo(self.suggestion)

    def _update_colors(self):
        """Update card background color based on status."""
        bg_color = STATUS_COLORS.get(self.suggestion.expert_action, '#E0E0E0')

        # Update the main frame and container backgrounds
        self.configure(bg=bg_color)
        if self.container:
            self.container.configure(bg=bg_color)

        # Update all child frames to match the card background
        if self.header_frame:
            self.header_frame.configure(bg=bg_color)
        if self.rule_frame:
            self.rule_frame.configure(bg=bg_color)
        if self.text_frame:
            self.text_frame.configure(bg=bg_color)
        if self.buttons_frame:
            self.buttons_frame.configure(bg=bg_color)
        if self.meta_frame:
            self.meta_frame.configure(bg=bg_color)

    def _update_suggested_text(self):
        """Update the suggested text display when modified."""
        if self.suggested_text and self.suggested_label:
            # Update label
            label_text = "Suggested (modified):" if self.suggestion.modified_text else "Suggested:"
            self.suggested_label.config(text=label_text)

            # Update text content
            display_text = self.suggestion.modified_text if self.suggestion.modified_text else self.suggestion.suggested
            self.suggested_text.config(state=tk.NORMAL)
            self.suggested_text.delete('1.0', tk.END)
            self.suggested_text.insert('1.0', display_text)
            self.suggested_text.config(state=tk.DISABLED)

    def _bind_click_events(self):
        """Bind click events to all widgets in the card."""
        if self.on_click:
            # Bind to the card itself
            self.bind('<Button-1>', lambda e: self.on_click(self.suggestion))

            # Bind to all child widgets recursively
            def bind_recursive(widget):
                widget.bind('<Button-1>', lambda e: self.on_click(self.suggestion))
                for child in widget.winfo_children():
                    # Skip buttons to allow them to function normally
                    if not isinstance(child, (ttk.Button, tk.Button)):
                        bind_recursive(child)

            if self.container:
                bind_recursive(self.container)


class AnnotationPanel(ttk.Frame):
    """
    GUI component for expert annotation of paragraphs.

    Displays suggestions as sortable, color-coded cards.
    """

    def __init__(self, parent, on_annotation_changed: Optional[Callable[[Paragraph], None]] = None):
        """
        Initialize the annotation panel.

        Args:
            parent: The parent tkinter widget
            on_annotation_changed: Callback function called when annotations change
        """
        super().__init__(parent)
        self.on_annotation_changed = on_annotation_changed
        self.current_paragraph: Optional[Paragraph] = None
        self.suggestion_cards: List[SuggestionCard] = []

        self._create_widgets()

    def _create_widgets(self):
        """Create and layout the GUI widgets."""
        # Header
        header_frame = ttk.Frame(self)
        header_frame.pack(fill=tk.X, padx=5, pady=(5, 0))

        ttk.Label(
            header_frame,
            text="Expert Annotation Panel",
            font=('TkDefaultFont', 10, 'bold')
        ).pack(side=tk.LEFT)

        # Paragraph display section
        para_frame = ttk.LabelFrame(self, text="Paragraph Text")
        para_frame.pack(fill=tk.X, padx=5, pady=5)

        # Text widget with scrollbar
        text_frame = ttk.Frame(para_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.paragraph_text = tk.Text(
            text_frame,
            height=6,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=('TkDefaultFont', 10)
        )
        para_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.paragraph_text.yview)
        self.paragraph_text.configure(yscrollcommand=para_scrollbar.set)

        # Configure highlight tags
        self.paragraph_text.tag_configure('highlight', background='yellow', foreground='black')
        self.paragraph_text.tag_configure('implemented', background='#D5F5E3')

        self.paragraph_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        para_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Suggestions section (scrollable cards)
        suggestions_label_frame = ttk.LabelFrame(self, text="")
        suggestions_label_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Suggestions header with title and Add Issue button
        suggestions_header = ttk.Frame(suggestions_label_frame)
        suggestions_header.pack(fill=tk.X, padx=5, pady=(5, 5))

        ttk.Label(
            suggestions_header,
            text="Suggestions",
            font=('TkDefaultFont', 10, 'bold')
        ).pack(side=tk.LEFT)

        # AI Process button
        self.process_button = ttk.Button(
            suggestions_header,
            text="AI Process Paragraph",
            command=self._process_current_paragraph,
            state=tk.DISABLED
        )
        self.process_button.pack(side=tk.RIGHT, padx=(5, 0))

        # Implement suggestions button
        self.implement_button = ttk.Button(
            suggestions_header,
            text="Implement Suggestions",
            command=self._implement_suggestions,
            state=tk.DISABLED
        )
        self.implement_button.pack(side=tk.RIGHT, padx=(5, 0))

        # Add Expert Issue button
        self.add_issue_button = ttk.Button(
            suggestions_header,
            text="+ Add Expert Issue",
            command=self._show_add_issue_dialog,
            state=tk.DISABLED
        )
        self.add_issue_button.pack(side=tk.RIGHT)

        # Create canvas for scrolling
        self.suggestions_canvas = tk.Canvas(suggestions_label_frame, bg='white')
        suggestions_scrollbar = ttk.Scrollbar(
            suggestions_label_frame,
            orient=tk.VERTICAL,
            command=self.suggestions_canvas.yview
        )
        self.suggestions_container = ttk.Frame(self.suggestions_canvas)

        self.suggestions_container.bind(
            '<Configure>',
            lambda e: self.suggestions_canvas.configure(scrollregion=self.suggestions_canvas.bbox('all'))
        )

        self.suggestions_canvas.create_window((0, 0), window=self.suggestions_container, anchor=tk.NW)
        self.suggestions_canvas.configure(yscrollcommand=suggestions_scrollbar.set)

        self.suggestions_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        suggestions_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Mouse wheel scrolling
        self.suggestions_canvas.bind_all('<MouseWheel>', self._on_mousewheel)

        # Status bar
        self.status_label = ttk.Label(
            self,
            text="No paragraph selected",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X, padx=5, pady=5)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        self.suggestions_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def load_paragraph(self, paragraph: Paragraph):
        """
        Load a paragraph for annotation.

        Args:
            paragraph: The paragraph to display
        """
        self.current_paragraph = paragraph
        self._refresh_display()
        self.process_button.config(state=tk.NORMAL)
        self.add_issue_button.config(state=tk.NORMAL)
        self._update_action_buttons_state()

    def clear_paragraph(self):
        """Clear the current paragraph display."""
        self.current_paragraph = None
        self.paragraph_text.config(state=tk.NORMAL)
        self.paragraph_text.delete('1.0', tk.END)
        self.paragraph_text.config(state=tk.DISABLED)
        self._clear_suggestions()
        self.process_button.config(state=tk.DISABLED)
        self.add_issue_button.config(state=tk.DISABLED)
        self.implement_button.config(state=tk.DISABLED)
        self.status_label.config(text="No paragraph selected")

    def _refresh_display(self):
        """Refresh the entire display."""
        if not self.current_paragraph:
            return

        # Update paragraph text
        self.paragraph_text.config(state=tk.NORMAL)
        self.paragraph_text.delete('1.0', tk.END)
        self.paragraph_text.insert('1.0', self.current_paragraph.text)
        self.paragraph_text.config(state=tk.DISABLED)

        # Update suggestions
        self._display_suggestions()

        # Highlight implemented text and update headers
        self._highlight_implemented_text()
        self._update_status_summary()

    def _display_suggestions(self):
        """Display all suggestions as sorted cards."""
        # Clear existing cards
        self._clear_suggestions()

        if not self.current_paragraph or not self.current_paragraph.suggestions:
            self._update_action_buttons_state()
            return

        # Sort suggestions: PENDING first, then others; AI before human within same status
        sorted_suggestions = self._sort_suggestions(self.current_paragraph.suggestions)

        # Create cards
        for suggestion in sorted_suggestions:
            card = SuggestionCard(
                self.suggestions_container,
                suggestion,
                self._on_card_action,
                self._on_card_click,
                lambda s=suggestion: self._undo_implemented_suggestion(s)
            )
            card.pack(fill=tk.X, padx=5, pady=5)
            self.suggestion_cards.append(card)

        self._update_action_buttons_state()

    def _sort_suggestions(self, suggestions: List[Suggestion]) -> List[Suggestion]:
        """
        Sort suggestions by status and source.

        Priority order:
        1. PENDING status (including None)
        2. Other statuses
        Within same status: AI suggestions before human suggestions
        """
        def sort_key(s: Suggestion):
            status_priority_map = {
                ExpertAction.PENDING: 0,
                ExpertAction.MODIFY: 1,
                ExpertAction.ACCEPT: 1,
                ExpertAction.HUMAN_SUGGESTED: 1,
                ExpertAction.REJECT: 2,
                ExpertAction.IMPLEMENTED: 3
            }
            status_priority = status_priority_map.get(s.expert_action, 1)

            # Secondary sort: AI (0) before human (1)
            source_priority = 0 if s.source == 'ai' else 1

            # Tertiary sort: by timestamp (older first)
            time_priority = s.timestamp.timestamp() if hasattr(s, 'timestamp') and s.timestamp else 0

            return (status_priority, source_priority, time_priority)

        return sorted(suggestions, key=sort_key)

    def _clear_suggestions(self):
        """Clear all suggestion cards."""
        for card in self.suggestion_cards:
            card.destroy()
        self.suggestion_cards.clear()
        self.paragraph_text.tag_remove('highlight', '1.0', tk.END)
        self.paragraph_text.tag_remove('implemented', '1.0', tk.END)

    def _on_card_click(self, suggestion: Suggestion):
        """
        Handle click on a suggestion card to highlight the original text.

        Args:
            suggestion: The suggestion that was clicked
        """
        if not self.current_paragraph:
            return

        # Clear any existing highlights
        self.paragraph_text.tag_remove('highlight', '1.0', tk.END)

        # Find the original text in the paragraph
        original_text = suggestion.original
        paragraph_text = self.current_paragraph.text

        # Find the position of the original text
        start_idx = paragraph_text.find(original_text)
        if start_idx != -1:
            # Calculate line and column positions
            lines_before = paragraph_text[:start_idx].count('\n')
            col_start = start_idx - paragraph_text[:start_idx].rfind('\n') - 1 if '\n' in paragraph_text[:start_idx] else start_idx

            end_idx = start_idx + len(original_text)
            lines_before_end = paragraph_text[:end_idx].count('\n')
            col_end = end_idx - paragraph_text[:end_idx].rfind('\n') - 1 if '\n' in paragraph_text[:end_idx] else end_idx

            # Create text widget indices
            start_pos = f"{lines_before + 1}.{col_start}"
            end_pos = f"{lines_before_end + 1}.{col_end}"

            # Apply highlight
            self.paragraph_text.tag_add('highlight', start_pos, end_pos)

            # Scroll to make the highlighted text visible
            self.paragraph_text.see(start_pos)

            logger.debug(f"Highlighted text from {start_pos} to {end_pos}: '{original_text}'")
        else:
            logger.warning(f"Could not find original text in paragraph: '{original_text}'")

    def _get_eligible_suggestions(self) -> List[Suggestion]:
        """Return suggestions that can be implemented."""
        if not self.current_paragraph:
            return []

        eligible = []
        for suggestion in self.current_paragraph.suggestions:
            if suggestion.expert_action == ExpertAction.IMPLEMENTED:
                continue
            if suggestion.expert_action == ExpertAction.ACCEPT:
                eligible.append(suggestion)
            elif suggestion.expert_action == ExpertAction.HUMAN_SUGGESTED:
                eligible.append(suggestion)
            elif suggestion.expert_action == ExpertAction.MODIFY and (suggestion.modified_text or suggestion.suggested):
                eligible.append(suggestion)
        return eligible

    def _implement_suggestions(self):
        """Apply eligible suggestions to the paragraph text."""
        if not self.current_paragraph:
            return

        eligible = self._get_eligible_suggestions()
        if not eligible:
            messagebox.showinfo(
                "No Eligible Suggestions",
                "There are no accepted, modified, or expert-added suggestions to implement.",
                parent=self
            )
            return

        paragraph_text = self.current_paragraph.text
        applied = []
        skipped = []

        for suggestion in eligible:
            replace_with = suggestion.modified_text if (
                suggestion.expert_action == ExpertAction.MODIFY and suggestion.modified_text
            ) else suggestion.suggested

            if not replace_with or not suggestion.original:
                skipped.append(suggestion)
                continue

            start_idx = paragraph_text.find(suggestion.original)
            if start_idx == -1:
                skipped.append(suggestion)
                continue

            end_idx = start_idx + len(suggestion.original)
            occurrence_index_original = self._count_occurrences_before(paragraph_text, suggestion.original, start_idx)
            paragraph_text = paragraph_text[:start_idx] + replace_with + paragraph_text[end_idx:]

            suggestion.pre_implementation_action = suggestion.expert_action
            suggestion.expert_action = ExpertAction.IMPLEMENTED
            suggestion.implemented_start = start_idx
            suggestion.implemented_end = start_idx + len(replace_with)
            suggestion.implemented_replacement = replace_with
            suggestion.implemented_original_occurrence_index = occurrence_index_original
            suggestion.implemented_replacement_occurrence_index = self._count_occurrences_before(paragraph_text, replace_with, start_idx)

            if suggestion.id not in self.current_paragraph.implemented_suggestion_ids:
                self.current_paragraph.implemented_suggestion_ids.append(suggestion.id)

            applied.append(suggestion)

        if not applied:
            messagebox.showwarning(
                "No Suggestions Applied",
                "The selected suggestions could not be located in the paragraph.",
                parent=self
            )
            return

        self.current_paragraph.text = paragraph_text
        self.current_paragraph.status = Status.DONE

        if self.on_annotation_changed:
            self.on_annotation_changed(self.current_paragraph)

        self._refresh_display()

        summary = f"Implemented {len(applied)} suggestion(s)."
        if skipped:
            summary += f"\nSkipped {len(skipped)} suggestion(s) that could not be located."

        messagebox.showinfo("Suggestions Applied", summary, parent=self)

    def _undo_implemented_suggestion(self, suggestion: Suggestion, show_message: bool = True) -> bool:
        """Undo a previously implemented suggestion."""
        if not self.current_paragraph or suggestion.expert_action != ExpertAction.IMPLEMENTED:
            if show_message:
                messagebox.showwarning(
                    "Cannot Undo",
                    "This suggestion is not currently implemented.",
                    parent=self
                )
            return False

        paragraph_text = self.current_paragraph.text
        replace_with = suggestion.implemented_replacement or suggestion.modified_text or suggestion.suggested
        if not replace_with:
            if show_message:
                messagebox.showwarning(
                    "Cannot Undo",
                    "Replacement text is missing for this suggestion.",
                    parent=self
                )
            return False

        start = suggestion.implemented_start
        end = suggestion.implemented_end

        if start is None or end is None or end > len(paragraph_text) or start < 0 or paragraph_text[start:end] != replace_with:
            start = self._locate_replacement_segment(paragraph_text, suggestion, replace_with)
            if start == -1:
                if show_message:
                    messagebox.showwarning(
                        "Cannot Undo",
                        "Could not locate the implemented text in the paragraph.",
                        parent=self
                    )
                return False
            end = start + len(replace_with)

        paragraph_text = paragraph_text[:start] + suggestion.original + paragraph_text[end:]
        self.current_paragraph.text = paragraph_text

        suggestion.expert_action = suggestion.pre_implementation_action or ExpertAction.ACCEPT
        suggestion.pre_implementation_action = None
        suggestion.implemented_start = None
        suggestion.implemented_end = None
        suggestion.implemented_replacement = None
        suggestion.implemented_original_occurrence_index = None
        suggestion.implemented_replacement_occurrence_index = None

        if suggestion.id in self.current_paragraph.implemented_suggestion_ids:
            self.current_paragraph.implemented_suggestion_ids.remove(suggestion.id)

        self.current_paragraph.status = Status.CHANGED

        if self.on_annotation_changed:
            self.on_annotation_changed(self.current_paragraph)

        self._refresh_display()
        if show_message:
            messagebox.showinfo("Undo Successful", "Suggestion has been reverted.", parent=self)
        return True

    def _highlight_implemented_text(self):
        """Highlight implemented segments within the paragraph text."""
        self.paragraph_text.tag_remove('implemented', '1.0', tk.END)

        if not self.current_paragraph:
            return

        text = self.current_paragraph.text
        for suggestion in self.current_paragraph.suggestions:
            if suggestion.expert_action == ExpertAction.IMPLEMENTED and \
               suggestion.implemented_start is not None and \
               suggestion.implemented_end is not None:
                start_idx = self._offset_to_index(text, suggestion.implemented_start)
                end_idx = self._offset_to_index(text, suggestion.implemented_end)
                self.paragraph_text.tag_add('implemented', start_idx, end_idx)

    def _offset_to_index(self, text: str, offset: int) -> str:
        """Convert a character offset to Tkinter text index."""
        offset = max(0, min(offset, len(text)))
        before = text[:offset]
        line = before.count('\n')
        last_newline = before.rfind('\n')
        column = offset if last_newline == -1 else offset - last_newline - 1
        return f"{line + 1}.{column}"

    def _update_status_summary(self):
        """Update the footer status label with section info and applied counts."""
        if not self.current_paragraph:
            self.status_label.config(text="No paragraph selected")
            return

        section_info = self.current_paragraph.section_title
        if self.current_paragraph.subsection_title:
            section_info += f" > {self.current_paragraph.subsection_title}"

        implemented_count = sum(
            1 for s in self.current_paragraph.suggestions
            if s.expert_action == ExpertAction.IMPLEMENTED
        )
        total = len(self.current_paragraph.suggestions)
        status_text = (
            f"{section_info} | Type: {self.current_paragraph.paragraph_type.value} | "
            f"Applied: {implemented_count}/{total}"
        )
        self.status_label.config(text=status_text)

    def _update_action_buttons_state(self):
        """Enable or disable the implement button based on eligibility."""
        if not self.current_paragraph:
            self.implement_button.config(state=tk.DISABLED)
            return

        has_eligible = len(self._get_eligible_suggestions()) > 0
        self.implement_button.config(state=tk.NORMAL if has_eligible else tk.DISABLED)

    def _locate_replacement_segment(self, paragraph_text: str, suggestion: Suggestion, replacement_text: str) -> int:
        """Find the start index of the applied replacement for undo operations."""
        if not replacement_text:
            return -1

        # Try using stored occurrence index
        if suggestion.implemented_replacement_occurrence_index is not None:
            idx = self._find_nth_occurrence(paragraph_text, replacement_text,
                                            suggestion.implemented_replacement_occurrence_index)
            if idx != -1:
                return idx

        # Fallback to matching the suggested or modified text directly
        positions = self._get_occurrence_positions(paragraph_text, replacement_text)
        if positions:
            if len(positions) > 1:
                logger.warning("Multiple occurrences of replacement text found; defaulting to first occurrence.")
            return positions[0]

        alt_text = suggestion.suggested or suggestion.modified_text
        if alt_text and alt_text != replacement_text:
            if suggestion.implemented_replacement_occurrence_index is not None:
                idx = self._find_nth_occurrence(paragraph_text, alt_text,
                                                suggestion.implemented_replacement_occurrence_index)
                if idx != -1:
                    return idx
            positions = self._get_occurrence_positions(paragraph_text, alt_text)
            if positions:
                if len(positions) > 1:
                    logger.warning("Multiple occurrences of alternate text found; defaulting to first occurrence.")
                return positions[0]

        return -1

    def _count_occurrences_before(self, text: str, substring: str, end_pos: Optional[int]) -> Optional[int]:
        """Count occurrences of substring in text before end_pos."""
        if not substring or end_pos is None:
            return None
        end_pos = max(0, min(end_pos, len(text)))
        if not substring:
            return None
        return text[:end_pos].count(substring)

    def _find_nth_occurrence(self, text: str, substring: str, n: Optional[int]) -> int:
        """Find the starting index of the nth occurrence (0-based) of substring."""
        if not substring or n is None or n < 0:
            return -1
        start = 0
        for _ in range(n + 1):
            index = text.find(substring, start)
            if index == -1:
                return -1
            start = index + max(len(substring), 1)
        return index

    def _get_occurrence_positions(self, text: str, substring: str) -> List[int]:
        """Return list of all start positions for substring in text."""
        if not substring:
            return []
        positions = []
        start = 0
        while True:
            index = text.find(substring, start)
            if index == -1:
                break
            positions.append(index)
            start = index + max(len(substring), 1)
        return positions

    def _on_card_action(self, suggestion: Suggestion, action: Optional[ExpertAction], modified_text: Optional[str]):
        """
        Handle action from a suggestion card.

        Args:
            suggestion: The suggestion that was acted upon
            action: The action taken (None for deletion)
            modified_text: Modified text (for MODIFY action)
        """
        if not self.current_paragraph:
            return

        # Handle deletion
        if action is None:
            if suggestion.expert_action == ExpertAction.IMPLEMENTED:
                if not self._undo_implemented_suggestion(suggestion, show_message=False):
                    logger.warning("Cannot delete implemented suggestion because undo failed")
                    return
            # Remove the suggestion from the paragraph
            self.current_paragraph.suggestions = [
                s for s in self.current_paragraph.suggestions
                if s.id != suggestion.id
            ]
            if suggestion.id in self.current_paragraph.implemented_suggestion_ids:
                self.current_paragraph.implemented_suggestion_ids.remove(suggestion.id)
            logger.info(f"Deleted suggestion {suggestion.id}")
        else:
            # Update suggestion in paragraph
            for i, s in enumerate(self.current_paragraph.suggestions):
                if s.id == suggestion.id:
                    self.current_paragraph.suggestions[i] = suggestion
                    break
            logger.info(f"Applied {action.value} to suggestion {suggestion.id}")

        # Set status to CHANGED (even if it was DONE - any change requires re-review)
        self.current_paragraph.status = Status.CHANGED

        # Notify change
        if self.on_annotation_changed:
            self.on_annotation_changed(self.current_paragraph)

        # Re-sort and refresh display
        self._display_suggestions()
        self._highlight_implemented_text()
        self._update_status_summary()

    def _process_current_paragraph(self):
        """Process the current paragraph with AI (called from main app)."""
        if hasattr(self.master, 'process_paragraph_with_ai'):
            self.master.process_paragraph_with_ai(self.current_paragraph)

    def _show_add_issue_dialog(self):
        """Show dialog to add a new expert-identified issue."""
        if not self.current_paragraph:
            return

        dialog = AddIssueDialog(self, self.current_paragraph)
        self.wait_window(dialog)  # Wait for dialog to close

        if dialog.result:
            # Set status to CHANGED (even if it was DONE - any change requires re-review)
            self.current_paragraph.status = Status.CHANGED

            # Notify change
            if self.on_annotation_changed:
                self.on_annotation_changed(self.current_paragraph)

            # Refresh display
            self._display_suggestions()


class AddIssueDialog(tk.Toplevel):
    """Dialog for adding expert-identified issues."""

    def __init__(self, parent, paragraph: Paragraph):
        super().__init__(parent)
        self.paragraph = paragraph
        self.result = None

        self.title("Add Expert Issue")
        self.geometry("700x600")
        self.transient(parent)
        self.grab_set()

        self._create_widgets()

        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (self.winfo_width() // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")

        # Set focus to original text field
        self.original_text.focus_set()

    def _create_widgets(self):
        """Create dialog widgets."""
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Add Expert-Identified Issue",
            font=('TkDefaultFont', 12, 'bold')
        )
        title_label.pack(anchor=tk.W, pady=(0, 15))

        # Instructions
        instructions = ttk.Label(
            main_frame,
            text="Identify an issue in the paragraph and provide the corrected version.",
            foreground='gray'
        )
        instructions.pack(anchor=tk.W, pady=(0, 15))

        # Problematic text (with character count)
        original_label_frame = ttk.Frame(main_frame)
        original_label_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(
            original_label_frame,
            text="Original (Problematic) Text:",
            font=('TkDefaultFont', 9, 'bold')
        ).pack(side=tk.LEFT)

        self.original_char_count = ttk.Label(
            original_label_frame,
            text="0 chars",
            foreground='gray',
            font=('TkDefaultFont', 8)
        )
        self.original_char_count.pack(side=tk.RIGHT)

        self.original_text = tk.Text(
            main_frame,
            height=4,
            wrap=tk.WORD,
            font=('TkDefaultFont', 10),
            relief=tk.SOLID,
            borderwidth=1
        )
        self.original_text.pack(fill=tk.X, pady=(0, 15))
        self.original_text.bind('<KeyRelease>', self._update_char_count)

        # Corrected text (with character count)
        corrected_label_frame = ttk.Frame(main_frame)
        corrected_label_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(
            corrected_label_frame,
            text="Corrected Text:",
            font=('TkDefaultFont', 9, 'bold')
        ).pack(side=tk.LEFT)

        self.corrected_char_count = ttk.Label(
            corrected_label_frame,
            text="0 chars",
            foreground='gray',
            font=('TkDefaultFont', 8)
        )
        self.corrected_char_count.pack(side=tk.RIGHT)

        self.corrected_text = tk.Text(
            main_frame,
            height=4,
            wrap=tk.WORD,
            font=('TkDefaultFont', 10),
            relief=tk.SOLID,
            borderwidth=1
        )
        self.corrected_text.pack(fill=tk.X, pady=(0, 20))
        self.corrected_text.bind('<KeyRelease>', self._update_char_count)

        # Metadata section (2 columns)
        metadata_frame = ttk.Frame(main_frame)
        metadata_frame.pack(fill=tk.X, pady=(0, 15))

        # Left column - Issue Type
        left_col = ttk.Frame(metadata_frame)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        ttk.Label(left_col, text="Issue Type:", font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        self.issue_type_var = tk.StringVar(value=IssueType.CMS_STYLE.value)
        issue_type_combo = ttk.Combobox(
            left_col,
            textvariable=self.issue_type_var,
            values=[t.value.replace('_', ' ').title() for t in IssueType],
            state='readonly',
            font=('TkDefaultFont', 10)
        )
        issue_type_combo.pack(fill=tk.X)
        # Set initial display value
        issue_type_combo.set(IssueType.CMS_STYLE.value.replace('_', ' ').title())

        # Right column - Severity
        right_col = ttk.Frame(metadata_frame)
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(right_col, text="Severity:", font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        self.severity_var = tk.StringVar(value=IssueSeverity.MAJOR.value)
        severity_combo = ttk.Combobox(
            right_col,
            textvariable=self.severity_var,
            values=[s.value.title() for s in IssueSeverity],
            state='readonly',
            font=('TkDefaultFont', 10)
        )
        severity_combo.pack(fill=tk.X)
        # Set initial display value
        severity_combo.set(IssueSeverity.MAJOR.value.title())

        # Notes (optional)
        ttk.Label(main_frame, text="Notes (Optional):", font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        self.notes_text = tk.Text(
            main_frame,
            height=3,
            wrap=tk.WORD,
            font=('TkDefaultFont', 10),
            relief=tk.SOLID,
            borderwidth=1
        )
        self.notes_text.pack(fill=tk.X, pady=(0, 20))

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)

        ttk.Button(
            button_frame,
            text="Cancel",
            command=self.destroy
        ).pack(side=tk.RIGHT, padx=(5, 0))

        ttk.Button(
            button_frame,
            text="Add Issue",
            command=self._add_issue
        ).pack(side=tk.RIGHT)

    def _update_char_count(self, event=None):
        """Update character count labels."""
        original_text = self.original_text.get('1.0', tk.END).strip()
        corrected_text = self.corrected_text.get('1.0', tk.END).strip()

        self.original_char_count.config(text=f"{len(original_text)} chars")
        self.corrected_char_count.config(text=f"{len(corrected_text)} chars")

    def _add_issue(self):
        """Add the expert issue."""
        original = self.original_text.get('1.0', tk.END).strip()
        corrected = self.corrected_text.get('1.0', tk.END).strip()
        notes = self.notes_text.get('1.0', tk.END).strip() or None

        if not original or not corrected:
            messagebox.showwarning(
                "Missing Information",
                "Please provide both original (problematic) text and corrected text.",
                parent=self
            )
            return

        if original == corrected:
            messagebox.showwarning(
                "No Change",
                "The original and corrected text are the same. Please make a correction.",
                parent=self
            )
            return

        paragraph_text = self.paragraph.text or ""
        if original not in paragraph_text:
            messagebox.showerror(
                "Original Text Not Found",
                "The original text you entered could not be found in the paragraph. "
                "Please ensure you copy the exact text from the paragraph before adding the issue.",
                parent=self
            )
            return

        try:
            # Convert display values back to enum values
            issue_type_display = self.issue_type_var.get()
            severity_display = self.severity_var.get()

            # Find matching enum
            issue_type = next(t for t in IssueType if t.value.replace('_', ' ').title() == issue_type_display)
            severity = next(s for s in IssueSeverity if s.value.title() == severity_display)

            # Create human suggestion
            human_suggestion = Suggestion.create_human_suggestion(
                original_text=original,
                corrected_text=corrected,
                issue_type=issue_type,
                severity=severity,
                notes=notes
            )

            # Add to paragraph
            self.paragraph.add_suggestion(human_suggestion)

            self.result = True
            self.destroy()

            logger.info(f"Added expert issue: {issue_type.value} ({severity.value})")

        except Exception as e:
            logger.error(f"Failed to add expert issue: {e}", exc_info=True)
            messagebox.showerror(
                "Error",
                f"Failed to add issue: {str(e)}",
                parent=self
            )
