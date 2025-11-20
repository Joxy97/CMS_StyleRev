"""
Rulebook View for the StyleRev CMS system.

This module provides a Tkinter-based view for managing rules in a rulebook,
with editable table, filtering, and import/export capabilities.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from typing import Optional, Callable
import threading
import logging

from models import Rulebook, Rule, RulePriority, settings_manager

logger = logging.getLogger(__name__)


class RulebookView(ttk.Frame):
    """
    View widget for editing a Rulebook.

    Provides a table-based interface for viewing and editing rules
    with filtering and import/export capabilities.
    """

    def __init__(self, parent, rulebook: Optional[Rulebook] = None,
                 on_changed: Optional[Callable] = None):
        """
        Initialize the RulebookView.

        Args:
            parent: Parent widget
            rulebook: Rulebook to display/edit
            on_changed: Callback when rulebook is modified
        """
        super().__init__(parent)

        self.rulebook = rulebook if rulebook else Rulebook()
        self.on_changed = on_changed
        self._selected_rule: Optional[Rule] = None
        self._filter_priority = None
        self._filter_text = ""

        self._setup_ui()
        self._refresh_table()

    def _setup_ui(self):
        """Set up the UI components."""
        # Main layout
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        # Toolbar
        self._setup_toolbar()

        # Filter section
        self._setup_filters()

        # Table
        self._setup_table()

        # Button row
        self._setup_buttons()

    def _setup_toolbar(self):
        """Set up the toolbar with import/export actions."""
        toolbar = ttk.Frame(self)
        toolbar.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # Import New button
        self.import_new_btn = ttk.Button(
            toolbar,
            text="Import New",
            command=self._import_new_rulebook
        )
        self.import_new_btn.pack(side=tk.LEFT, padx=2)

        # Add from File button
        self.add_from_file_btn = ttk.Button(
            toolbar,
            text="Add from File",
            command=self._add_rules_from_file
        )
        self.add_from_file_btn.pack(side=tk.LEFT, padx=2)

        # Export button
        self.export_btn = ttk.Button(
            toolbar,
            text="Export",
            command=self._export_rulebook
        )
        self.export_btn.pack(side=tk.LEFT, padx=2)

        # Separator
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Rule count label
        self.count_label = ttk.Label(toolbar, text="Rules: 0")
        self.count_label.pack(side=tk.LEFT, padx=5)

    def _setup_filters(self):
        """Set up filter controls."""
        filter_frame = ttk.Frame(self)
        filter_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        # Priority filter
        ttk.Label(filter_frame, text="Priority:").pack(side=tk.LEFT, padx=5)
        self.priority_combo = ttk.Combobox(
            filter_frame,
            values=["All"] + [p.name for p in RulePriority],
            state="readonly",
            width=12
        )
        self.priority_combo.set("All")
        self.priority_combo.pack(side=tk.LEFT, padx=5)
        self.priority_combo.bind("<<ComboboxSelected>>", self._on_filter_changed)

        # Search box
        ttk.Label(filter_frame, text="Search:").pack(side=tk.LEFT, padx=5)
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(filter_frame, textvariable=self.search_var, width=30)
        self.search_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.search_var.trace("w", self._on_search_changed)

    def _setup_table(self):
        """Set up the rules table."""
        # Table frame with scrollbars
        table_frame = ttk.Frame(self)
        table_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        # Treeview
        columns = ("id", "title", "priority", "content", "section")
        self.tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            selectmode="browse"
        )

        # Configure columns
        self.tree.heading("id", text="ID", anchor=tk.W)
        self.tree.heading("title", text="Title", anchor=tk.W)
        self.tree.heading("priority", text="Priority", anchor=tk.W)
        self.tree.heading("content", text="Content", anchor=tk.W)
        self.tree.heading("section", text="Section", anchor=tk.W)

        self.tree.column("id", width=50, minwidth=40)
        self.tree.column("title", width=200, minwidth=100)
        self.tree.column("priority", width=100, minwidth=80)
        self.tree.column("content", width=300, minwidth=150)
        self.tree.column("section", width=150, minwidth=100)

        # Scrollbars
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # Grid layout
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        # Bind events
        self.tree.bind("<<TreeviewSelect>>", self._on_selection_changed)
        self.tree.bind("<Double-1>", self._on_double_click)

    def _setup_buttons(self):
        """Set up action buttons."""
        button_frame = ttk.Frame(self)
        button_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)

        # Add button
        self.add_btn = ttk.Button(
            button_frame,
            text="Add New",
            command=self._add_rule
        )
        self.add_btn.pack(side=tk.LEFT, padx=2)

        # Edit button
        self.edit_btn = ttk.Button(
            button_frame,
            text="Edit",
            command=self._edit_rule
        )
        self.edit_btn.pack(side=tk.LEFT, padx=2)

        # Delete button
        self.delete_btn = ttk.Button(
            button_frame,
            text="Delete",
            command=self._delete_rule
        )
        self.delete_btn.pack(side=tk.LEFT, padx=2)

    def set_rulebook(self, rulebook: Rulebook):
        """Set the rulebook to display."""
        self.rulebook = rulebook
        self._refresh_table()

    def _refresh_table(self):
        """Refresh the table with current rulebook data."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Get filtered rules
        filtered_rules = self._get_filtered_rules()

        # Insert rules
        for rule in filtered_rules:
            # Truncate long content for display
            content_display = rule.content[:100] + "..." if len(rule.content) > 100 else rule.content
            content_display = content_display.replace("\n", " ")

            self.tree.insert("", tk.END, iid=str(rule.rule_id), values=(
                rule.rule_id,
                rule.title,
                rule.priority.name if rule.priority else "NORMAL",
                content_display,
                rule.section_path
            ))

        # Update count
        self.count_label.config(text=f"Rules: {len(filtered_rules)}/{self.rulebook.get_rule_count()}")

    def _get_filtered_rules(self):
        """Get rules filtered by current criteria."""
        rules = self.rulebook.rules

        # Filter by priority
        if self._filter_priority and self._filter_priority != "All":
            rules = [r for r in rules if r.priority.name == self._filter_priority]

        # Filter by search text
        if self._filter_text:
            search_lower = self._filter_text.lower()
            rules = [r for r in rules if (
                search_lower in r.title.lower() or
                search_lower in r.content.lower() or
                search_lower in r.section_path.lower() or
                search_lower in r.positive_examples.lower() or
                search_lower in r.negative_examples.lower()
            )]

        return rules

    def _on_filter_changed(self, event=None):
        """Handle filter combo change."""
        selected = self.priority_combo.get()
        self._filter_priority = selected if selected != "All" else None
        self._refresh_table()

    def _on_search_changed(self, *args):
        """Handle search text change."""
        self._filter_text = self.search_var.get()
        self._refresh_table()

    def _on_selection_changed(self, event=None):
        """Handle tree selection change."""
        selection = self.tree.selection()
        if selection:
            rule_id = int(selection[0])
            self._selected_rule = self.rulebook.get_rule_by_id(rule_id)
        else:
            self._selected_rule = None

    def _on_double_click(self, event):
        """Handle double-click to edit."""
        if self._selected_rule:
            self._edit_rule()

    def _add_rule(self):
        """Add a new rule."""
        new_rule = self.rulebook.create_new_rule_with_defaults()
        self.rulebook.add_rule(new_rule)
        self._refresh_table()
        self._notify_changed()

        # Select and edit the new rule
        self.tree.selection_set(str(new_rule.rule_id))
        self._selected_rule = new_rule
        self._edit_rule()

    def _edit_rule(self):
        """Edit the selected rule."""
        if not self._selected_rule:
            messagebox.showwarning("No Selection", "Please select a rule to edit.")
            return

        # Open edit dialog
        dialog = RuleEditDialog(self, self._selected_rule)
        self.wait_window(dialog)

        if dialog.result:
            # Update rule with dialog result
            rule = self._selected_rule
            rule.title = dialog.result['title']
            rule.content = dialog.result['content']
            rule.section_path = dialog.result['section_path']
            rule.priority = dialog.result['priority']
            rule.positive_examples = dialog.result['positive_examples']
            rule.negative_examples = dialog.result['negative_examples']

            # Regenerate embeddings
            rule.regenerate_embeddings()

            self._refresh_table()
            self._notify_changed()

    def _delete_rule(self):
        """Delete the selected rule."""
        if not self._selected_rule:
            messagebox.showwarning("No Selection", "Please select a rule to delete.")
            return

        if messagebox.askyesno("Confirm Delete",
                              f"Are you sure you want to delete rule '{self._selected_rule.title}'?"):
            self.rulebook.remove_rule(self._selected_rule)
            self._selected_rule = None
            self._refresh_table()
            self._notify_changed()

    def _import_new_rulebook(self):
        """Import a new rulebook, replacing existing rules."""
        file_path = filedialog.askopenfilename(
            title="Import New Rulebook",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if not file_path:
            return

        if not messagebox.askyesno("Confirm Import",
                                   "This will REPLACE all existing rules. Continue?"):
            return

        # Show progress dialog
        progress = ProgressDialog(self, "Importing Rulebook")

        def import_task():
            try:
                success = self.rulebook.import_new_from_json(
                    file_path,
                    progress_callback=progress.update_progress
                )

                self.after(0, lambda: self._import_complete(success, progress))

            except Exception as e:
                logger.error(f"Import failed: {e}")
                self.after(0, lambda: self._import_complete(False, progress, str(e)))

        thread = threading.Thread(target=import_task)
        thread.daemon = True
        thread.start()

    def _add_rules_from_file(self):
        """Add rules from a file to existing rulebook."""
        file_path = filedialog.askopenfilename(
            title="Add Rules from File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if not file_path:
            return

        # Show progress dialog
        progress = ProgressDialog(self, "Adding Rules")

        def import_task():
            try:
                success = self.rulebook.import_from_json(
                    file_path,
                    progress_callback=progress.update_progress
                )

                self.after(0, lambda: self._import_complete(success, progress))

            except Exception as e:
                logger.error(f"Import failed: {e}")
                self.after(0, lambda: self._import_complete(False, progress, str(e)))

        thread = threading.Thread(target=import_task)
        thread.daemon = True
        thread.start()

    def _import_complete(self, success: bool, progress: 'ProgressDialog', error: str = None):
        """Handle import completion."""
        progress.destroy()

        if success:
            self._refresh_table()
            self._notify_changed()
            messagebox.showinfo("Import Complete",
                              f"Successfully imported {self.rulebook.get_rule_count()} rules.")
        else:
            messagebox.showerror("Import Failed",
                               f"Failed to import rules: {error or 'Unknown error'}")

    def _export_rulebook(self):
        """Export the rulebook to a JSON file."""
        if self.rulebook.is_empty():
            messagebox.showwarning("No Rules", "There are no rules to export.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Export Rulebook",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if not file_path:
            return

        if self.rulebook.export_to_json(file_path):
            messagebox.showinfo("Export Complete",
                              f"Exported {self.rulebook.get_rule_count()} rules to {file_path}")
        else:
            messagebox.showerror("Export Failed", "Failed to export rules.")

    def _notify_changed(self):
        """Notify that the rulebook has changed."""
        if self.on_changed:
            self.on_changed(self.rulebook)


class RuleEditDialog(tk.Toplevel):
    """Dialog for editing a rule."""

    def __init__(self, parent, rule: Rule):
        super().__init__(parent)
        self.title(f"Edit Rule - {rule.title}")
        self.geometry("600x500")
        self.resizable(True, True)

        self.rule = rule
        self.result = None

        self._setup_ui()

        # Make modal
        self.transient(parent)
        self.grab_set()

        # Center on parent
        self.geometry(f"+{parent.winfo_rootx() + 50}+{parent.winfo_rooty() + 50}")

    def _setup_ui(self):
        """Set up the dialog UI."""
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        ttk.Label(main_frame, text="Title:").grid(row=0, column=0, sticky="w", pady=2)
        self.title_var = tk.StringVar(value=self.rule.title)
        ttk.Entry(main_frame, textvariable=self.title_var, width=50).grid(
            row=0, column=1, sticky="ew", pady=2
        )

        # Priority
        ttk.Label(main_frame, text="Priority:").grid(row=1, column=0, sticky="w", pady=2)
        self.priority_var = tk.StringVar(value=self.rule.priority.name if self.rule.priority else "NORMAL")
        priority_combo = ttk.Combobox(
            main_frame,
            textvariable=self.priority_var,
            values=[p.name for p in RulePriority],
            state="readonly",
            width=15
        )
        priority_combo.grid(row=1, column=1, sticky="w", pady=2)

        # Section Path
        ttk.Label(main_frame, text="Section:").grid(row=2, column=0, sticky="w", pady=2)
        self.section_var = tk.StringVar(value=self.rule.section_path)
        ttk.Entry(main_frame, textvariable=self.section_var, width=50).grid(
            row=2, column=1, sticky="ew", pady=2
        )

        # Content
        ttk.Label(main_frame, text="Content:").grid(row=3, column=0, sticky="nw", pady=2)
        self.content_text = tk.Text(main_frame, height=5, width=50, wrap=tk.WORD)
        self.content_text.grid(row=3, column=1, sticky="ew", pady=2)
        self.content_text.insert("1.0", self.rule.content)

        # Positive Examples
        ttk.Label(main_frame, text="Positive Examples:").grid(row=4, column=0, sticky="nw", pady=2)
        self.positive_text = tk.Text(main_frame, height=3, width=50, wrap=tk.WORD)
        self.positive_text.grid(row=4, column=1, sticky="ew", pady=2)
        self.positive_text.insert("1.0", self.rule.positive_examples)

        # Negative Examples
        ttk.Label(main_frame, text="Negative Examples:").grid(row=5, column=0, sticky="nw", pady=2)
        self.negative_text = tk.Text(main_frame, height=3, width=50, wrap=tk.WORD)
        self.negative_text.grid(row=5, column=1, sticky="ew", pady=2)
        self.negative_text.insert("1.0", self.rule.negative_examples)

        # Configure column weight
        main_frame.columnconfigure(1, weight=1)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=10)

        ttk.Button(button_frame, text="Save", command=self._save).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=5)

    def _save(self):
        """Save changes and close dialog."""
        title = self.title_var.get().strip()
        if not title:
            messagebox.showwarning("Invalid Input", "Title cannot be empty.")
            return

        self.result = {
            'title': title,
            'content': self.content_text.get("1.0", tk.END).strip(),
            'section_path': self.section_var.get().strip(),
            'priority': RulePriority[self.priority_var.get()],
            'positive_examples': self.positive_text.get("1.0", tk.END).strip(),
            'negative_examples': self.negative_text.get("1.0", tk.END).strip()
        }

        self.destroy()


class ProgressDialog(tk.Toplevel):
    """Dialog showing progress of a long-running operation."""

    def __init__(self, parent, title: str):
        super().__init__(parent)
        self.title(title)
        self.geometry("400x120")
        self.resizable(False, False)

        # Make modal
        self.transient(parent)
        self.grab_set()

        # Center on parent
        self.geometry(f"+{parent.winfo_rootx() + 100}+{parent.winfo_rooty() + 100}")

        # UI
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.message_label = ttk.Label(main_frame, text="Starting...")
        self.message_label.pack(pady=5)

        self.progress_bar = ttk.Progressbar(main_frame, length=350, mode='determinate')
        self.progress_bar.pack(pady=10)

        # Prevent closing
        self.protocol("WM_DELETE_WINDOW", lambda: None)

    def update_progress(self, current: int, total: int, message: str):
        """Update the progress display."""
        def update():
            if total > 0:
                self.progress_bar['value'] = (current / total) * 100
            self.message_label.config(text=message)
            self.update_idletasks()

        self.after(0, update)


class RulebookEditorWindow(tk.Toplevel):
    """Standalone window for editing a rulebook."""

    def __init__(self, parent, rulebook: Rulebook, on_save: Optional[Callable] = None,
                 rag_model: Optional[str] = None):
        super().__init__(parent)
        self.title("Rulebook Editor")
        self.geometry("900x600")
        self.minsize(700, 400)

        self.rulebook = rulebook
        self.on_save = on_save
        self.rag_model = rag_model or settings_manager.rag_model

        # Info bar showing RAG model
        info_frame = ttk.Frame(self)
        info_frame.pack(fill=tk.X, padx=5, pady=(5, 0))

        ttk.Label(
            info_frame,
            text=f"RAG Embedding Model: {self.rag_model}",
            foreground='gray'
        ).pack(side=tk.LEFT)

        ttk.Label(
            info_frame,
            text=f"({rulebook.get_rule_count()} rules)",
            foreground='gray'
        ).pack(side=tk.RIGHT)

        # Create view
        self.rulebook_view = RulebookView(
            self,
            rulebook=rulebook,
            on_changed=self._on_rulebook_changed
        )
        self.rulebook_view.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Make modal
        self.transient(parent)
        self.grab_set()

        # Handle close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_rulebook_changed(self, rulebook: Rulebook):
        """Handle rulebook changes."""
        if self.on_save:
            self.on_save(rulebook)

    def _on_close(self):
        """Handle window close."""
        self.destroy()
