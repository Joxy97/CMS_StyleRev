"""
Settings Dialog for StyleRev CMS configuration.

Provides a comprehensive UI for editing application and project-specific settings.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from typing import Optional, Callable
import logging

from models import (
    ApplicationSettings,
    ModelSettings,
    APIKeys,
    ProcessingSettings,
    LLM_MODEL_OPTIONS,
    RAG_MODEL_OPTIONS,
    DEVICE_OPTIONS,
    LLM_PROVIDERS,
    CLOUD_MODELS,
    OLLAMA_MODELS,
    OLLAMA_MODEL_LIST,
    get_ollama_model_size
)
import subprocess
import webbrowser
import threading
import os

logger = logging.getLogger(__name__)


class SettingsDialog(tk.Toplevel):
    """
    Settings configuration dialog.

    Provides a tabbed interface for configuring:
    - LLM Provider settings
    - RAG Model settings
    - Help
    """

    def __init__(self, parent, current_settings: ApplicationSettings,
                 on_save: Optional[Callable[[ApplicationSettings], None]] = None,
                 on_save_as_default: Optional[Callable[[ApplicationSettings], None]] = None):
        """
        Initialize the settings dialog.

        Args:
            parent: Parent window
            current_settings: Current settings to edit
            on_save: Callback when settings are saved to project
            on_save_as_default: Callback when settings are saved as default
        """
        super().__init__(parent)

        self.title("Project Settings")
        self.geometry("800x600")
        self.resizable(True, True)

        # Make dialog modal
        self.transient(parent)
        self.grab_set()

        # Settings data
        self.current_settings = current_settings
        self.on_save = on_save
        self.on_save_as_default = on_save_as_default

        # Initialize API key variables early (needed before _create_widgets)
        self.anthropic_key_var = tk.StringVar()
        self.openai_key_var = tk.StringVar()
        self.google_key_var = tk.StringVar()

        # Store installed models cache
        self._installed_ollama_models = set()
        self._pending_model_install = None
        self._pull_success = None

        # Create UI
        self._create_widgets()
        self._load_settings()

        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (self.winfo_width() // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")

    def _create_widgets(self):
        """Create the dialog widgets."""
        # Main container
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Create tabs
        self._create_llm_provider_tab()
        self._create_rag_model_tab()
        self._create_help_tab()

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)

        # Left side buttons
        ttk.Button(
            button_frame,
            text="Reset to Defaults",
            command=self._reset_to_defaults
        ).pack(side=tk.LEFT)

        # Right side buttons
        ttk.Button(
            button_frame,
            text="Cancel",
            command=self.destroy
        ).pack(side=tk.RIGHT)

        ttk.Button(
            button_frame,
            text="Save",
            command=self._save_settings
        ).pack(side=tk.RIGHT, padx=5)

        ttk.Button(
            button_frame,
            text="Set as Default",
            command=self._save_as_default
        ).pack(side=tk.RIGHT, padx=5)

    def _create_llm_provider_tab(self):
        """Create the LLM Provider settings tab."""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="LLM Provider")

        # Scrollable frame
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Provider Selection
        ttk.Label(scrollable_frame, text="Provider:", font=('', 10, 'bold')).grid(
            row=0, column=0, sticky=tk.W, pady=(0, 5)
        )

        provider_frame = ttk.Frame(scrollable_frame)
        provider_frame.grid(row=1, column=0, sticky=tk.EW, pady=(0, 10))

        self.provider_var = tk.StringVar()
        provider_combo = ttk.Combobox(
            provider_frame,
            textvariable=self.provider_var,
            values=LLM_PROVIDERS,
            width=20,
            state='readonly'
        )
        provider_combo.grid(row=0, column=0, sticky=tk.W)
        provider_combo.bind('<<ComboboxSelected>>', self._on_provider_changed)

        # Model Selection (next to provider)
        ttk.Label(provider_frame, text="Model:").grid(row=0, column=1, sticky=tk.W, padx=(20, 5))
        self.llm_model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            provider_frame,
            textvariable=self.llm_model_var,
            width=35
        )
        self.model_combo.grid(row=0, column=2, sticky=tk.W)
        self.model_combo.bind('<<ComboboxSelected>>', self._on_model_selected)

        # Dynamic content frame (for API key or Ollama info + Device)
        self.dynamic_frame = ttk.Frame(scrollable_frame)
        self.dynamic_frame.grid(row=2, column=0, sticky=tk.EW, pady=(0, 15))

        # Model Parameters Section
        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).grid(
            row=3, column=0, sticky=tk.EW, pady=10
        )
        ttk.Label(scrollable_frame, text="Model Parameters", font=('', 11, 'bold')).grid(
            row=4, column=0, sticky=tk.W, pady=(0, 10)
        )

        # Max Tokens
        params_frame = ttk.Frame(scrollable_frame)
        params_frame.grid(row=5, column=0, sticky=tk.EW, pady=(0, 15))

        ttk.Label(params_frame, text="Max Tokens:").grid(row=0, column=0, sticky=tk.W)
        self.max_tokens_var = tk.IntVar()
        ttk.Spinbox(
            params_frame,
            from_=500,
            to=10000,
            increment=100,
            textvariable=self.max_tokens_var,
            width=10
        ).grid(row=0, column=1, padx=(10, 0))

        # Temperature
        ttk.Label(params_frame, text="Temperature:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.temperature_var = tk.DoubleVar()
        ttk.Spinbox(
            params_frame,
            from_=0.0,
            to=3.0,
            increment=0.1,
            textvariable=self.temperature_var,
            width=10
        ).grid(row=1, column=1, padx=(10, 0), pady=(5, 0))

        # Timeout
        ttk.Label(params_frame, text="Timeout (seconds):").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self.timeout_var = tk.IntVar()
        ttk.Spinbox(
            params_frame,
            from_=30,
            to=600,
            increment=30,
            textvariable=self.timeout_var,
            width=10
        ).grid(row=2, column=1, padx=(10, 0), pady=(5, 0))

        scrollable_frame.columnconfigure(0, weight=1)

    def _on_provider_changed(self, event=None):
        """Handle provider selection change."""
        provider = self.provider_var.get()

        # Clear dynamic frame
        for widget in self.dynamic_frame.winfo_children():
            widget.destroy()

        if provider == "Local (Ollama)":
            self._setup_ollama_ui()
        else:
            self._pending_model_install = None
            self._setup_cloud_api_ui(provider)

    def _setup_cloud_api_ui(self, provider):
        """Setup UI for cloud API providers."""
        # Update model list
        if provider in CLOUD_MODELS:
            self.model_combo['values'] = CLOUD_MODELS[provider]
            if CLOUD_MODELS[provider]:
                self.model_combo.set(CLOUD_MODELS[provider][0])

        # Show API key field
        api_frame = ttk.Frame(self.dynamic_frame)
        api_frame.pack(fill=tk.X, pady=(5, 0))

        if provider == "Anthropic":
            ttk.Label(api_frame, text="Anthropic API Key:").pack(side=tk.LEFT)
            self.current_api_entry = ttk.Entry(api_frame, textvariable=self.anthropic_key_var, show='*', width=40)
        elif provider == "OpenAI":
            ttk.Label(api_frame, text="OpenAI API Key:").pack(side=tk.LEFT)
            self.current_api_entry = ttk.Entry(api_frame, textvariable=self.openai_key_var, show='*', width=40)
        elif provider == "Google":
            ttk.Label(api_frame, text="Google API Key:").pack(side=tk.LEFT)
            self.current_api_entry = ttk.Entry(api_frame, textvariable=self.google_key_var, show='*', width=40)

        self.current_api_entry.pack(side=tk.LEFT, padx=(10, 0))

    def _setup_ollama_ui(self):
        """Setup UI for Ollama local models."""
        # Check if Ollama is installed
        ollama_installed = self._check_ollama_installed()

        info_frame = ttk.Frame(self.dynamic_frame)
        info_frame.pack(fill=tk.X, pady=(5, 0))

        if ollama_installed:
            # Show green status message
            ttk.Label(
                info_frame,
                text="Ollama is available on this device.",
                foreground='green'
            ).pack(anchor=tk.W)

            # Get installed models
            self._refresh_installed_models()

            # Device selection for local models
            device_frame = ttk.Frame(info_frame)
            device_frame.pack(fill=tk.X, pady=(10, 0))

            ttk.Label(device_frame, text="Device:").pack(side=tk.LEFT)
            self.device_var = tk.StringVar()
            ttk.Combobox(
                device_frame,
                textvariable=self.device_var,
                values=DEVICE_OPTIONS,
                width=15
            ).pack(side=tk.LEFT, padx=(10, 0))

            self.use_gpu_var = tk.BooleanVar()
            ttk.Checkbutton(
                device_frame,
                text="Use GPU if Available",
                variable=self.use_gpu_var
            ).pack(side=tk.LEFT, padx=(20, 0))
        else:
            # Show download warning
            ttk.Label(
                info_frame,
                text="Ollama is required to run local models.",
                foreground='orange'
            ).pack(anchor=tk.W)

            link_label = ttk.Label(
                info_frame,
                text="Download Ollama from https://ollama.ai",
                foreground='blue',
                cursor='hand2'
            )
            link_label.pack(anchor=tk.W, pady=(5, 0))
            link_label.bind('<Button-1>', lambda e: webbrowser.open('https://ollama.ai'))

            ttk.Label(
                info_frame,
                text="After installing, restart this dialog to see available models.",
                foreground='gray'
            ).pack(anchor=tk.W, pady=(5, 0))

            # Initialize device variables even when Ollama not installed
            self.device_var = tk.StringVar(value="auto")
            self.use_gpu_var = tk.BooleanVar(value=True)

        # Build model list with (installed) markers
        model_list = []
        for family, models in OLLAMA_MODELS.items():
            for model_name, size in models:
                if model_name in self._installed_ollama_models:
                    model_list.append(f"{model_name} (installed)")
                else:
                    model_list.append(model_name)

        self.model_combo['values'] = model_list
        if model_list:
            # Try to select first installed model, otherwise first model
            installed = [m for m in model_list if '(installed)' in m]
            if installed:
                self.model_combo.set(installed[0])
            else:
                self.model_combo.set(model_list[0])

    def _check_ollama_installed(self) -> bool:
        """Check if Ollama is installed and running."""
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _refresh_installed_models(self):
        """Refresh the list of installed Ollama models."""
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            if result.returncode == 0:
                self._installed_ollama_models = set()
                for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                    if line.strip():
                        model_name = line.split()[0]
                        self._installed_ollama_models.add(model_name)
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    def _on_model_selected(self, event=None):
        """Handle model selection, prompt for Ollama model installation if needed."""
        provider = self.provider_var.get()
        if provider != "Local (Ollama)":
            self._pending_model_install = None
            return

        selected = self.llm_model_var.get()
        model_name = selected.replace(' (installed)', '').strip()

        if model_name in self._installed_ollama_models:
            self._pending_model_install = None
            return

        self._pending_model_install = model_name
        messagebox.showinfo(
            "Model Not Installed",
            f"Model '{model_name}' is not installed yet. It will be downloaded when you click Save.",
            parent=self
        )

    def _ensure_ollama_model_available(self, model_name: str) -> bool:
        """Ensure the selected Ollama model is installed before saving settings."""
        clean_name = model_name.replace(' (installed)', '').replace('ollama/', '').strip()
        if not clean_name:
            return False

        if clean_name in self._installed_ollama_models:
            return True

        size = get_ollama_model_size(clean_name)
        if os.name == 'nt':
            ollama_dir = os.path.expanduser('~\\.ollama\\models')
        else:
            ollama_dir = os.path.expanduser('~/.ollama/models')

        msg = (
            f"Model '{clean_name}' is not installed.\n\n"
            f"Size: {size:.1f} GB\n"
            f"Installation location: {ollama_dir}\n\n"
            f"Do you want to download and install this model now?"
        )

        if not messagebox.askyesno("Install Model", msg, parent=self):
            return False

        success = self._pull_ollama_model(clean_name)
        if success:
            self._pending_model_install = None
            return True

        messagebox.showerror("Installation Failed", f"Model '{clean_name}' could not be installed.", parent=self)
        return False

    def _pull_ollama_model(self, model_name: str) -> bool:
        """Pull an Ollama model in the background with log display."""
        # Create progress dialog
        progress_window = tk.Toplevel(self)
        progress_window.title("Installing Model")
        progress_window.geometry("500x350")
        progress_window.transient(self)
        progress_window.grab_set()

        # Store cancel flag
        self._pull_cancelled = False
        self._pull_process = None
        self._pull_success = None

        ttk.Label(
            progress_window,
            text=f"Downloading {model_name}...",
            font=('', 10, 'bold')
        ).pack(pady=(10, 5))

        # Log display
        log_frame = ttk.Frame(progress_window)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        log_text = scrolledtext.ScrolledText(
            log_frame,
            height=12,
            width=55,
            font=('Consolas', 9),
            state='disabled'
        )
        log_text.pack(fill=tk.BOTH, expand=True)

        # Progress bar
        progress_bar = ttk.Progressbar(progress_window, mode='indeterminate', length=400)
        progress_bar.pack(pady=5)
        progress_bar.start(10)

        # Cancel button
        cancel_btn = ttk.Button(
            progress_window,
            text="Cancel",
            command=lambda: self._cancel_pull(progress_window)
        )
        cancel_btn.pack(pady=(5, 10))

        def append_log(text):
            """Append text to log display."""
            log_text.config(state='normal')
            log_text.insert(tk.END, text)
            log_text.see(tk.END)
            log_text.config(state='disabled')

        def pull_model():
            try:
                # Use Popen for streaming output
                self._pull_process = subprocess.Popen(
                    ['ollama', 'pull', model_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=1,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )

                buffer = ""

                while True:
                    chunk = self._pull_process.stdout.read(1)

                    if not chunk:
                        if self._pull_process.poll() is not None:
                            if buffer:
                                line = buffer.replace('\r', '')
                                display_line = line if line.endswith('\n') else f"{line}\n"
                                self.after(0, lambda text=display_line: append_log(text))
                            break
                        continue

                    if self._pull_cancelled:
                        self._pull_process.terminate()
                        self.after(0, lambda: self._on_pull_complete(
                            progress_window, model_name, False, "Download cancelled by user"
                        ))
                        return

                    if chunk in ('\n', '\r'):
                        line = buffer
                        buffer = ""

                        if not line and chunk == '\n':
                            continue

                        display_line = line if line.endswith('\n') else f"{line}\n"
                        self.after(0, lambda text=display_line: append_log(text))
                        continue

                    buffer += chunk

                self._pull_process.wait()
                success = self._pull_process.returncode == 0

                self.after(0, lambda: self._on_pull_complete(
                    progress_window, model_name, success,
                    "" if success else "Installation failed. Check log for details."
                ))
            except Exception as e:
                self.after(0, lambda: self._on_pull_complete(
                    progress_window, model_name, False, str(e)
                ))

        thread = threading.Thread(target=pull_model, daemon=True)
        thread.start()

        # Wait until the progress window is closed (pull completes or is cancelled)
        self.wait_window(progress_window)
        return bool(self._pull_success)

    def _cancel_pull(self, progress_window):
        """Cancel the ongoing model pull."""
        self._pull_cancelled = True
        if self._pull_process:
            try:
                self._pull_process.terminate()
            except:
                pass

    def _on_pull_complete(self, progress_window, model_name, success, error_msg):
        """Handle completion of model pull."""
        progress_window.destroy()

        if success:
            self._installed_ollama_models.add(model_name)
            messagebox.showinfo("Success", f"Model '{model_name}' installed successfully!")
            # Refresh model list while keeping the current selection if possible
            current_provider = self.provider_var.get()
            current_selection = self.llm_model_var.get()
            self._setup_ollama_ui()
            if current_provider == "Local (Ollama)":
                installed_label = f"{model_name} (installed)"
                if installed_label in self.model_combo['values']:
                    self.llm_model_var.set(installed_label)
                elif current_selection in self.model_combo['values']:
                    self.llm_model_var.set(current_selection)
        else:
            messagebox.showerror("Error", f"Failed to install model: {error_msg}")

        self._pull_success = success

    def _get_provider_from_model(self, model_name: str) -> str:
        """Determine provider from model name."""
        # Strip ollama/ prefix for checking
        clean_name = model_name.replace('ollama/', '').replace(' (installed)', '').strip()

        if model_name.startswith('claude'):
            return "Anthropic"
        elif model_name.startswith('gpt') or model_name.startswith('o1') or model_name.startswith('o3') or model_name.startswith('o4'):
            return "OpenAI"
        elif model_name.startswith('gemini'):
            return "Google"
        elif model_name.startswith('ollama/') or clean_name in OLLAMA_MODEL_LIST or any(clean_name in [m for m, _ in family] for family in OLLAMA_MODELS.values()):
            return "Local (Ollama)"
        return "Anthropic"  # Default

    def _create_rag_model_tab(self):
        """Create the RAG Model settings tab."""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="RAG Model")

        # RAG Model Selection
        ttk.Label(tab, text="RAG Embedding Model:", font=('', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        self.rag_model_var = tk.StringVar()
        rag_combo = ttk.Combobox(
            tab,
            textvariable=self.rag_model_var,
            values=RAG_MODEL_OPTIONS,
            width=50
        )
        rag_combo.pack(fill=tk.X, pady=(0, 15))

        # Top K Rules
        top_k_frame = ttk.Frame(tab)
        top_k_frame.pack(fill=tk.X, pady=(0, 15))
        ttk.Label(top_k_frame, text="Top K Rules:").pack(side=tk.LEFT)
        self.top_k_var = tk.IntVar()
        ttk.Spinbox(
            top_k_frame,
            from_=1,
            to=50,
            textvariable=self.top_k_var,
            width=10
        ).pack(side=tk.LEFT, padx=(10, 0))

        # Processing Options Section
        ttk.Separator(tab, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=20)
        ttk.Label(tab, text="Processing Options", font=('', 11, 'bold')).pack(anchor=tk.W, pady=(0, 10))

        self.rebuild_db_var = tk.BooleanVar()
        ttk.Checkbutton(
            tab,
            text="Rebuild Rules Database on Startup",
            variable=self.rebuild_db_var
        ).pack(anchor=tk.W, pady=2)

        self.parallel_processing_var = tk.BooleanVar()
        ttk.Checkbutton(
            tab,
            text="Enable Parallel Processing (Experimental)",
            variable=self.parallel_processing_var
        ).pack(anchor=tk.W, pady=2)

        # Concurrent corrections
        concurrent_frame = ttk.Frame(tab)
        concurrent_frame.pack(anchor=tk.W, pady=(10, 0))
        ttk.Label(concurrent_frame, text="Max Concurrent Corrections:").pack(side=tk.LEFT)
        self.max_concurrent_var = tk.IntVar()
        ttk.Spinbox(
            concurrent_frame,
            from_=1,
            to=10,
            textvariable=self.max_concurrent_var,
            width=10
        ).pack(side=tk.LEFT, padx=(10, 0))

        # Performance Options
        ttk.Separator(tab, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=20)
        ttk.Label(tab, text="Performance Options", font=('', 11, 'bold')).pack(anchor=tk.W, pady=(0, 10))

        self.cache_embeddings_var = tk.BooleanVar()
        ttk.Checkbutton(
            tab,
            text="Cache Embeddings",
            variable=self.cache_embeddings_var
        ).pack(anchor=tk.W, pady=2)

        # Batch size
        batch_frame = ttk.Frame(tab)
        batch_frame.pack(anchor=tk.W, pady=(10, 0))
        ttk.Label(batch_frame, text="Embedding Batch Size:").pack(side=tk.LEFT)
        self.batch_size_var = tk.IntVar()
        ttk.Spinbox(
            batch_frame,
            from_=8,
            to=128,
            increment=8,
            textvariable=self.batch_size_var,
            width=10
        ).pack(side=tk.LEFT, padx=(10, 0))

        # Initialize variables that are no longer in UI but needed for settings
        self.enable_local_var = tk.BooleanVar(value=True)
        self.generate_pdf_var = tk.BooleanVar(value=True)
        self.generate_corrected_var = tk.BooleanVar(value=False)
        self.auto_save_var = tk.BooleanVar(value=True)

    def _create_help_tab(self):
        """Create the help tab."""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Help")

        help_text = """Settings Configuration Help

This dialog allows you to configure StyleRev CMS settings.

=== LLM PROVIDER TAB ===

Provider: Select your AI provider
  - Anthropic: Claude models (claude-sonnet, claude-haiku, claude-opus)
  - OpenAI: GPT models (gpt-5, gpt-4.1)
  - Google: Gemini models (gemini-3-pro, gemini-2.5)
  - Local (Ollama): Run models locally on your machine

Model: Select the specific model to use

API Key: Required for cloud providers
  - Leave blank to use environment variables
  - Keys are stored securely with the project

Device (Local only): Select compute device
  - auto: Automatically detect best device
  - cpu: Force CPU usage
  - cuda: Use NVIDIA GPU
  - mps: Use Apple Silicon GPU

Use GPU if Available: Enable GPU acceleration for local models

Model Parameters:
  - Max Tokens: Maximum response length (500-10000)
  - Temperature: Creativity level (0.0=deterministic, 3.0=creative)
  - Timeout: Request timeout in seconds (30-600)

=== RAG MODEL TAB ===

RAG Embedding Model: Model for semantic search
  - BAAI/bge-base-en-v1.5 (recommended)
  - sentence-transformers models

Top K Rules: Number of style rules to consider (1-50)

Processing Options:
  - Rebuild Rules Database: Regenerate on startup
  - Parallel Processing: Process multiple items concurrently
  - Max Concurrent Corrections: Simultaneous correction limit

Performance Options:
  - Cache Embeddings: Cache for faster repeated lookups
  - Embedding Batch Size: Items per embedding batch

=== SETTINGS SCOPE ===

  - Save: Save to current project only
  - Set as Default: Apply to all new projects
  - Reset to Defaults: Restore factory settings

=== LOCAL MODELS (OLLAMA) ===

Available model families:
  - Meta Llama (llama3.1, llama3.2)
  - Alibaba Qwen (qwen2.5)
  - DeepSeek Reasoning (deepseek-r1)
  - Google Gemma (gemma2)
  - Microsoft Phi (phi3)
  - Mistral AI (mixtral, mistral)

Models marked with (installed) are ready to use.
Selecting an uninstalled model will prompt for download.

Download Ollama from: https://ollama.ai
"""

        text_widget = scrolledtext.ScrolledText(
            tab,
            wrap=tk.WORD,
            width=70,
            height=25,
            font=('Consolas', 9)
        )
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert('1.0', help_text)
        text_widget.config(state='disabled')

    def _load_settings(self):
        """Load current settings into the form."""
        # Model settings
        model_name = self.current_settings.model_settings.llm_model

        # Determine and set provider based on model
        provider = self._get_provider_from_model(model_name)
        self.provider_var.set(provider)

        # Trigger provider change to setup UI
        self._on_provider_changed()

        # For Ollama models, strip the ollama/ prefix for display
        if provider == "Local (Ollama)" and model_name.startswith('ollama/'):
            display_name = model_name.replace('ollama/', '')
            # Check if installed and add marker
            if display_name in self._installed_ollama_models:
                display_name = f"{display_name} (installed)"
            self.llm_model_var.set(display_name)
        else:
            self.llm_model_var.set(model_name)

        self.rag_model_var.set(self.current_settings.model_settings.rag_model)
        self.top_k_var.set(self.current_settings.model_settings.top_k_rules)
        self.max_tokens_var.set(self.current_settings.model_settings.max_tokens)
        self.temperature_var.set(self.current_settings.model_settings.temperature)
        self.timeout_var.set(self.current_settings.model_settings.timeout)

        # Device settings (may be set by _setup_ollama_ui or need default)
        if hasattr(self, 'device_var'):
            self.device_var.set(self.current_settings.model_settings.huggingface_device)
        if hasattr(self, 'use_gpu_var'):
            self.use_gpu_var.set(self.current_settings.model_settings.use_gpu_if_available)

        self.enable_local_var.set(self.current_settings.model_settings.enable_local_models)

        # API keys (only if set)
        if self.current_settings.api_keys.anthropic_api_key:
            self.anthropic_key_var.set(self.current_settings.api_keys.anthropic_api_key)
        if self.current_settings.api_keys.openai_api_key:
            self.openai_key_var.set(self.current_settings.api_keys.openai_api_key)
        if self.current_settings.api_keys.google_api_key:
            self.google_key_var.set(self.current_settings.api_keys.google_api_key)

        # Processing settings
        self.rebuild_db_var.set(self.current_settings.processing_settings.rebuild_db)
        self.parallel_processing_var.set(self.current_settings.processing_settings.parallel_processing)
        self.max_concurrent_var.set(self.current_settings.processing_settings.max_concurrent_corrections)
        self.cache_embeddings_var.set(self.current_settings.processing_settings.cache_embeddings)
        self.batch_size_var.set(self.current_settings.processing_settings.embedding_batch_size)
        self.generate_pdf_var.set(self.current_settings.processing_settings.generate_pdf)
        self.generate_corrected_var.set(self.current_settings.processing_settings.generate_corrected)
        self.auto_save_var.set(self.current_settings.processing_settings.auto_save)

    def _save_settings(self):
        """Save settings and close dialog."""
        try:
            # Clean model name (remove ' (installed)' suffix if present)
            model_name = self.llm_model_var.get().replace(' (installed)', '').strip()

            # For Ollama models, ensure installation before adding prefix
            provider = self.provider_var.get()
            if provider == "Local (Ollama)":
                if not self._ensure_ollama_model_available(model_name):
                    return
                if not model_name.startswith('ollama/'):
                    model_name = f"ollama/{model_name}"

            # Get device settings
            device = self.device_var.get() if hasattr(self, 'device_var') else "auto"
            use_gpu = self.use_gpu_var.get() if hasattr(self, 'use_gpu_var') else True

            # Create new settings object
            new_settings = ApplicationSettings(
                model_settings=ModelSettings(
                    llm_model=model_name,
                    rag_model=self.rag_model_var.get(),
                    top_k_rules=self.top_k_var.get(),
                    max_tokens=self.max_tokens_var.get(),
                    temperature=self.temperature_var.get(),
                    timeout=self.timeout_var.get(),
                    huggingface_device=device,
                    enable_local_models=self.enable_local_var.get(),
                    use_gpu_if_available=use_gpu
                ),
                api_keys=APIKeys(
                    anthropic_api_key=self.anthropic_key_var.get() or None,
                    openai_api_key=self.openai_key_var.get() or None,
                    google_api_key=self.google_key_var.get() or None
                ),
                processing_settings=ProcessingSettings(
                    rebuild_db=self.rebuild_db_var.get(),
                    parallel_processing=self.parallel_processing_var.get(),
                    max_concurrent_corrections=self.max_concurrent_var.get(),
                    cache_embeddings=self.cache_embeddings_var.get(),
                    embedding_batch_size=self.batch_size_var.get(),
                    generate_pdf=self.generate_pdf_var.get(),
                    generate_corrected=self.generate_corrected_var.get(),
                    auto_save=self.auto_save_var.get()
                ),
                is_project_specific=True
            )

            # Call the save callback
            if self.on_save:
                self.on_save(new_settings)

            logger.info("Settings saved to project successfully")
            self.destroy()

        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")

    def _save_as_default(self):
        """Save settings as default for new projects."""
        try:
            # Clean model name (remove ' (installed)' suffix if present)
            model_name = self.llm_model_var.get().replace(' (installed)', '').strip()

            # For Ollama models, ensure installation before adding prefix
            provider = self.provider_var.get()
            if provider == "Local (Ollama)":
                if not self._ensure_ollama_model_available(model_name):
                    return
                if not model_name.startswith('ollama/'):
                    model_name = f"ollama/{model_name}"

            # Get device settings
            device = self.device_var.get() if hasattr(self, 'device_var') else "auto"
            use_gpu = self.use_gpu_var.get() if hasattr(self, 'use_gpu_var') else True

            # Create new settings object
            new_settings = ApplicationSettings(
                model_settings=ModelSettings(
                    llm_model=model_name,
                    rag_model=self.rag_model_var.get(),
                    top_k_rules=self.top_k_var.get(),
                    max_tokens=self.max_tokens_var.get(),
                    temperature=self.temperature_var.get(),
                    timeout=self.timeout_var.get(),
                    huggingface_device=device,
                    enable_local_models=self.enable_local_var.get(),
                    use_gpu_if_available=use_gpu
                ),
                api_keys=APIKeys(
                    anthropic_api_key=self.anthropic_key_var.get() or None,
                    openai_api_key=self.openai_key_var.get() or None,
                    google_api_key=self.google_key_var.get() or None
                ),
                processing_settings=ProcessingSettings(
                    rebuild_db=self.rebuild_db_var.get(),
                    parallel_processing=self.parallel_processing_var.get(),
                    max_concurrent_corrections=self.max_concurrent_var.get(),
                    cache_embeddings=self.cache_embeddings_var.get(),
                    embedding_batch_size=self.batch_size_var.get(),
                    generate_pdf=self.generate_pdf_var.get(),
                    generate_corrected=self.generate_corrected_var.get(),
                    auto_save=self.auto_save_var.get()
                ),
                is_project_specific=False
            )

            # Call the save as default callback
            if self.on_save_as_default:
                self.on_save_as_default(new_settings)

            logger.info("Settings saved as default successfully")
            messagebox.showinfo("Success", "Settings saved as default for new projects")

        except Exception as e:
            logger.error(f"Failed to save default settings: {e}")
            messagebox.showerror("Error", f"Failed to save default settings: {str(e)}")

    def _reset_to_defaults(self):
        """Reset all settings to defaults."""
        if messagebox.askyesno("Reset to Defaults", "Are you sure you want to reset all settings to defaults?"):
            default_settings = ApplicationSettings()
            self.current_settings = default_settings
            self._load_settings()
            logger.info("Settings reset to defaults")
