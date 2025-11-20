# StyleRev CMS

An AI-powered expert annotation system for reviewing and correcting style issues in CMS (Compact Muon Solenoid) physics papers according to CMS publication guidelines.

## Features

- **Expert Annotation System**: Interactive GUI for domain experts to review, validate, and modify AI suggestions
- **AI-Powered Suggestions**: Automated detection of CMS style guideline violations
- **Project Management**: Save, load, and export annotated projects
- **Multi-LLM Support**: Anthropic Claude, OpenAI GPT, Google Gemini, and local HuggingFace models
- **RAG-Based Rule Retrieval**: Semantic search through CMS publication guidelines
- **Physics-Aware LaTeX Parsing**: Preserves mathematical notation and physics-specific environments

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd StyleRev_CMS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys in [config.py](config.py) or as environment variables:
```bash
# For Anthropic Claude
export ANTHROPIC_API_KEY="your-key-here"

# For OpenAI GPT
export OPENAI_API_KEY="your-key-here"

# For Google Gemini
export GEMINI_API_KEY="your-key-here"

# For HuggingFace (optional)
export HUGGINGFACE_TOKEN="your-token-here"
```

## Usage

### Running the Application

```bash
python run_expert_system.py
```

Or directly:
```bash
python main.py
```

### Workflow

1. **Create/Load Project**: Load a LaTeX file or existing project
2. **AI Processing**: Process paragraphs with AI to generate style suggestions
3. **Expert Review**: Accept, reject, or modify AI suggestions
4. **Add Missed Issues**: Manually add issues that AI didn't catch
5. **Export**: Export the final corrected LaTeX document

## Project Structure

```
StyleRev_CMS/
├── models/                    # Domain models (DDD architecture)
│   ├── annotation_models.py   # Core entities & value objects
│   └── enums.py              # Type enumerations
├── views/                     # GUI components (Tkinter)
│   ├── project_selector.py   # Project creation/loading
│   ├── paragraph_list.py     # Paragraph list view
│   └── annotation_panel.py   # Main annotation interface
├── services/                  # Business logic layer
│   ├── project_manager.py    # Project lifecycle, file I/O
│   └── annotation_service.py # AI processing, annotations
├── corrector_services/        # AI correction engine
│   └── single_stage_corrector.py
├── llm_support/              # Multi-provider LLM interface
│   └── llm_providers.py
├── parsers/                  # Document parsing
│   ├── physics_latex_parser.py
│   ├── latex_parser.py
│   └── pdf_parser.py
├── resources/               # Shared static assets & rulebook
│   ├── CMS_Icon.png/.ico    # Application icons
│   └── guidelines.json      # CMS style rulebook
├── main.py                   # Main GUI application
├── run_expert_system.py      # Application launcher
└── config.py                 # Configuration
```

## Configuration

### Settings Dialog (Recommended)

Access settings through the GUI via **Edit > Global Settings** or **Edit > Project Settings**:

- **Global Settings**: Configure default settings for all projects
  - LLM and RAG model selection
  - API keys
  - Processing parameters
  - Output options

- **Project Settings**: Override global settings for specific projects
  - Model selection per project
  - Custom processing parameters
  - API keys are inherited from global settings for security

### Manual Configuration

Alternatively, edit [config.py](config.py) directly to set defaults:
- LLM provider and model selection
- RAG settings (embedding model, top-k rules)
- API keys (stored in config.py or environment variables)
- Processing parameters (temperature, max tokens)

## Architecture

The project follows **Model-View Architecture** with clear separation of concerns:

- **Models**: Domain entities (Project, Paragraph, Suggestion, ExpertAnnotation)
- **Views**: Tkinter-based GUI components
- **Services**: Business logic for AI processing and project management
- **Infrastructure**: Parsers, RAG, LLM providers

## License

[Add your license here]
