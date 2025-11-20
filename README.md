# StyleRev CMS - Expert Annotation System

A desktop application for expert review and annotation of AI-generated style corrections for CMS (Compact Muon Solenoid) physics papers. The system combines LLM-powered suggestions with RAG (Retrieval-Augmented Generation) to enforce CMS style guidelines on LaTeX documents.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Manual](#usage-manual)
  - [Creating a New Project](#creating-a-new-project)
  - [Working with Paragraphs](#working-with-paragraphs)
  - [AI Processing](#ai-processing)
  - [Expert Annotation](#expert-annotation)
  - [Rulebook Editor](#rulebook-editor)
  - [Exporting Results](#exporting-results)
  - [Settings Configuration](#settings-configuration)
- [Algorithm Description](#algorithm-description)
  - [System Overview](#system-overview)
  - [Document Processing Pipeline](#document-processing-pipeline)
  - [RAG-Based Rule Retrieval](#rag-based-rule-retrieval)
  - [LLM Correction Process](#llm-correction-process)
  - [Response Parsing](#response-parsing)

## Installation

### Requirements

- Python 3.8+
- API key for at least one LLM provider (Anthropic, OpenAI, or Google)
- Download Ollama for using local models (https://ollama.com/download)

### Setup
Method 1:
Download `dist/CMS_StyleRev.exe`

or

Method 2:
```bash
# Clone or download the repository
cd CMS_StyleRev

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Dependencies

- **LLM Providers**: anthropic, openai, google-generativeai
- **Embeddings**: sentence-transformers, torch, transformers
- **PDF Generation**: reportlab
- **Numerical Computing**: numpy

## Quick Start

1. Launch the application: `python main.py` or `CMS_StyleRev.exe`
2. Configure API keys if using cloud models: Edit → Project Settings, or download Ollama for local models
3. Create a project: File → New Project → Select a LaTeX file
4. Process paragraphs: Process All Paragraphs or AI Process Paragraph
5. Review and annotate AI suggestions in the right panel
6. Export corrected document: File → Export Paragraphs to LaTeX

## Usage Manual

### Creating a New Project

1. **File → New Project** (Ctrl+N)
2. Select a LaTeX (.tex) file
3. The system parses the document and extracts paragraphs
4. Save the project as a .cms file for later use

### Opening Existing Projects

- **File → Open Project** (Ctrl+O)
- Select a .cms project file
- Projects contain all paragraphs, suggestions, annotations, and settings

### Working with Paragraphs

The left panel displays a list of all paragraphs extracted from your document:

- **Gray**: Not processed
- **Yellow/Orange**: Has pending suggestions
- **Green**: All suggestions reviewed
- **Red**: Processing error

Click a paragraph to view and annotate it in the right panel.

### AI Processing

#### Process Single Paragraph
- Select a paragraph
- Click "Process with AI" in the annotation panel
- The system retrieves relevant rules and generates style suggestions

#### Process All Paragraphs
- **Tools → Process All Paragraphs**
- Processes every paragraph in the document
- Progress bar shows completion status
- Can be paused and resumed

### Expert Annotation

For each AI-generated suggestion:

1. **Review** the original text, suggested correction, and rule reference
2. **Choose an action**:
   - **Accept**: Apply the suggestion as-is
   - **Reject**: Dismiss the suggestion
   - **Modify**: Accept with custom modifications
3. **Implement**: Mark accepted/modified suggestions as implemented in the text

#### Adding Expert Issues

Experts can add their own corrections that the AI missed:
1. Select text in the original paragraph
2. Provide the corrected version
3. Specify issue type and severity
4. Add optional notes

### Rulebook Editor

**Edit → Rulebook Editor** opens a window to manage CMS style rules:

- **Import**: Load rules from JSON files
- **Add/Edit/Delete**: Manage individual rules
- **Priority**: Set rule importance (critical, high, normal, low)
- **Examples**: Add positive and negative examples for each rule
- **Re-embed**: Regenerate embeddings when changing rules or RAG model

Rules include:
- Title and content description
- Positive examples (correct usage)
- Negative examples (incorrect usage)
- Priority level

### Exporting Results

**File → Export Paragraphs to LaTeX**

Requirements for export:
- All suggestions must be reviewed (accepted, rejected, or modified)
- Accepted/modified suggestions must be marked as implemented

The export creates a LaTeX file with all implemented corrections applied.

### Settings Configuration

**Edit → Project Settings** opens the configuration dialog:

#### Model Settings
- **LLM Model**: Choose the language model (Claude, GPT, Gemini, Ollama)
- **RAG Model**: Embedding model for semantic rule search
- **Top K Rules**: Number of rules to retrieve per paragraph
- **Temperature**: LLM creativity (0.0 = deterministic)
- **Max Tokens**: Maximum response length

#### API Keys
- Anthropic API key (for Claude models)
- OpenAI API key (for GPT models)
- Google API key (for Gemini models)

#### Processing Settings
- Auto-save enabled by default
- Embedding caching for performance

Settings can be saved as project-specific or as defaults for new projects.

### Keyboard Shortcuts

- **Ctrl+N**: New Project
- **Ctrl+O**: Open Project
- **Ctrl+S**: Save Project
- **Ctrl+Shift+S**: Save Project As

## Algorithm Description

### System Overview

StyleRev CMS uses a RAG (Retrieval-Augmented Generation) architecture to enforce CMS style guidelines. The system combines semantic search over style rules with LLM-powered text correction to provide context-aware suggestions.

```
Input Document → LaTeX Parser → Paragraphs
                                    ↓
                           RAG Rule Retrieval
                                    ↓
                         LLM Correction Engine
                                    ↓
                          Structured Suggestions
                                    ↓
                           Expert Annotation
                                    ↓
                         Corrected Document
```

### Document Processing Pipeline

#### 1. LaTeX Parsing

The `PhysicsLatexParser` extracts content from LaTeX documents:

- Preserves physics notation and equations
- Identifies sections and subsections
- Splits content into logical paragraphs
- Maintains document structure metadata (section titles, labels)

#### 2. Paragraph Extraction

Each paragraph becomes a `Paragraph` object containing:
- Original text
- Section context (title, label)
- Paragraph type (text, equation, figure, table)
- Status tracking (not processed, changed, done, error)

### RAG-Based Rule Retrieval

#### Embedding Generation

Style rules are embedded using sentence transformers (default: `BAAI/bge-base-en-v1.5`):

```python
# For each rule, create searchable embedding
rule_embedding = sentence_transformer.encode(
    f"{rule.title} {rule.content} {rule.positive_examples}"
)
```

#### Semantic Search

When processing a paragraph:

1. **Generate query embedding** for the input paragraph
2. **Compute similarity scores** against all rule embeddings using cosine similarity
3. **Select top-K rules** (default K=10) with highest similarity scores
4. **Return ranked rules** with similarity scores for LLM context

```python
def search(paragraph_text, top_k=10):
    query_embedding = encode(paragraph_text)
    similarities = cosine_similarity(query_embedding, rule_embeddings)
    top_indices = argsort(similarities)[-top_k:]
    return [(rules[i], similarities[i]) for i in top_indices]
```

### LLM Correction Process

#### Single-Stage Correction

The `CMSSingleStageCorrector` processes each paragraph in one LLM call:

1. **Format retrieved rules** with titles, content, examples, and priorities
2. **Construct prompt** with rules and paragraph text
3. **Request structured JSON output** with corrections

#### Prompt Structure

```
You are a CMS physics paper style editor. Analyze the paragraph below
against the provided CMS style guidelines...

CMS STYLE GUIDELINES:
Rule 1: [title]
Content: [description]
Positive examples: [correct usage]
Negative examples: [incorrect usage]
Priority: [level]
...

PARAGRAPH TO ANALYZE:
[input paragraph]

TASK: Output corrections in JSON format with:
- corrected_paragraph
- changes_made (boolean)
- edits (array of individual corrections)
```

#### Expected LLM Response

```json
{
  "corrected_paragraph": "The fully corrected paragraph text",
  "changes_made": true,
  "edits": [
    {
      "original": "exact text that was changed",
      "corrected": "exact replacement text",
      "rule_title": "CMS Rule Title",
      "rule_content": "Explanation of why this change was needed",
      "confidence": 0.95
    }
  ]
}
```

### Response Parsing

#### JSON Extraction

The system handles various LLM response formats:

1. **Strip markdown fencing** (```json blocks)
2. **Extract JSON object** from response text
3. **Parse and validate** required fields
4. **Normalize confidence scores** to 0.0-1.0 range

#### Fallback Handling

If JSON parsing fails:
1. Attempt regex extraction of corrected text
2. Return original paragraph with error flag
3. Log parsing failure for debugging

#### Suggestion Creation

Each edit becomes a `Suggestion` object:

```python
suggestion = Suggestion.create(
    original=edit['original'],
    suggested=edit['corrected'],
    rule_title=edit['rule_title'],
    rule_content=edit['rule_content'],
    confidence=edit['confidence']
)
```

### Multi-Provider LLM Support

The `LLMManager` provides unified access to multiple providers:

- **Anthropic**: Claude models (claude-sonnet-4-5-20250929, etc.)
- **OpenAI**: GPT models (gpt-4, gpt-4-turbo, etc.)
- **Google**: Gemini models
- **Ollama**: Local models (no API key required)

Provider selection is automatic based on model name prefix or explicit `provider:model` format.

### Annotation Workflow

#### Expert Review Loop

1. AI generates suggestions for each paragraph
2. Expert reviews each suggestion:
   - **Accept**: Mark suggestion for implementation
   - **Reject**: Dismiss suggestion (not applied)
   - **Modify**: Accept with custom text changes
3. Accepted/modified suggestions are implemented into the paragraph text
4. Export validates all suggestions are resolved

#### Project Persistence

Projects are saved as `.cms` files containing:
- All paragraphs with original and current text
- AI suggestions with rule references
- Expert annotations and decisions
- Project-specific settings
- Custom rulebook (if modified)

### Performance Optimizations

- **Lazy loading**: Corrector initialized on first use
- **Embedding caching**: Rule embeddings stored to avoid recomputation
- **Background threading**: AI processing runs in separate threads
- **Incremental updates**: UI updates as each paragraph completes

## File Formats

### Project Files (.cms)

Binary pickle format containing:
- Project metadata
- Paragraph list with annotations
- Custom rulebook
- Project settings

### Guidelines (JSON)

```json
{
  "rules": [
    {
      "title": "Rule Title",
      "content": "Description of the style rule",
      "positive_examples": "Correct: ...",
      "negative_examples": "Incorrect: ...",
      "priority": "normal"
    }
  ]
}
```

## Version

Version 1.0.0 (Development)

## Architecture

Model-View-Service pattern with:
- **Models**: Data structures (paragraphs, projects, rules, settings)
- **Views**: Tkinter GUI components
- **Services**: Business logic (annotation, project management, export)
