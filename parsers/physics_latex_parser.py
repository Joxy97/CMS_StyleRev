"""
Physics-Friendly LaTeX Parser for CMS Style Revision

This parser is specifically designed for physics papers, preserving mathematical expressions,
citations, figures, tables, and other essential scientific content while extracting paragraphs
for style correction.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PhysicsLaTeXParser:
    """Physics-friendly LaTeX parser that preserves mathematical content and scientific formatting."""

    def __init__(self):
        self.current_section = ""
        self.current_section_number = 0
        self.current_subsection = ""
        self.current_subsection_number = 0
        self.paragraph_counter = 0

        # Track processed equations, figures, tables for context
        self.equations = []
        self.figures = []
        self.tables = []
        self.citations = set()

        # Track base directory for resolving relative paths
        self.base_directory = None

        # Physics-specific LaTeX commands that should be PRESERVED
        self.preserve_commands = [
            # Mathematical expressions - keep as-is (order matters!)
            r'\$\$[^$]+\$\$',  # Display math first (longer pattern)
            r'\$[^$]+\$',      # Inline math second
            r'\\begin\{equation\*?\}.*?\\end\{equation\*?\}',
            r'\\begin\{align\*?\}.*?\\end\{align\*?\}',
            r'\\begin\{eqnarray\*?\}.*?\\end\{eqnarray\*?\}',
            r'\\begin\{gather\*?\}.*?\\end\{gather\*?\}',
            r'\\begin\{multline\*?\}.*?\\end\{multline\*?\}',

            # Physics symbols and notation
            r'\\lambda_\{?[^}]*\}?',  # Greek letters with subscripts
            r'\\mu_\{?[^}]*\}?',
            r'\\sigma_\{?[^}]*\}?',
            r'\\alpha_\{?[^}]*\}?',
            r'\\beta_\{?[^}]*\}?',
            r'\\gamma_\{?[^}]*\}?',
            r'\\delta_\{?[^}]*\}?',
            r'\\epsilon_\{?[^}]*\}?',
            r'\\theta_\{?[^}]*\}?',
            r'\\kappa_\{?[^}]*\}?',
            r'\\nu_\{?[^}]*\}?',
            r'\\xi_\{?[^}]*\}?',
            r'\\pi_\{?[^}]*\}?',
            r'\\rho_\{?[^}]*\}?',
            r'\\tau_\{?[^}]*\}?',
            r'\\upsilon_\{?[^}]*\}?',
            r'\\phi_\{?[^}]*\}?',
            r'\\chi_\{?[^}]*\}?',
            r'\\psi_\{?[^}]*\}?',
            r'\\omega_\{?[^}]*\}?',

            # Physics units and notation
            r'\\GeV\\?',
            r'\\TeV\\?',
            r'\\MeV\\?',
            r'\\keV\\?',
            r'\\eV\\?',
            r'\\fb\\?',
            r'\\pb\\?',
            r'\\nb\\?',
            r'\\mb\\?',

            # Citations and references - keep for context
            r'\\cite\{[^}]+\}',
            r'\\citep\{[^}]+\}',
            r'\\citet\{[^}]+\}',
            r'\\ref\{[^}]+\}',
            r'\\eqref\{[^}]+\}',
            r'\\autoref\{[^}]+\}',

            # Figure and table references
            r'Fig\.?~?\\ref\{[^}]+\}',
            r'Figure~?\\ref\{[^}]+\}',
            r'Table~?\\ref\{[^}]+\}',
            r'Eq\.?~?\\eqref\{[^}]+\}',
            r'Equation~?\\eqref\{[^}]+\}',

            # Physics particle notation
            r'\\PH\\?',  # Higgs
            r'\\PW\\?',  # W boson
            r'\\PZ\\?',  # Z boson
            r'\\PQb\\?', # b quark
            r'\\PQc\\?', # c quark
            r'\\PQt\\?', # t quark
            r'\\PQu\\?', # u quark
            r'\\PQd\\?', # d quark
            r'\\PQs\\?', # s quark
            r'\\Pp\\?',  # proton
            r'\\Pn\\?',  # neutron
            r'\\Pe\\?',  # electron
            r'\\Pmu\\?', # muon
            r'\\Ptau\\?', # tau
            r'\\Pnu\\?', # neutrino
        ]

        # Commands to clean but preserve structure
        self.safe_clean_commands = {
            # Text formatting - remove commands but keep content
            r'\\textbf\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}': r'\1',
            r'\\textit\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}': r'\1',
            r'\\textsc\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}': r'\1',
            r'\\emph\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}': r'\1',
            r'\\textrm\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}': r'\1',

            # Remove labels but keep content nearby
            r'\\label\{[^}]*\}': '',

            # Spacing commands
            r'\\xspace': '',
            r'\\\\': '\n',  # Line breaks

            # Special characters that should be preserved literally
            r'\\&': '&',
            r'\\%': '%',
            r'\\\$': '$',
            r'\\#': '#',
            r'\\_': '_',
        }

        # Environments to extract content from but remove environment commands
        self.content_environments = {
            'abstract', 'introduction', 'conclusion', 'discussion',
            'section', 'subsection', 'subsubsection', 'paragraph'
        }

        # Environments to completely remove (but may extract captions)
        self.remove_environments = {
            'figure', 'table', 'algorithm', 'lstlisting', 'verbatim',
            'tikzpicture', 'pgfplots'
        }

        # Math environments to preserve entirely
        self.math_environments = {
            'equation', 'equation*', 'align', 'align*', 'eqnarray', 'eqnarray*',
            'gather', 'gather*', 'multline', 'multline*', 'split',
            'array', 'matrix', 'pmatrix', 'bmatrix', 'vmatrix', 'Vmatrix'
        }

    def preserve_protected_content(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Extract and preserve important content that shouldn't be modified."""
        preserved = {}
        counter = 0

        # Preserve math expressions first
        for pattern in self.preserve_commands:
            matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                placeholder = f"__PRESERVED_{counter}__"
                preserved[placeholder] = match.group(0)
                text = text.replace(match.group(0), placeholder, 1)
                counter += 1

        return text, preserved

    def restore_preserved_content(self, text: str, preserved: Dict[str, str]) -> str:
        """Restore preserved content back to the text."""
        for placeholder, content in preserved.items():
            text = text.replace(placeholder, content)
        return text

    def resolve_includes(self, content: str, base_path: Path) -> str:
        r"""
        Recursively resolve \input{} and \include{} commands.

        Args:
            content: The LaTeX content with possible include/input commands
            base_path: Base directory path for resolving relative file paths

        Returns:
            str: Content with all includes expanded
        """
        # Pattern to match \input{filename} or \include{filename}
        include_pattern = r'\\(input|include)\{([^}]+)\}'

        def replace_include(match):
            command = match.group(1)  # 'input' or 'include'
            filename = match.group(2).strip()

            # Try to find the file with various extensions
            possible_paths = []

            # Add .tex extension if not present
            if not filename.endswith('.tex'):
                possible_paths.append(base_path / f"{filename}.tex")
            possible_paths.append(base_path / filename)

            # Also check in subdirectories relative to base_path
            for ext in ['', '.tex']:
                possible_paths.append(base_path / f"{filename}{ext}")

            # Find the first existing file
            included_file = None
            for path in possible_paths:
                if path.exists():
                    included_file = path
                    break

            if included_file is None:
                logger.warning(f"Could not find included file: {filename} (searched in {base_path})")
                return f"\n% [File not found: {filename}]\n"

            try:
                with open(included_file, 'r', encoding='utf-8', errors='ignore') as f:
                    included_content = f.read()

                # Recursively resolve includes in the included file
                # Use the directory of the included file as base for nested includes
                included_base_path = included_file.parent
                included_content = self.resolve_includes(included_content, included_base_path)

                logger.debug(f"Resolved {command} for: {included_file.name}")

                # For \include, add page breaks (LaTeX does this automatically)
                if command == 'include':
                    return f"\n\\clearpage\n{included_content}\n\\clearpage\n"
                else:
                    return f"\n{included_content}\n"

            except Exception as e:
                logger.error(f"Error reading included file {included_file}: {e}")
                return f"\n% [Error reading file: {filename}]\n"

        # Replace all includes recursively
        prev_content = None
        iterations = 0
        max_iterations = 100  # Prevent infinite loops

        while prev_content != content and iterations < max_iterations:
            prev_content = content
            content = re.sub(include_pattern, replace_include, content)
            iterations += 1

        if iterations >= max_iterations:
            logger.warning("Maximum include resolution depth reached - possible circular includes")

        return content

    def extract_math_expressions(self, text: str) -> List[str]:
        """Extract mathematical expressions for context."""
        math_expressions = []

        # Inline math
        inline_math = re.findall(r'\$([^$]+)\$', text)
        math_expressions.extend(inline_math)

        # Display math
        display_math = re.findall(r'\$\$([^$]+)\$\$', text)
        math_expressions.extend(display_math)

        # Equation environments
        eq_patterns = [
            r'\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}',
            r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}',
            r'\\begin\{eqnarray\*?\}(.*?)\\end\{eqnarray\*?\}'
        ]

        for pattern in eq_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            math_expressions.extend(matches)

        return math_expressions

    def extract_citations(self, text: str) -> Set[str]:
        """Extract citation keys for context."""
        citations = set()

        # Various citation formats
        cite_patterns = [
            r'\\cite\{([^}]+)\}',
            r'\\citep\{([^}]+)\}',
            r'\\citet\{([^}]+)\}',
        ]

        for pattern in cite_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Handle multiple citations: \cite{ref1,ref2,ref3}
                refs = [ref.strip() for ref in match.split(',')]
                citations.update(refs)

        return citations

    def clean_latex_text(self, text: str) -> str:
        """Clean LaTeX commands while preserving physics and math content."""

        # First, preserve important content
        text, preserved = self.preserve_protected_content(text)

        # Remove comments but be careful with math comments
        text = re.sub(r'(?<!\\)%.*$', '', text, flags=re.MULTILINE)

        # Handle environments carefully
        text = self.process_environments(text)

        # Apply safe cleaning commands
        for pattern, replacement in self.safe_clean_commands.items():
            text = re.sub(pattern, replacement, text)

        # Clean up excessive whitespace but preserve paragraph breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple blank lines -> double
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces -> single space
        text = re.sub(r' +\n', '\n', text)  # Trailing spaces

        # Restore preserved content
        text = self.restore_preserved_content(text, preserved)

        return text.strip()

    def process_environments(self, text: str) -> str:
        """Process LaTeX environments appropriately."""

        # Remove figure and table environments but extract captions
        for env in self.remove_environments:
            pattern = rf'\\begin\{{{env}\*?\}}(.*?)\\end\{{{env}\*?\}}'
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                env_content = match.group(1)
                # Extract caption if present
                caption_match = re.search(r'\\caption\{([^}]+)\}', env_content)
                if caption_match:
                    caption = caption_match.group(1)
                    # Keep caption as context
                    replacement = f" [{env.capitalize()}: {caption}] "
                else:
                    replacement = f" [{env.capitalize()}] "
                text = text.replace(match.group(0), replacement, 1)

        return text

    def extract_title_and_abstract(self, content: str) -> Tuple[str, str]:
        """Extract document title and abstract while preserving formatting."""
        title = ""
        abstract = ""

        # Extract title
        title_match = re.search(r'\\title\{([^}]+)\}', content)
        if title_match:
            title = self.clean_latex_text(title_match.group(1))

        # Extract abstract - be more careful with math content
        abstract_match = re.search(r'\\abstract\{(.*?)\}', content, re.DOTALL)
        if not abstract_match:
            abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', content, re.DOTALL)

        if abstract_match:
            abstract_text = abstract_match.group(1)
            # Extract math and citations before cleaning
            math_exprs = self.extract_math_expressions(abstract_text)
            citations = self.extract_citations(abstract_text)

            # Store for context
            if math_exprs:
                self.equations.extend(math_exprs)
            if citations:
                self.citations.update(citations)

            abstract = self.clean_latex_text(abstract_text)

        return title, abstract

    def parse_sections(self, content: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse document content and extract sections with paragraphs, preserving scientific content."""
        sections = []
        paragraphs = []

        # Extract document content between \begin{document} and \end{document}
        doc_match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', content, re.DOTALL)
        if doc_match:
            content = doc_match.group(1)

        # Split content by sections and subsections
        section_pattern = r'\\(sub)?section\{([^}]+)\}'
        parts = re.split(section_pattern, content)

        current_section = "Introduction"
        current_section_number = 0
        current_subsection = ""
        current_subsection_number = 0

        i = 0
        while i < len(parts):
            if i == 0:
                # Content before first section
                text = parts[i]
                if text.strip():
                    paras = self.extract_paragraphs_from_text(
                        text, current_section, current_section_number,
                        current_subsection, current_subsection_number
                    )
                    paragraphs.extend(paras)
                i += 1
            elif i + 2 < len(parts):
                # This is a section/subsection match
                is_subsection = parts[i] == "sub"
                section_title = self.clean_latex_text(parts[i + 1])
                section_text = parts[i + 2]

                if is_subsection:
                    current_subsection = section_title
                    current_subsection_number += 1
                else:
                    current_section = section_title
                    current_section_number += 1
                    current_subsection = ""
                    current_subsection_number = 0

                # Extract paragraphs from this section's text
                paras = self.extract_paragraphs_from_text(
                    section_text, current_section, current_section_number,
                    current_subsection, current_subsection_number
                )
                paragraphs.extend(paras)

                i += 3
            else:
                i += 1

        # Create sections summary
        section_titles = list(set([(p['section_number'], p['section_title']) for p in paragraphs]))
        section_titles.sort(key=lambda x: x[0])

        for sec_num, sec_title in section_titles:
            sections.append({
                "section_number": sec_num,
                "section_title": sec_title,
                "subsections": list(set([p['subsection_title'] for p in paragraphs
                                       if p['section_number'] == sec_num and p['subsection_title']]))
            })

        return sections, paragraphs

    def extract_paragraphs_from_text(self, text: str, section_title: str, section_number: int,
                                   subsection_title: str, subsection_number: int) -> List[Dict[str, Any]]:
        """Extract individual paragraphs from a text block while preserving scientific content."""
        paragraphs = []

        # Clean the text while preserving important content
        cleaned_text = self.clean_latex_text(text)

        # Split into paragraphs (double newlines or more)
        para_splits = re.split(r'\n\s*\n+', cleaned_text)

        for para_text in para_splits:
            para_text = para_text.strip()

            # Skip empty paragraphs
            if len(para_text) < 10:
                continue

            # Skip paragraphs that are only LaTeX commands (but not math)
            if self.is_only_latex_commands(para_text):
                continue

            # Extract context information
            math_exprs = self.extract_math_expressions(para_text)
            citations = self.extract_citations(para_text)

            # Store context
            if math_exprs:
                self.equations.extend(math_exprs)
            if citations:
                self.citations.update(citations)

            self.paragraph_counter += 1

            paragraph = {
                "text": para_text,
                "section_number": section_number,
                "section_title": section_title,
                "subsection_number": subsection_number if subsection_title else None,
                "subsection_title": subsection_title if subsection_title else None,
                "page_number": None,
                "paragraph_type": self.determine_paragraph_type(para_text),
                "math_expressions": math_exprs,  # Additional context
                "citations": list(citations),   # Additional context
            }

            paragraphs.append(paragraph)

        return paragraphs

    def is_only_latex_commands(self, text: str) -> bool:
        """Check if text is only LaTeX commands and should be skipped."""
        # Remove preserved content temporarily
        temp_text = text
        for pattern in self.preserve_commands:
            temp_text = re.sub(pattern, '', temp_text)

        # Remove known commands
        temp_text = re.sub(r'\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})*', '', temp_text)

        # Check if anything meaningful remains
        meaningful_text = re.sub(r'[^\w\s]', '', temp_text).strip()

        return len(meaningful_text) < 5

    def determine_paragraph_type(self, text: str) -> str:
        """Determine the type of paragraph based on content and context."""
        text_lower = text.lower()

        # Check for math-heavy content
        if len(re.findall(r'\$[^$]+\$', text)) > 2:
            return 'mathematical'

        # Check for citation-heavy content
        if len(re.findall(r'\\cite\{[^}]+\}', text)) > 3:
            return 'literature_review'

        # Standard classifications
        if any(word in text_lower for word in ['abstract', 'summary']):
            return 'abstract'
        elif any(word in text_lower for word in ['introduction', 'background']):
            return 'introduction'
        elif any(word in text_lower for word in ['method', 'approach', 'algorithm', 'procedure']):
            return 'methodology'
        elif any(word in text_lower for word in ['result', 'finding', 'measurement', 'data']):
            return 'results'
        elif any(word in text_lower for word in ['conclusion', 'summary', 'discussion']):
            return 'conclusion'
        else:
            return 'body'

    def parse_latex_file(self, file_path: str) -> Dict[str, Any]:
        """Parse a LaTeX file and return structured data preserving physics content."""
        logger.info(f"Parsing LaTeX file with physics preservation: {file_path}")

        try:
            file_path_obj = Path(file_path)
            self.base_directory = file_path_obj.parent

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Resolve all \input{} and \include{} commands
            logger.info(f"Resolving includes from base directory: {self.base_directory}")
            content = self.resolve_includes(content, self.base_directory)

            # Reset context trackers
            self.equations = []
            self.figures = []
            self.tables = []
            self.citations = set()

            # Extract title and abstract
            title, abstract = self.extract_title_and_abstract(content)

            # Parse sections and paragraphs
            sections, paragraphs = self.parse_sections(content)

            # Add abstract as first paragraph if it exists
            if abstract:
                abstract_para = {
                    "text": abstract,
                    "section_number": 0,
                    "section_title": "Abstract",
                    "subsection_number": None,
                    "subsection_title": None,
                    "page_number": None,
                    "paragraph_type": "abstract",
                    "math_expressions": self.extract_math_expressions(abstract),
                    "citations": list(self.extract_citations(abstract)),
                }
                paragraphs.insert(0, abstract_para)

                sections.insert(0, {
                    "section_number": 0,
                    "section_title": "Abstract",
                    "subsections": []
                })

            # Create final document structure with additional physics context
            document_data = {
                "document_info": {
                    "title": title,
                    "total_pages": None,
                    "sections": sections,
                    "total_equations": len(self.equations),
                    "total_citations": len(self.citations),
                    "citation_keys": list(self.citations),
                },
                "paragraphs": paragraphs,
                "physics_context": {
                    "equations": self.equations[:10],  # Sample of equations
                    "citations": list(self.citations),
                    "has_math": len(self.equations) > 0,
                    "has_figures": len(self.figures) > 0,
                    "has_tables": len(self.tables) > 0,
                }
            }

            logger.info(f"Successfully parsed with {len(paragraphs)} paragraphs, "
                       f"{len(self.equations)} math expressions, {len(self.citations)} citations")
            return document_data

        except Exception as e:
            logger.error(f"Error parsing LaTeX file: {e}")
            raise

    def save_to_json(self, data: Dict[str, Any], output_path: str):
        """Save parsed data to JSON file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved parsed data to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")
            raise


def main():
    """Main function to demonstrate physics-friendly LaTeX parsing."""
    parser = PhysicsLaTeXParser()

    # Parse the HHH_SPA_Net project
    latex_file = "docs/LaTex/HHH_SPA_Net/main.tex"
    output_file = "reports/physics_parsed_latex.json"

    try:
        # Parse LaTeX file
        parsed_data = parser.parse_latex_file(latex_file)

        # Save to JSON
        parser.save_to_json(parsed_data, output_file)

        # Print summary
        print(f"Physics-friendly LaTeX parsing completed:")
        print(f"- Title: {parsed_data['document_info']['title']}")
        print(f"- Sections: {len(parsed_data['document_info']['sections'])}")
        print(f"- Paragraphs: {len(parsed_data['paragraphs'])}")
        print(f"- Math expressions: {parsed_data['document_info']['total_equations']}")
        print(f"- Citations: {parsed_data['document_info']['total_citations']}")

        # Show sample paragraphs with preserved content
        print("\nSample paragraphs with preserved physics content:")
        for i, para in enumerate(parsed_data['paragraphs'][:3]):
            print(f"\n{i+1}. Section {para['section_number']}: {para['section_title']}")
            print(f"   Type: {para['paragraph_type']}")
            print(f"   Math expressions: {len(para.get('math_expressions', []))}")
            print(f"   Citations: {len(para.get('citations', []))}")
            print(f"   Text: {para['text'][:150]}...")

    except Exception as e:
        logger.error(f"Failed to parse LaTeX file: {e}")


if __name__ == "__main__":
    main()