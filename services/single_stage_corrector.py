"""
Single-stage CMS Style Corrector that identifies and applies corrections in one pass.
Outputs structured results with original paragraph, corrected paragraph, and detailed edit list.
"""

import logging
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from llm_support.llm_providers import LLMManager
from models import Rulebook, settings_manager
from models.rule_models import set_embedding_model_name

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CMSSingleStageCorrector:
    """
    Single-stage corrector that identifies and applies CMS style corrections in one LLM pass.
    Returns structured output with original text, corrected text, and detailed edit explanations.
    """

    def __init__(self,
                 llm_model: str = None,
                 rag_model: str = None,
                 guidelines_path: str = None,
                 rulebook: Optional[Rulebook] = None):
        """
        Initialize the single-stage corrector.

        Args:
            llm_model: LLM model configuration (provider:model format)
            rag_model: Embedding model for rule retrieval
            guidelines_path: Path to CMS guidelines JSON
            rulebook: Optional project-specific rulebook to use instead of global rules
        """

        logger.info("Initializing CMS Single-Stage Corrector...")

        # Use settings_manager defaults if not provided
        llm_model = llm_model or settings_manager.llm_model
        rag_model = rag_model or settings_manager.rag_model
        rag_model = rag_model or settings_manager.rag_model
        guidelines_path = Path(guidelines_path or settings_manager.rules_json)

        # Store the optional project rulebook
        self.rulebook = rulebook

        # Initialize LLM manager
        llm_settings = {
            'max_tokens': settings_manager.max_tokens,
            'temperature': settings_manager.temperature,
            'cache_dir': str(settings_manager.cache_dir),
            'device': settings_manager.device,
            'api_key': self._get_api_key_for_provider(llm_model),
        }

        self.llm_manager = LLMManager(
            primary_model=llm_model,
            **llm_settings
        )

        # Configure embedding model for rulebook-based retrieval
        if rag_model:
            set_embedding_model_name(rag_model)

        # Initialize global rulebook fallback
        self.global_rulebook = self._load_global_rulebook(guidelines_path)

        # Check system availability
        provider_status = self.llm_manager.get_available_providers()
        logger.info(f"LLM Status: {provider_status}")

        if not any(provider_status.values()):
            raise Exception("No LLM providers available. Please check API keys and model configurations.")

        logger.info("Single-stage corrector initialized successfully")

    def set_rulebook(self, rulebook: Optional[Rulebook]):
        """Set or update the project-specific rulebook."""
        self.rulebook = rulebook
        if rulebook and rulebook.has_rules():
            logger.info(f"Using project rulebook with {rulebook.get_rule_count()} rules")
        else:
            logger.info("Using global CMS rulebook")

    def _load_global_rulebook(self, guidelines_path: Path) -> Rulebook:
        """Load the global CMS rulebook from disk to use as fallback."""
        rulebook = Rulebook()

        if not guidelines_path.exists():
            logger.warning(f"CMS guidelines not found at {guidelines_path}")
            return rulebook

        try:
            logger.info(f"Loading CMS guidelines from {guidelines_path}")
            success = rulebook.import_from_json(str(guidelines_path))
            if success:
                logger.info(f"Loaded {rulebook.get_rule_count()} CMS rules into global rulebook")
            else:
                logger.warning("CMS guidelines file could not be loaded")
        except Exception as exc:
            logger.error(f"Failed to load CMS guidelines: {exc}")

        return rulebook

    def _get_api_key_for_provider(self, model_config: str) -> Optional[str]:
        """Get API key for specific provider based on model name."""
        if not model_config:
            return settings_manager.anthropic_api_key

        # Handle legacy format with colon
        if ":" in model_config:
            provider = model_config.split(":")[0].lower()
            if provider == "anthropic":
                return settings_manager.anthropic_api_key
            elif provider == "openai":
                return settings_manager.openai_api_key
            return None

        # Auto-detect provider from model name
        model_lower = model_config.lower()
        if model_lower.startswith("claude"):
            return settings_manager.anthropic_api_key
        elif model_lower.startswith(("gpt", "o1", "o3", "o4")):
            return settings_manager.openai_api_key
        elif model_lower.startswith("ollama"):
            return None  # Ollama doesn't need API key

        # Default to Anthropic
        return settings_manager.anthropic_api_key

    def correct_paragraph(self, paragraph: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Perform single-stage correction on a paragraph.

        Args:
            paragraph: Input paragraph text
            top_k: Number of top rules to consider

        Returns:
            Dictionary containing:
            - original: Original paragraph
            - corrected: Corrected paragraph
            - changed: Boolean indicating if changes were made
            - edits: List of specific edits made
            - rules_used: List of rules that guided the corrections
        """

        logger.info("Processing paragraph for single-stage correction...")

        try:
            # Step 1: Retrieve relevant rules
            # Choose which rulebook to use (project-specific overrides global fallback)
            logger.debug("Retrieving relevant rules...")
            active_rulebook = None
            if self.rulebook and self.rulebook.has_rules():
                logger.debug(f"Using project rulebook with {self.rulebook.get_rule_count()} rules")
                active_rulebook = self.rulebook
            elif self.global_rulebook and self.global_rulebook.has_rules():
                logger.debug("Using global fallback rulebook")
                active_rulebook = self.global_rulebook

            if not active_rulebook:
                logger.warning("No CMS rules are available for retrieval")
                return {
                    'original': paragraph,
                    'corrected': paragraph,
                    'changed': False,
                    'edits': [],
                    'rules_used': [],
                    'error': "No CMS rulebook available for retrieval"
                }

            relevant_rules = active_rulebook.search(paragraph, top_k=top_k)

            if not relevant_rules:
                logger.warning("No relevant rules found for paragraph")
                return {
                    'original': paragraph,
                    'corrected': paragraph,
                    'changed': False,
                    'edits': [],
                    'rules_used': [],
                    'error': None
                }

            # Step 2: Create single-stage correction prompt
            logger.debug("Creating correction prompt...")
            prompt = self._create_single_stage_prompt(paragraph, relevant_rules)

            # Step 3: Get structured correction from LLM
            logger.debug("Calling LLM for corrections...")
            response = self.llm_manager.generate(prompt)

            if not response:
                raise Exception("No response from LLM")

            # Step 4: Parse structured response
            logger.debug("Parsing LLM response...")
            sanitized_response = self._sanitize_llm_output(response)
            result = self._parse_structured_response(paragraph, sanitized_response, raw_response=response)

            logger.info(f"Correction completed. Changes made: {result['changed']}")
            return result

        except Exception as e:
            logger.error(f"Error during correction: {e}")
            return {
                'original': paragraph,
                'corrected': paragraph,
                'changed': False,
                'edits': [],
                'rules_used': [],
                'error': str(e)
            }

    def _create_single_stage_prompt(self, paragraph: str, rules: List[Tuple[Dict, float]]) -> str:
        """
        Create single-stage prompt that identifies and applies corrections.

        Args:
            paragraph: Input paragraph
            rules: Retrieved relevant rules with similarity scores

        Returns:
            Complete prompt for single-stage correction
        """

        # Format rules for prompt
        rules_text = self._format_rules_for_prompt(rules)

        prompt = f"""You are a CMS physics paper style editor. Analyze the paragraph below against the provided CMS style guidelines and provide corrections in the exact JSON format specified.

CMS STYLE GUIDELINES:
{rules_text}

PARAGRAPH TO ANALYZE:
{paragraph}

TASK: Analyze the paragraph and provide corrections following CMS guidelines. Output your response in this exact JSON format:

{{
  "corrected_paragraph": "The fully corrected paragraph text here",
  "changes_made": true/false,
  "edits": [
    {{
      "original": "exact text that was changed",
      "corrected": "exact replacement text",
      "rule_title": "Title of the CMS rule that guided this change",
      "rule_content": "Brief explanation of why this change was needed based on CMS guidelines",
      "confidence": 0.0-1.0 numeric confidence score for this edit
    }}
  ]
}}

IMPORTANT INSTRUCTIONS:
1. If no corrections are needed, set "changes_made" to false and "edits" to empty array
2. For each edit, provide the exact original text and its exact replacement
3. Reference specific CMS guidelines that motivated each change
4. If you make a correction not explicitly covered by the provided rules, explain your reasoning in "rule_content"
5. Set "confidence" to a numeric value between 0.0 and 1.0 indicating your certainty in the edit
6. Ensure the corrected_paragraph incorporates ALL edits
7. Output ONLY valid JSON, no other text

Your response:"""

        return prompt

    def _format_rules_for_prompt(self, rules: List[Tuple[Dict, float]]) -> str:
        """Format retrieved rules for the prompt."""
        formatted_rules = []

        for i, (rule, score) in enumerate(rules, 1):
            rule_text = f"""Rule {i}: {rule['title']}
Content: {rule['content']}
Positive examples: {rule.get('positive_examples', 'N/A')}
Negative examples: {rule.get('negative_examples', 'N/A')}
Priority: {rule.get('priority', 'normal')}"""

            formatted_rules.append(rule_text)

        return "\n\n".join(formatted_rules)

    def _parse_structured_response(self, original_paragraph: str, response: str, raw_response: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse the structured JSON response from the LLM.

        Args:
            original_paragraph: Original paragraph text
            response: Sanitized LLM response
            raw_response: Full raw LLM response

        Returns:
            Parsed result dictionary
        """

        try:
            json_str = self._extract_json_block(response)

            # Parse JSON
            parsed = json.loads(json_str)

            # Validate required fields
            corrected = parsed.get('corrected_paragraph', original_paragraph)
            changed = parsed.get('changes_made', False)
            raw_edits = parsed.get('edits', [])
            edits = []
            for edit in raw_edits:
                normalized_edit = dict(edit)
                try:
                    confidence = float(normalized_edit.get('confidence', 0.8))
                except (TypeError, ValueError):
                    confidence = 0.8
                confidence = max(0.0, min(confidence, 1.0))
                normalized_edit['confidence'] = confidence
                edits.append(normalized_edit)

            # Extract rules used from edits
            rules_used = []
            for edit in edits:
                if 'rule_title' in edit and edit['rule_title'] not in [r['title'] for r in rules_used]:
                    rules_used.append({
                        'title': edit['rule_title'],
                        'content': edit.get('rule_content', '')
                    })

            # Validate that corrected text is different if changes_made is True
            if changed and corrected.strip() == original_paragraph.strip():
                logger.warning("LLM reported changes but corrected text is identical to original")
                changed = False
                edits = []
                rules_used = []

            return {
                'original': original_paragraph,
                'corrected': corrected,
                'changed': changed,
                'edits': edits,
                'rules_used': rules_used,
                'error': None
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {raw_response or response}")

            # Fallback: try to extract corrected text manually
            corrected = self._extract_corrected_text_fallback(raw_response or response, original_paragraph)

            return {
                'original': original_paragraph,
                'corrected': corrected,
                'changed': corrected != original_paragraph,
                'edits': [{"original": "parsing error", "corrected": "manual extraction used",
                          "rule_title": "System Error", "rule_content": f"JSON parsing failed: {str(e)}",
                          "confidence": 0.0}] if corrected != original_paragraph else [],
                'rules_used': [],
                'error': f"JSON parsing error: {str(e)}"
            }

        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}")
            return {
                'original': original_paragraph,
                'corrected': original_paragraph,
                'changed': False,
                'edits': [],
                'rules_used': [],
                'error': f"Response parsing error: {str(e)}"
            }

    def _extract_corrected_text_fallback(self, response: str, original: str) -> str:
        """Fallback method to extract corrected text if JSON parsing fails."""

        # Look for common patterns in LLM responses
        patterns = [
            r'"corrected_paragraph":\s*"([^"]*)"',
            r'corrected.*?:\s*"([^"]*)"',
            r'Corrected.*?:\s*(.+?)(?:\n|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                corrected = match.group(1).strip()
                if corrected and corrected != original:
                    return corrected

        # If no pattern matches, return original
        return original

    def _sanitize_llm_output(self, response: str) -> str:
        """Strip chain-of-thought traces and markdown fences from LLM responses."""
        if not response:
            return ""

        cleaned = response.replace("\r\n", "\n").strip()
        cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.replace("```", "").strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]

        return cleaned.strip()

    def _extract_json_block(self, response: str) -> str:
        """Extract the JSON portion from the response string."""
        sanitized = response.strip()
        json_match = re.search(r'\{.*\}', sanitized, re.DOTALL)
        if json_match:
            return json_match.group(0)
        return sanitized


class CMSDocumentOrchestrator:
    """Orchestrates the single-stage correction process for entire documents."""

    def __init__(self, corrector: CMSSingleStageCorrector):
        """Initialize with a single-stage corrector."""
        self.corrector = corrector

    def correct_paragraph(self, paragraph: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Correct a single paragraph using the single-stage approach.

        Args:
            paragraph: Input paragraph text
            top_k: Number of rules to consider

        Returns:
            Correction results with the expected format
        """

        result = self.corrector.correct_paragraph(paragraph, top_k)

        # Format for compatibility with existing main.py
        return {
            'original': result['original'],
            'corrected': result['corrected'],
            'changed': result['changed'],
            'suggestions': self._format_edits_as_suggestions(result['edits']) if result['edits'] else "No corrections needed",
            'edits': result['edits'],
            'rules_used': result['rules_used'],
            'error': result.get('error')
        }

    def _format_edits_as_suggestions(self, edits: List[Dict[str, str]]) -> str:
        """Format edits as a readable suggestions string."""
        suggestions = []
        for edit in edits:
            original = edit.get('original', 'N/A')
            corrected = edit.get('corrected', 'N/A')
            rule_title = edit.get('rule_title', 'General Style')
            rule_content = edit.get('rule_content', 'Style improvement')

            suggestion = f"Edit: {original} -> {corrected}\nReason: {rule_title}: {rule_content}"
            suggestions.append(suggestion)

        return "\n\n".join(suggestions)
