"""
Rule models for the StyleRev CMS Rulebook system.

This module provides Rule and RulePriority models with embedding support
for semantic similarity search.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import numpy as np
import re
import base64
import logging

logger = logging.getLogger(__name__)

# Lazy import of sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    _EMBEDDING_MODEL = None  # Lazy loaded on first use
    EMBEDDING_AVAILABLE = True
except ImportError:
    _EMBEDDING_MODEL = None
    EMBEDDING_AVAILABLE = False
    logger.warning("sentence-transformers not available, embeddings will be disabled")

_EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

# Model version for cache invalidation
EMBEDDING_MODEL_VERSION = "v2"


def set_embedding_model_name(model_name: str):
    """
    Configure which sentence-transformer model should be used for rule embeddings.

    Args:
        model_name: HuggingFace model identifier
    """
    global _EMBEDDING_MODEL_NAME, _EMBEDDING_MODEL
    if not model_name:
        return

    if model_name != _EMBEDDING_MODEL_NAME:
        _EMBEDDING_MODEL_NAME = model_name
        _EMBEDDING_MODEL = None  # Force reload with the new model


def _get_embedding_model():
    """Lazy load the embedding model to avoid startup delay."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None and EMBEDDING_AVAILABLE:
        _EMBEDDING_MODEL = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL


class RulePriority(Enum):
    """Priority levels for rules."""
    NONE = 0
    RECOMMENDED = 1
    NORMAL = 2
    MANDATORY = 3
    CRUCIAL = 4


@dataclass
class Rule:
    """
    A Rule in Rulebook with vector embeddings for semantic similarity search.

    Each rule maintains two embeddings:
    1. rule_embedding: Embedding of the rule title (primary identifier)
    2. context_embedding: Embedding optimized for text-pattern matching
    """
    rule_id: int
    title: str
    content: str
    section_path: str = ""
    priority: RulePriority = RulePriority.NORMAL
    positive_examples: str = ""
    negative_examples: str = ""

    # Private embedding fields (not serialized directly)
    _rule_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    _context_embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        """Generate embeddings after initialization if content is available."""
        # Convert priority from int if needed
        if isinstance(self.priority, int):
            self.priority = RulePriority(self.priority)

        # Generate embeddings if we have content and embeddings are available
        if EMBEDDING_AVAILABLE and self.title and self._rule_embedding is None:
            self._generate_embeddings()

    @property
    def rule_embedding(self) -> Optional[np.ndarray]:
        """Get the embedding of the rule title."""
        return self._rule_embedding

    @property
    def context_embedding(self) -> Optional[np.ndarray]:
        """Get the embedding optimized for text-pattern matching."""
        return self._context_embedding

    def _generate_context_string_for_embedding(self) -> str:
        """
        Generate the context string for embedding optimized for text-pattern matching.

        Format: "Fixes text like: {negative_examples} | Correct version: {positive_examples} |
                 Applies to: {content} | Rule: {title} | Category: {section_path}"
        """
        parts = []

        # Lead with examples - these show WHAT TEXT this rule applies to
        if self.negative_examples:
            parts.append(f"Fixes text like: {self.negative_examples}")
        if self.positive_examples:
            parts.append(f"Correct version: {self.positive_examples}")

        # Rule description focused on applicability
        if self.content:
            parts.append(f"Applies to: {self.content}")
        if self.title:
            parts.append(f"Rule: {self.title}")

        # Context for disambiguation
        if self.section_path:
            # Get the most specific category
            category = self.section_path.split('/')[-2] if '/' in self.section_path else self.section_path
            if category:
                parts.append(f"Category: {category}")

        return " | ".join(parts)

    def _generate_embeddings(self):
        """Generate both rule and context embeddings."""
        model = _get_embedding_model()
        if not model:
            return

        # Generate rule embedding from title
        if self.title:
            self._rule_embedding = model.encode(
                self.title,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

        # Generate context embedding
        context_string = self._generate_context_string_for_embedding()
        if context_string:
            self._context_embedding = model.encode(
                context_string,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

    def regenerate_embeddings(self):
        """Regenerate embeddings (useful if rule properties change)."""
        self._generate_embeddings()

    @staticmethod
    def generate_embeddings_batch(rules: List['Rule'], progress_callback=None):
        """
        Generate embeddings for multiple rules in batch (5-10x faster than individual).

        Args:
            rules: List of Rule objects to generate embeddings for
            progress_callback: Optional callback function(current, total, message)
        """
        if not EMBEDDING_AVAILABLE:
            return

        model = _get_embedding_model()
        if not model:
            return

        # Separate rules that need embeddings
        rules_to_process = [r for r in rules if r.title and r._rule_embedding is None]

        if not rules_to_process:
            return

        if progress_callback:
            progress_callback(0, len(rules_to_process), "Preparing rule data for batch processing...")

        # Prepare all text for batch encoding
        rule_texts = [r.title for r in rules_to_process]
        context_texts = [r._generate_context_string_for_embedding() for r in rules_to_process]

        if progress_callback:
            progress_callback(0, len(rules_to_process), f"Encoding {len(rules_to_process)} rules...")

        # Batch encode all rule texts
        rule_embeddings = model.encode(
            rule_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32
        )

        if progress_callback:
            progress_callback(len(rules_to_process) // 2, len(rules_to_process),
                            f"Encoding {len(rules_to_process)} context strings...")

        # Batch encode all context texts
        context_embeddings = model.encode(
            context_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32
        )

        # Assign embeddings back to rules
        for i, rule in enumerate(rules_to_process):
            rule._rule_embedding = rule_embeddings[i]
            rule._context_embedding = context_embeddings[i]

            if progress_callback:
                progress_callback(i + 1, len(rules_to_process),
                                f"Processed {i + 1}/{len(rules_to_process)} rules")

        if progress_callback:
            progress_callback(len(rules_to_process), len(rules_to_process),
                            f"Completed! Generated embeddings for {len(rules_to_process)} rules.")

    def similarity_to_text(self, text: str, use_context: bool = True) -> float:
        """
        Calculate cosine similarity between this rule and given text.

        Args:
            text: Text to compare against
            use_context: If True, use context_embedding; otherwise use rule_embedding

        Returns:
            Similarity score (0.0 to 1.0), or 0.0 if embeddings not available
        """
        model = _get_embedding_model()
        if not model:
            return 0.0

        # Choose which embedding to use
        rule_vec = self.context_embedding if use_context else self.rule_embedding

        if rule_vec is None:
            return 0.0

        # Encode the input text
        text_vec = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Calculate cosine similarity (normalized vectors = dot product)
        similarity = np.dot(rule_vec, text_vec)

        return float(similarity)

    def matches_text(self, text: str) -> bool:
        """
        Check if the rule title appears in the given text.
        """
        return len(self.find_matches(text)) > 0

    def find_matches(self, text: str) -> List[Tuple[int, int]]:
        """
        Find all occurrences of this rule's title in the text.

        Returns:
            List of (start, end) positions of matches
        """
        if not self.title:
            return []

        matches = []
        pattern = re.escape(self.title)
        pattern = r'\b' + pattern + r'\b'

        for match in re.finditer(pattern, text, re.IGNORECASE):
            matches.append((match.start(), match.end()))

        return matches

    def rule_string(self) -> str:
        """Returns string format of the Rule (suitable for prompts)."""
        return f'''
Rule ID {self.rule_id}: {self.title}
{self.content}
Here are some negative examples:
{self.negative_examples}
Here are some positive examples:
{self.positive_examples}
'''

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the rule to a dictionary, including embeddings."""
        result = {
            "rule_id": self.rule_id,
            "section_path": self.section_path,
            "title": self.title,
            "content": self.content,
            "priority": self.priority.value if isinstance(self.priority, RulePriority) else self.priority,
            "positive_examples": self.positive_examples,
            "negative_examples": self.negative_examples
        }

        # Store embeddings if available
        if self._rule_embedding is not None and self._context_embedding is not None:
            result["_rule_embedding_base64"] = base64.b64encode(
                self._rule_embedding.astype(np.float32).tobytes()
            ).decode('utf-8')
            result["_context_embedding_base64"] = base64.b64encode(
                self._context_embedding.astype(np.float32).tobytes()
            ).decode('utf-8')
            result["_embedding_version"] = EMBEDDING_MODEL_VERSION

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any], skip_auto_generation: bool = False) -> 'Rule':
        """
        Create a rule from a dictionary, loading embeddings if available.

        Args:
            data: Dictionary containing rule data
            skip_auto_generation: If True, skip automatic embedding generation
        """
        # Extract embedding data
        rule_emb_base64 = data.get('_rule_embedding_base64')
        context_emb_base64 = data.get('_context_embedding_base64')
        emb_version = data.get('_embedding_version')

        # Map priority
        priority_value = data.get('priority', 2)
        if isinstance(priority_value, int):
            priority = RulePriority(priority_value)
        elif isinstance(priority_value, str):
            priority = RulePriority[priority_value.upper()]
        else:
            priority = RulePriority.NORMAL

        # Create rule without auto-generating embeddings initially
        rule = cls(
            rule_id=data.get('rule_id', 0),
            section_path=data.get('section_path', ''),
            title=data.get('title', ''),
            content=data.get('content', ''),
            priority=priority,
            positive_examples=data.get('positive_examples', ''),
            negative_examples=data.get('negative_examples', '')
        )

        # Load embeddings from base64 if available and version matches
        if rule_emb_base64 and context_emb_base64 and emb_version == EMBEDDING_MODEL_VERSION:
            try:
                rule._rule_embedding = np.frombuffer(
                    base64.b64decode(rule_emb_base64), dtype=np.float32
                )
                rule._context_embedding = np.frombuffer(
                    base64.b64decode(context_emb_base64), dtype=np.float32
                )
            except Exception as e:
                logger.warning(f"Failed to load embeddings: {e}")
                if not skip_auto_generation:
                    rule._generate_embeddings()
        elif not skip_auto_generation and rule._rule_embedding is None:
            # Generate if not skipping and not already generated
            rule._generate_embeddings()

        return rule
