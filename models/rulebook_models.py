"""
Rulebook model for the StyleRev CMS system.

This module provides the Rulebook class that manages a collection of rules
with import/export capabilities and semantic search functionality.
"""

from typing import List, Optional, Tuple, Dict, Any, Callable
import json
import os
import numpy as np
import logging

from .rule_models import Rule, RulePriority, EMBEDDING_AVAILABLE, _get_embedding_model

logger = logging.getLogger(__name__)


class Rulebook:
    """
    Rulebook class that maintains a collection of rules.

    Provides methods for rule management, filtering, semantic search,
    and import/export operations.
    """

    def __init__(self, rules: Optional[List[Rule]] = None):
        """
        Initialize a Rulebook.

        Args:
            rules: Optional list of rules to initialize with
        """
        self.rules: List[Rule] = rules if rules else []

    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the rulebook."""
        self.rules.append(rule)

    def add_rule_at(self, index: int, rule: Rule) -> None:
        """Add a rule at a specific index."""
        self.rules.insert(index, rule)

    def remove_rule(self, rule: Rule) -> None:
        """Remove a rule from the rulebook."""
        if rule in self.rules:
            self.rules.remove(rule)

    def get_rule_by_id(self, rule_id: int) -> Optional[Rule]:
        """Get a rule by its ID."""
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        return None

    def find_rules(self, text: str, use_longest_match: bool = True) -> List[Rule]:
        """
        Find all rules that appear in the given text.

        Args:
            text: The text to search in
            use_longest_match: If True, use longest-match-first strategy

        Returns:
            List of rules found in the text
        """
        if not use_longest_match:
            return [rule for rule in self.rules if rule.matches_text(text)]

        # Longest-match-first strategy
        sorted_rules = sorted(self.rules, key=lambda r: len(r.title), reverse=True)
        matched_spans = []
        matched_rules = []

        for rule in sorted_rules:
            rule_matches = rule.find_matches(text)

            for start, end in rule_matches:
                overlaps = False
                for existing_start, existing_end in matched_spans:
                    if not (end <= existing_start or start >= existing_end):
                        overlaps = True
                        break

                if not overlaps:
                    matched_spans.append((start, end))
                    if rule not in matched_rules:
                        matched_rules.append(rule)

        return matched_rules

    def find_k_nearest_rules(
        self,
        text: str,
        k: int = 5,
        use_context: bool = True
    ) -> List[Tuple[Rule, float]]:
        """
        Find K nearest rules to text using semantic similarity.

        Args:
            text: Text to find similar rules for
            k: Number of rules to return
            use_context: Use context_embedding (True) or rule_embedding (False)

        Returns:
            List of (Rule, similarity_score) tuples, sorted by score descending.
        """
        if not self.rules or not EMBEDDING_AVAILABLE:
            return []

        model = _get_embedding_model()
        if not model:
            return []

        # Encode the query text
        text_embedding = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Collect valid rules and their embeddings
        rule_embeddings_list = []
        valid_rules = []

        for rule in self.rules:
            embedding = rule.context_embedding if use_context else rule.rule_embedding
            if embedding is not None and embedding.size > 0:
                rule_embeddings_list.append(embedding)
                valid_rules.append(rule)

        if not valid_rules:
            return []

        # Stack embeddings and compute similarities
        rule_embeddings = np.vstack(rule_embeddings_list)
        similarities = np.dot(rule_embeddings, text_embedding)

        # Get top K
        k_actual = min(k, len(valid_rules))
        top_k_indices = np.argsort(similarities)[-k_actual:][::-1]

        result = []
        for idx in top_k_indices:
            result.append((valid_rules[idx], float(similarities[idx])))

        return result

    def search(self, query_text: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for rules relevant to query text.

        Returns results in format compatible with RulesDB.

        Args:
            query_text: Text to search for
            top_k: Number of results to return

        Returns:
            List of (rule_dict, similarity_score) tuples
        """
        results = self.find_k_nearest_rules(query_text, k=top_k, use_context=True)

        # Convert to dict format for compatibility
        return [(self._rule_to_dict_for_search(rule), score) for rule, score in results]

    def _rule_to_dict_for_search(self, rule: Rule) -> Dict[str, Any]:
        """Convert rule to dict format expected by corrector service."""
        return {
            'rule_id': rule.rule_id,
            'title': rule.title,
            'content': rule.content,
            'section_path': rule.section_path,
            'priority': rule.priority.name.lower() if rule.priority else 'normal',
            'positive_examples': rule.positive_examples,
            'negative_examples': rule.negative_examples
        }

    def get_relevant_rules(
        self,
        paragraph_text: str,
        context_before: str = "",
        context_after: str = "",
        top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Get rules relevant to a paragraph with priority-based re-ranking.

        Args:
            paragraph_text: The paragraph to find rules for
            context_before: Text before the paragraph
            context_after: Text after the paragraph
            top_k: Number of rules to return

        Returns:
            List of (rule_dict, final_score) tuples
        """
        full_context = f"{context_before} {paragraph_text} {context_after}".strip()

        # Get initial candidates
        candidates = self.find_k_nearest_rules(full_context, k=min(15, len(self.rules)), use_context=True)

        # Re-rank based on priority
        ranked_rules = []
        for rule, similarity_score in candidates:
            final_score = similarity_score

            # Priority boost
            if rule.priority == RulePriority.CRUCIAL:
                final_score *= 2
            elif rule.priority == RulePriority.MANDATORY:
                final_score *= 1.2
            elif rule.priority == RulePriority.RECOMMENDED:
                final_score *= 1.1
            elif rule.priority == RulePriority.NONE:
                final_score *= 0

            # Boost if examples overlap with query
            if rule.negative_examples and len(rule.negative_examples) > 10:
                query_words = set(paragraph_text.lower().split())
                example_words = set(rule.negative_examples.lower().split())
                overlap = len(query_words & example_words) / max(len(query_words), 1)
                if overlap > 0.3:
                    final_score *= 1.1

            ranked_rules.append((self._rule_to_dict_for_search(rule), final_score))

        ranked_rules.sort(key=lambda x: x[1], reverse=True)
        return ranked_rules[:top_k]

    def import_from_json(self, file_path: str, progress_callback: Optional[Callable] = None) -> bool:
        """
        Import rules from a JSON file (adds to existing rules).

        Args:
            file_path: Path to the JSON file
            progress_callback: Optional callback(current, total, message)

        Returns:
            True if import was successful
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle both single rule and list
            if isinstance(data, dict):
                rules_list = [data]
            elif isinstance(data, list):
                rules_list = data
            else:
                logger.error(f"Invalid JSON format in {file_path}")
                return False

            # Create rules without auto-generating embeddings
            new_rules = []
            for rule_data in rules_list:
                rule = Rule.from_dict(rule_data, skip_auto_generation=True)
                new_rules.append(rule)
                self.add_rule(rule)

            # Generate embeddings in batch
            if new_rules and EMBEDDING_AVAILABLE:
                Rule.generate_embeddings_batch(new_rules, progress_callback=progress_callback)

            logger.info(f"Imported {len(new_rules)} rules from {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error importing from JSON: {e}")
            return False

    def import_new_from_json(self, file_path: str, progress_callback: Optional[Callable] = None) -> bool:
        """
        Import rules from JSON, replacing all existing rules.

        Args:
            file_path: Path to the JSON file
            progress_callback: Optional callback(current, total, message)

        Returns:
            True if import was successful
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, dict):
                rules_list = [data]
            elif isinstance(data, list):
                rules_list = data
            else:
                logger.error(f"Invalid JSON format in {file_path}")
                return False

            # Create new rules
            new_rules = []
            for rule_data in rules_list:
                rule = Rule.from_dict(rule_data, skip_auto_generation=True)
                new_rules.append(rule)

            # Clear and replace
            self.rules.clear()
            self.rules.extend(new_rules)

            # Generate embeddings in batch
            if new_rules and EMBEDDING_AVAILABLE:
                Rule.generate_embeddings_batch(new_rules, progress_callback=progress_callback)

            logger.info(f"Replaced rulebook with {len(new_rules)} rules from {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error importing from JSON: {e}")
            return False

    def export_to_json(self, file_path: str) -> bool:
        """
        Export rules to a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            True if export was successful
        """
        try:
            rules_data = []
            for rule in self.rules:
                rule_dict = {
                    'rule_id': rule.rule_id,
                    'section_path': rule.section_path,
                    'title': rule.title,
                    'content': rule.content,
                    'priority': rule.priority.value if rule.priority else 0,
                    'positive_examples': rule.positive_examples,
                    'negative_examples': rule.negative_examples
                }
                rules_data.append(rule_dict)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(rules_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported {len(self.rules)} rules to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            return False

    def get_priorities(self) -> List[str]:
        """Get all unique priorities from the rulebook."""
        priorities = set()
        for rule in self.rules:
            if rule.priority:
                priorities.add(rule.priority.name)
        return sorted(list(priorities))

    def create_new_rule_with_defaults(self) -> Rule:
        """Create a new rule with default values."""
        # Find next available ID
        max_id = max((r.rule_id for r in self.rules), default=0)

        return Rule(
            rule_id=max_id + 1,
            title="New Rule",
            content="Rule description",
            priority=RulePriority.NORMAL
        )

    def get_rule_count(self) -> int:
        """Get the total number of rules."""
        return len(self.rules)

    def is_empty(self) -> bool:
        """Check if the rulebook is empty."""
        return len(self.rules) == 0

    def has_rules(self) -> bool:
        """Check if the rulebook has any rules."""
        return len(self.rules) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the rulebook to a dictionary."""
        return {
            "rules": [rule.to_dict() for rule in self.rules]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], progress_callback: Optional[Callable] = None) -> 'Rulebook':
        """
        Create a rulebook from a dictionary.

        Args:
            data: Dictionary containing rulebook data
            progress_callback: Optional callback for embedding generation progress
        """
        rules_data = data.get("rules", [])

        # Create rules without auto-generating embeddings
        rules = []
        for rule_data in rules_data:
            rule = Rule.from_dict(rule_data, skip_auto_generation=True)
            rules.append(rule)

        rulebook = cls(rules=rules)

        # Generate embeddings for rules that don't have them
        rules_needing_embeddings = [r for r in rules if r._rule_embedding is None]
        if rules_needing_embeddings and EMBEDDING_AVAILABLE:
            Rule.generate_embeddings_batch(rules_needing_embeddings, progress_callback=progress_callback)

        return rulebook
