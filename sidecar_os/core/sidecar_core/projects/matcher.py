"""Project name normalization and semantic matching."""

import re
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..llm import LLMService


class MatchConfidence(Enum):
    """Confidence levels for project matching."""
    HIGH = "high"       # 85%+: Auto-match silently
    MEDIUM = "medium"   # 60-85%: Ask for confirmation
    LOW = "low"         # <60%: Create new project


@dataclass
class ProjectMatchResult:
    """Result of project matching analysis."""

    # Normalized names
    canonical_id: str           # e.g., "lrp_ftg"
    display_name: str          # e.g., "Lrp Ftg"

    # Matching results
    matched_project_id: Optional[str] = None
    confidence_level: MatchConfidence = MatchConfidence.LOW
    confidence_score: float = 0.0
    reasoning: str = ""

    # Whether this should create a new project or use existing
    should_create_new: bool = True


class ProjectMatcher:
    """Handles project name normalization and semantic matching."""

    def __init__(self, llm_service: Optional[LLMService] = None):
        """Initialize project matcher.

        Args:
            llm_service: LLM service for semantic matching (optional)
        """
        self.llm_service = llm_service or LLMService()

    def normalize_project_name(self, raw_name: str) -> Tuple[str, str]:
        """Normalize project name to canonical format.

        Args:
            raw_name: Raw project name from user input

        Returns:
            Tuple of (canonical_id, display_name)
        """
        # Clean and normalize
        cleaned = raw_name.strip()

        # Convert to lowercase and replace spaces/hyphens with underscores
        canonical_id = re.sub(r'[^\w\s-]', '', cleaned.lower())
        canonical_id = re.sub(r'[\s-]+', '_', canonical_id)

        # Remove leading/trailing underscores
        canonical_id = canonical_id.strip('_')

        # Create display name (Title Case)
        display_name = ' '.join(word.capitalize() for word in canonical_id.split('_'))

        return canonical_id, display_name

    async def find_best_match(
        self,
        user_project: str,
        existing_projects: Dict[str, str]  # project_id -> display_name
    ) -> ProjectMatchResult:
        """Find best matching project using LLM semantic analysis.

        Args:
            user_project: User's project input
            existing_projects: Dict of existing project_id -> display_name

        Returns:
            ProjectMatchResult with matching decision
        """
        canonical_id, display_name = self.normalize_project_name(user_project)

        # If no existing projects, create new
        if not existing_projects:
            return ProjectMatchResult(
                canonical_id=canonical_id,
                display_name=display_name,
                should_create_new=True,
                reasoning="No existing projects to match against"
            )

        # Check for exact canonical ID match first
        if canonical_id in existing_projects:
            return ProjectMatchResult(
                canonical_id=canonical_id,
                display_name=existing_projects[canonical_id],
                matched_project_id=canonical_id,
                confidence_level=MatchConfidence.HIGH,
                confidence_score=1.0,
                should_create_new=False,
                reasoning=f"Exact match with existing project '{canonical_id}'"
            )

        # Use LLM for semantic matching
        if len(existing_projects) > 0:
            try:
                llm_result = await self._llm_semantic_match(user_project, existing_projects)
                if llm_result:
                    matched_id, confidence, reasoning = llm_result

                    # Determine confidence level
                    if confidence >= 0.85:
                        confidence_level = MatchConfidence.HIGH
                    elif confidence >= 0.60:
                        confidence_level = MatchConfidence.MEDIUM
                    else:
                        confidence_level = MatchConfidence.LOW

                    return ProjectMatchResult(
                        canonical_id=canonical_id,
                        display_name=display_name,
                        matched_project_id=matched_id if confidence >= 0.60 else None,
                        confidence_level=confidence_level,
                        confidence_score=confidence,
                        should_create_new=confidence < 0.60,
                        reasoning=reasoning
                    )

            except Exception as e:
                print(f"LLM semantic matching failed: {e}")

        # Fallback: create new project
        return ProjectMatchResult(
            canonical_id=canonical_id,
            display_name=display_name,
            should_create_new=True,
            reasoning="No semantic matches found"
        )

    async def _llm_semantic_match(
        self,
        user_project: str,
        existing_projects: Dict[str, str]
    ) -> Optional[Tuple[str, float, str]]:
        """Use LLM to find semantic matches with existing projects.

        Returns:
            Tuple of (matched_project_id, confidence_score, reasoning) or None
        """
        # Build existing projects context
        projects_list = []
        for proj_id, display_name in existing_projects.items():
            projects_list.append(f"'{proj_id}' (displayed as '{display_name}')")

        system_prompt = """You are a project name matching expert. Your job is to determine if a user's project reference matches any existing projects.

Consider these factors:
- Acronyms and abbreviations (e.g., "LRP FTG" = "lrp ftg")
- Word order variations (e.g., "FTG LRP" = "LRP FTG")
- Common project naming patterns
- Case variations and formatting differences

Respond with ONLY a JSON object in this exact format:
{
    "match": "project_id_if_match_found_or_null",
    "confidence": 0.95,
    "reasoning": "Brief explanation of why this matches or doesn't"
}

Confidence scale:
- 0.90-1.0: Extremely confident (clear semantic match)
- 0.75-0.89: Very confident (likely same project with variations)
- 0.60-0.74: Moderately confident (possible match, ask user)
- 0.0-0.59: Low confidence (different project)"""

        user_prompt = f"""User project reference: "{user_project}"

Existing projects:
{chr(10).join(projects_list)}

Does the user's project reference match any existing project? Consider acronyms, word order, and common variations."""

        try:
            response = await self.llm_service.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=200,
                temperature=0.1  # Low temperature for consistent matching
            )

            # Debug: Check what we got back
            content = response.content.strip()
            if not content:
                print(f"DEBUG: Empty response from LLM for project matching")
                return None

            # Strip markdown code blocks if present
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()

            # Parse JSON response
            import json
            try:
                result = json.loads(content)
            except json.JSONDecodeError as json_err:
                print(f"DEBUG: Invalid JSON from LLM: '{content[:100]}...' - {json_err}")
                # Try to extract information from non-JSON response
                if "no match" in content.lower() or "different project" in content.lower():
                    return (None, 0.1, "LLM indicated no match (non-JSON response)")
                else:
                    return None

            matched_project = result.get("match")
            confidence = float(result.get("confidence", 0.0))
            reasoning = result.get("reasoning", "")

            # Handle null values properly
            if matched_project == "null" or matched_project == "project_id_if_match_found_or_null":
                matched_project = None

            if matched_project and matched_project in existing_projects:
                return (matched_project, confidence, reasoning)
            else:
                return (None, confidence, reasoning)

        except Exception as e:
            print(f"Error in LLM semantic matching: {e}")
            return None

    def cleanup_project_name(self, messy_name: str) -> Optional[str]:
        """Clean up obviously bad project names from existing data.

        Args:
            messy_name: Potentially messy project name

        Returns:
            Cleaned name or None if should be deleted
        """
        # Common patterns of bad project names to filter out
        bad_patterns = [
            r'^yes\s+need\s+to\s+complete',
            r'^status\s+update$',
            r'^does\s+this\s+relate',
            r'^task\s*$',
            r'^[a-z\s]+\s+to\s+[a-z\s]+$',  # Generic phrases
        ]

        name_lower = messy_name.lower().strip()

        # Check against bad patterns
        for pattern in bad_patterns:
            if re.match(pattern, name_lower):
                return None  # Should be deleted

        # If it looks like clarification text rather than project name
        if len(name_lower.split()) > 4 and any(word in name_lower for word in
                                              ['complete', 'need to', 'relate to', 'this is']):
            return None

        # If it passes basic sanity checks, normalize it
        if len(name_lower.strip()) > 0:
            canonical_id, display_name = self.normalize_project_name(messy_name)
            return canonical_id

        return None