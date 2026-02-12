"""Intelligent Ask System for Natural Language Query Processing.

This module provides AI-powered question answering by orchestrating existing system
capabilities to handle complex user queries intelligently.
"""

import asyncio
import json
import logging
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional, Callable

from .llm.service import LLMService
from .state.models import SidecarState
from .artifacts.store import ArtifactStore

logger = logging.getLogger(__name__)


class CommandExecutionResult:
    """Result of executing a single command in the execution plan."""

    def __init__(self, command: str, success: bool, data: Any = None, error: str = None):
        self.command = command
        self.success = success
        self.data = data
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command,
            "success": self.success,
            "data": self.data,
            "error": self.error
        }


class IntelligentAskHandler:
    """Orchestrates existing system commands to answer complex natural language questions."""

    def __init__(self, state: SidecarState, artifact_store: ArtifactStore, llm_service: LLMService):
        self.state = state
        self.artifact_store = artifact_store
        self.llm_service = llm_service

        # Register available command functions
        self.command_registry = {
            "get_tasks_due_today": self._get_tasks_due_today,
            "get_overdue_tasks": self._get_overdue_tasks,
            "get_active_tasks": self._get_active_tasks,
            "get_completed_tasks": self._get_completed_tasks,
            "get_tasks_by_project": self._get_tasks_by_project,
            "get_project_artifacts": self._get_project_artifacts,
            "get_all_artifacts": self._get_all_artifacts,
            "get_artifacts_by_type": self._get_artifacts_by_type,
            "search_artifacts_by_content": self._search_artifacts_by_content,
            "get_project_details": self._get_project_details,
            "get_all_projects": self._get_all_projects,
            "count_tasks_by_status": self._count_tasks_by_status,
            "count_artifacts_by_type": self._count_artifacts_by_type,
            "get_recent_activity": self._get_recent_activity,
            "search_tasks_by_keywords": self._search_tasks_by_keywords,
            "get_high_priority_tasks": self._get_high_priority_tasks,
            "get_tasks_without_due_date": self._get_tasks_without_due_date,
        }

    async def handle_question(self, question: str) -> str:
        """Main entry point for handling intelligent questions.

        Args:
            question: Natural language question from user

        Returns:
            Natural language response
        """
        try:
            # Step 1: Analyze intent and create execution plan
            intent_analysis = await self.llm_service.analyze_question_intent(
                question=question,
                available_commands=list(self.command_registry.keys()),
                system_context=self._build_system_context()
            )

            confidence = intent_analysis.get('confidence', 0.0)
            if confidence < 0.4:
                return self._handle_low_confidence(question, intent_analysis)

            # Step 2: Execute the planned commands
            execution_results = await self._execute_plan(intent_analysis.get('execution_plan', []))

            # Step 3: Synthesize response
            response = await self.llm_service.synthesize_response(
                original_question=question,
                execution_results=[result.to_dict() for result in execution_results],
                intent_analysis=intent_analysis
            )

            return response

        except Exception as e:
            logger.error(f"Intelligent ask handling failed: {e}")
            return f"I encountered an error while processing your question: {str(e)}. Please try rephrasing your question."

    async def _execute_plan(self, execution_plan: List[Dict[str, Any]]) -> List[CommandExecutionResult]:
        """Execute the planned commands in sequence.

        Args:
            execution_plan: List of commands to execute with parameters

        Returns:
            List of execution results
        """
        results = []

        for step in execution_plan:
            command_name = step.get('command')
            parameters = step.get('parameters', {})
            fallback = step.get('fallback')

            # Try primary command
            result = await self._execute_command(command_name, parameters)

            # If primary failed and we have a fallback, try it
            if not result.success and fallback:
                logger.info(f"Primary command {command_name} failed, trying fallback: {fallback}")
                result = await self._execute_command(fallback, parameters)

            results.append(result)

        return results

    async def _execute_command(self, command_name: str, parameters: Dict[str, Any]) -> CommandExecutionResult:
        """Execute a single command with parameters.

        Args:
            command_name: Name of command to execute
            parameters: Parameters for the command

        Returns:
            Command execution result
        """
        try:
            if command_name not in self.command_registry:
                return CommandExecutionResult(
                    command=command_name,
                    success=False,
                    error=f"Unknown command: {command_name}"
                )

            command_func = self.command_registry[command_name]
            data = await command_func(**parameters)

            return CommandExecutionResult(
                command=command_name,
                success=True,
                data=data
            )

        except Exception as e:
            logger.error(f"Command execution failed for {command_name}: {e}")
            return CommandExecutionResult(
                command=command_name,
                success=False,
                error=str(e)
            )

    def _build_system_context(self) -> Dict[str, Any]:
        """Build current system context for LLM analysis."""
        return {
            "projects": {p.project_id: {"name": p.name, "aliases": p.aliases}
                        for p in self.state.projects.values()},
            "active_tasks": [{"task_id": t.task_id, "title": t.title, "project_id": t.project_id}
                           for t in self.state.get_active_tasks()],
            "completed_tasks": [{"task_id": t.task_id, "title": t.title, "project_id": t.project_id}
                              for t in self.state.get_completed_tasks()],
            "artifacts": {a.artifact_id: {"title": a.title, "type": a.artifact_type, "project_id": a.project_id}
                         for a in self.state.artifacts.values() if not a.archived_at},
            "current_focus_project": self.state.current_focus_project
        }

    def _handle_low_confidence(self, question: str, intent_analysis: Dict[str, Any]) -> str:
        """Handle questions where intent analysis had low confidence."""
        error = intent_analysis.get('error', 'Unknown parsing error')

        suggestions = [
            "• Try being more specific about what you're looking for",
            "• Use keywords like 'tasks', 'artifacts', 'projects'",
            "• Specify time ranges like 'today', 'this week'",
            "• Mention specific project names if relevant"
        ]

        return f"""I had trouble understanding your question. {error}

Here are some suggestions to help me assist you better:
{chr(10).join(suggestions)}

You can also try commands like:
• 'What tasks are due today?'
• 'Show me artifacts for [project name]'
• 'How many completed tasks do I have?'"""

    # Command implementations that wrap existing state methods
    async def _get_tasks_due_today(self, **kwargs) -> List[Dict[str, Any]]:
        """Get tasks due today."""
        tasks = self.state.get_tasks_due_today()
        return [self._task_to_dict(task) for task in tasks]

    async def _get_overdue_tasks(self, **kwargs) -> List[Dict[str, Any]]:
        """Get overdue tasks."""
        tasks = self.state.get_overdue_tasks()
        return [self._task_to_dict(task) for task in tasks]

    async def _get_active_tasks(self, **kwargs) -> List[Dict[str, Any]]:
        """Get all active tasks."""
        tasks = self.state.get_active_tasks()
        return [self._task_to_dict(task) for task in tasks]

    async def _get_completed_tasks(self, **kwargs) -> List[Dict[str, Any]]:
        """Get all completed tasks."""
        tasks = self.state.get_completed_tasks()
        return [self._task_to_dict(task) for task in tasks]

    async def _get_tasks_by_project(self, project: str, **kwargs) -> List[Dict[str, Any]]:
        """Get tasks for a specific project."""
        # Find project by name or alias
        target_project_id = self._find_project_id(project)
        if not target_project_id:
            return []

        all_tasks = self.state.get_active_tasks() + self.state.get_completed_tasks()
        project_tasks = [t for t in all_tasks if t.project_id == target_project_id]
        return [self._task_to_dict(task) for task in project_tasks]

    async def _get_project_artifacts(self, project: str, **kwargs) -> List[Dict[str, Any]]:
        """Get artifacts for a specific project."""
        target_project_id = self._find_project_id(project)
        if not target_project_id:
            return []

        project_artifacts = [
            a for a in self.state.artifacts.values()
            if a.project_id == target_project_id and not a.archived_at
        ]
        return [self._artifact_to_dict(artifact) for artifact in project_artifacts]

    async def _get_all_artifacts(self, **kwargs) -> List[Dict[str, Any]]:
        """Get all artifacts."""
        artifacts = [a for a in self.state.artifacts.values() if not a.archived_at]
        return [self._artifact_to_dict(artifact) for artifact in artifacts]

    async def _get_artifacts_by_type(self, artifact_type: str, **kwargs) -> List[Dict[str, Any]]:
        """Get artifacts by type."""
        artifacts = [
            a for a in self.state.artifacts.values()
            if a.artifact_type == artifact_type and not a.archived_at
        ]
        return [self._artifact_to_dict(artifact) for artifact in artifacts]

    async def _search_artifacts_by_content(self, keywords: List[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """Search artifacts by content keywords."""
        if not keywords:
            return []

        matching_artifacts = []
        keywords_lower = [k.lower() for k in keywords]

        for artifact in self.state.artifacts.values():
            if artifact.archived_at:
                continue

            searchable_text = f"{artifact.title} {artifact.content or ''}".lower()
            if any(keyword in searchable_text for keyword in keywords_lower):
                matching_artifacts.append(artifact)

        return [self._artifact_to_dict(artifact) for artifact in matching_artifacts]

    async def _get_project_details(self, project: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Get details for a specific project."""
        target_project_id = self._find_project_id(project)
        if not target_project_id or target_project_id not in self.state.projects:
            return None

        project_obj = self.state.projects[target_project_id]

        # Get related tasks and artifacts
        all_tasks = self.state.get_active_tasks() + self.state.get_completed_tasks()
        project_tasks = [t for t in all_tasks if t.project_id == target_project_id]
        project_artifacts = [
            a for a in self.state.artifacts.values()
            if a.project_id == target_project_id and not a.archived_at
        ]

        return {
            "project_id": project_obj.project_id,
            "name": project_obj.name,
            "aliases": project_obj.aliases,
            "created_at": project_obj.created_at.isoformat() if project_obj.created_at else None,
            "task_count": len(project_tasks),
            "active_task_count": len([t for t in project_tasks if t.status != 'completed']),
            "completed_task_count": len([t for t in project_tasks if t.status == 'completed']),
            "artifact_count": len(project_artifacts),
            "focus_count": project_obj.focus_count,
            "last_focused_at": project_obj.last_focused_at.isoformat() if project_obj.last_focused_at else None
        }

    async def _get_all_projects(self, **kwargs) -> List[Dict[str, Any]]:
        """Get all projects with summary info."""
        project_summaries = []

        for project in self.state.projects.values():
            # Get task counts
            all_tasks = self.state.get_active_tasks() + self.state.get_completed_tasks()
            project_tasks = [t for t in all_tasks if t.project_id == project.project_id]

            project_summaries.append({
                "project_id": project.project_id,
                "name": project.name,
                "task_count": len(project_tasks),
                "active_tasks": len([t for t in project_tasks if t.status != 'completed']),
                "completed_tasks": len([t for t in project_tasks if t.status == 'completed'])
            })

        return project_summaries

    async def _count_tasks_by_status(self, **kwargs) -> Dict[str, int]:
        """Count tasks by status."""
        all_tasks = self.state.get_active_tasks() + self.state.get_completed_tasks()
        status_counts = {}

        for task in all_tasks:
            status_counts[task.status] = status_counts.get(task.status, 0) + 1

        return status_counts

    async def _count_artifacts_by_type(self, **kwargs) -> Dict[str, int]:
        """Count artifacts by type."""
        type_counts = {}

        for artifact in self.state.artifacts.values():
            if not artifact.archived_at:
                type_counts[artifact.artifact_type] = type_counts.get(artifact.artifact_type, 0) + 1

        return type_counts

    async def _get_high_priority_tasks(self, **kwargs) -> List[Dict[str, Any]]:
        """Get high priority tasks."""
        high_priority_tasks = [
            t for t in self.state.get_active_tasks()
            if t.priority in ['high', 'urgent']
        ]
        return [self._task_to_dict(task) for task in high_priority_tasks]

    async def _get_tasks_without_due_date(self, **kwargs) -> List[Dict[str, Any]]:
        """Get tasks without due dates."""
        tasks_no_due = [
            t for t in self.state.get_active_tasks()
            if not t.scheduled_for
        ]
        return [self._task_to_dict(task) for task in tasks_no_due]

    async def _search_tasks_by_keywords(self, keywords: List[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """Search tasks by keywords in title/description."""
        if not keywords:
            return []

        matching_tasks = []
        keywords_lower = [k.lower() for k in keywords]

        all_tasks = self.state.get_active_tasks() + self.state.get_completed_tasks()
        for task in all_tasks:
            searchable_text = f"{task.title} {task.description or ''}".lower()
            if any(keyword in searchable_text for keyword in keywords_lower):
                matching_tasks.append(task)

        return [self._task_to_dict(task) for task in matching_tasks]

    async def _get_recent_activity(self, days: int = 7, **kwargs) -> Dict[str, Any]:
        """Get recent activity summary."""
        from datetime import timedelta

        cutoff_date = datetime.now(UTC) - timedelta(days=days)

        # Recent completed tasks
        recent_completed = [
            t for t in self.state.get_completed_tasks()
            if t.completed_at and t.completed_at >= cutoff_date
        ]

        # Recent artifacts
        recent_artifacts = [
            a for a in self.state.artifacts.values()
            if a.created_at >= cutoff_date and not a.archived_at
        ]

        return {
            "period_days": days,
            "completed_tasks": len(recent_completed),
            "new_artifacts": len(recent_artifacts),
            "recent_completed_tasks": [self._task_to_dict(t) for t in recent_completed[-5:]],
            "recent_artifacts": [self._artifact_to_dict(a) for a in recent_artifacts[-5:]]
        }

    def _find_project_id(self, project_name: str) -> Optional[str]:
        """Find project ID by name or alias (case insensitive with fuzzy matching)."""
        project_name_lower = project_name.lower().strip()

        # Remove common suffixes that users might add
        project_name_clean = project_name_lower
        for suffix in [' project', ' proj', ' work']:
            if project_name_clean.endswith(suffix):
                project_name_clean = project_name_clean[:-len(suffix)].strip()

        for project in self.state.projects.values():
            project_name_db = project.name.lower()

            # Check exact name match (original and cleaned)
            if project_name_db == project_name_lower or project_name_db == project_name_clean:
                return project.project_id

            # Check aliases
            for alias in project.aliases:
                alias_lower = alias.lower()
                if alias_lower == project_name_lower or alias_lower == project_name_clean:
                    return project.project_id

            # Check if cleaned project name contains the search term
            if project_name_clean in project_name_db:
                return project.project_id

            # Check if database project name contains the search term (reverse match)
            if project_name_db in project_name_clean:
                return project.project_id

        return None

    def _task_to_dict(self, task) -> Dict[str, Any]:
        """Convert task object to dictionary."""
        return {
            "task_id": task.task_id,
            "title": task.title,
            "description": task.description,
            "status": task.status,
            "priority": task.priority,
            "project_id": task.project_id,
            "project_name": self.state.projects[task.project_id].name if task.project_id and task.project_id in self.state.projects else None,
            "scheduled_for": task.scheduled_for.isoformat() if task.scheduled_for else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "duration_minutes": task.duration_minutes
        }

    def _artifact_to_dict(self, artifact) -> Dict[str, Any]:
        """Convert artifact object to dictionary."""
        return {
            "artifact_id": artifact.artifact_id,
            "title": artifact.title,
            "artifact_type": artifact.artifact_type,
            "content_preview": (artifact.content or '')[:200] if artifact.content else None,
            "url": artifact.url,
            "source": artifact.source,
            "project_id": artifact.project_id,
            "project_name": self.state.projects[artifact.project_id].name if artifact.project_id and artifact.project_id in self.state.projects else None,
            "task_id": artifact.task_id,
            "created_at": artifact.created_at.isoformat() if artifact.created_at else None,
            "created_by": artifact.created_by
        }