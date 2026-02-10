"""Summary generation service for analyzing user activity and generating reports."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass

from ..events import EventStore, BaseEvent
from ..state import project_events_to_state
from ..llm import LLMService, LLMConfig


class SummaryStyle(Enum):
    """Summary generation styles."""
    EXECUTIVE = "exec"  # Brief, high-level, action-oriented
    FRIENDLY = "friendly"  # Detailed, conversational, encouraging


class SummaryPeriod(Enum):
    """Summary time periods."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class ActivitySummary:
    """Raw activity data for summary generation."""
    period_start: datetime
    period_end: datetime
    total_events: int
    inbox_items_added: int
    tasks_created: int
    tasks_completed: int
    projects_created: int
    projects_focused: int
    clarifications_requested: int
    clarifications_resolved: int
    most_active_project: Optional[str]
    activity_by_day: Dict[str, int]  # Date -> event count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LLM processing."""
        return {
            "period": f"{self.period_start.strftime('%Y-%m-%d')} to {self.period_end.strftime('%Y-%m-%d')}",
            "total_events": self.total_events,
            "inbox_items_added": self.inbox_items_added,
            "tasks_created": self.tasks_created,
            "tasks_completed": self.tasks_completed,
            "projects_created": self.projects_created,
            "projects_focused": self.projects_focused,
            "clarifications_requested": self.clarifications_requested,
            "clarifications_resolved": self.clarifications_resolved,
            "most_active_project": self.most_active_project or "None",
            "daily_activity": self.activity_by_day
        }


class SummaryGenerator:
    """Generate intelligent summaries of user activity."""

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        """Initialize summary generator.

        Args:
            llm_config: LLM configuration (uses defaults if None)
        """
        self.event_store = EventStore()
        self.llm_service = LLMService(config=llm_config)

    def generate_summary(
        self,
        period: SummaryPeriod,
        style: SummaryStyle = SummaryStyle.EXECUTIVE,
        days_back: Optional[int] = None
    ) -> str:
        """Generate activity summary for specified period.

        Args:
            period: Time period for summary
            style: Summary style (exec or friendly)
            days_back: Override default period with custom days back

        Returns:
            Generated summary text
        """
        # Determine time range
        end_time = datetime.now(timezone.utc)

        if days_back:
            start_time = end_time - timedelta(days=days_back)
        elif period == SummaryPeriod.DAILY:
            start_time = end_time - timedelta(days=1)
        elif period == SummaryPeriod.WEEKLY:
            start_time = end_time - timedelta(days=7)
        else:  # MONTHLY
            start_time = end_time - timedelta(days=30)

        # Analyze activity
        activity = self._analyze_activity(start_time, end_time)

        # Generate summary using LLM
        try:
            summary = asyncio.run(self._generate_llm_summary(activity, style, period))
            return summary
        except Exception as e:
            # Fallback to basic summary if LLM fails
            return self._generate_basic_summary(activity, style, period)

    def _analyze_activity(self, start_time: datetime, end_time: datetime) -> ActivitySummary:
        """Analyze user activity in the specified time range."""
        events = self.event_store.read_all()

        # Filter events by time range
        period_events = [
            event for event in events
            if start_time <= event.timestamp <= end_time
        ]

        # Count activity by type
        inbox_items = sum(1 for e in period_events if e.event_type == "inbox_captured")
        tasks_created = sum(1 for e in period_events if e.event_type == "task_created")
        tasks_completed = sum(1 for e in period_events if e.event_type == "task_completed")
        projects_created = sum(1 for e in period_events if e.event_type == "project_created")
        projects_focused = sum(1 for e in period_events if e.event_type == "project_focused")
        clarifications_requested = sum(1 for e in period_events if e.event_type == "clarification_requested")
        clarifications_resolved = sum(1 for e in period_events if e.event_type == "clarification_resolved")

        # Find most active project
        project_activity = {}
        for event in period_events:
            if hasattr(event, 'payload') and event.payload:
                project_id = event.payload.get('project_id')
                if project_id:
                    project_activity[project_id] = project_activity.get(project_id, 0) + 1

        most_active_project = max(project_activity.keys(), key=lambda k: project_activity[k]) if project_activity else None

        # Activity by day
        activity_by_day = {}
        for event in period_events:
            day_key = event.timestamp.strftime('%Y-%m-%d')
            activity_by_day[day_key] = activity_by_day.get(day_key, 0) + 1

        return ActivitySummary(
            period_start=start_time,
            period_end=end_time,
            total_events=len(period_events),
            inbox_items_added=inbox_items,
            tasks_created=tasks_created,
            tasks_completed=tasks_completed,
            projects_created=projects_created,
            projects_focused=projects_focused,
            clarifications_requested=clarifications_requested,
            clarifications_resolved=clarifications_resolved,
            most_active_project=most_active_project,
            activity_by_day=activity_by_day
        )

    async def _generate_llm_summary(
        self,
        activity: ActivitySummary,
        style: SummaryStyle,
        period: SummaryPeriod
    ) -> str:
        """Generate summary using LLM."""

        # Build context-aware prompt
        context = {
            "activity_data": activity.to_dict(),
            "period_name": period.value,
            "style_preference": style.value
        }

        if style == SummaryStyle.EXECUTIVE:
            system_prompt = """You are an executive assistant generating concise activity summaries.

Create a brief, professional summary focusing on:
- Key accomplishments and progress
- Important metrics and outcomes
- Action items and next steps
- Strategic insights

Keep it under 200 words and use bullet points for clarity."""

            user_prompt = f"""Generate an executive summary for this {period.value} activity:

Activity Data:
{context['activity_data']}

Focus on productivity metrics, completed work, and strategic insights. Be concise and action-oriented."""

        else:  # FRIENDLY style
            system_prompt = """You are a supportive productivity coach generating encouraging activity summaries.

Create a warm, detailed summary that:
- Celebrates progress and achievements
- Provides encouraging insights about productivity patterns
- Offers gentle suggestions for improvement
- Uses a friendly, conversational tone

Make it personalized and motivating, around 250-300 words."""

            user_prompt = f"""Generate a friendly, encouraging summary for this {period.value} activity:

Activity Data:
{context['activity_data']}

Celebrate the user's progress, highlight positive patterns, and provide supportive insights. Be warm and encouraging."""

        try:
            response = await self.llm_service.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=400,
                temperature=0.7
            )
            return response.text.strip()

        except Exception as e:
            raise Exception(f"LLM summary generation failed: {e}")

    def _generate_basic_summary(
        self,
        activity: ActivitySummary,
        style: SummaryStyle,
        period: SummaryPeriod
    ) -> str:
        """Generate basic summary without LLM (fallback)."""

        period_str = f"{activity.period_start.strftime('%B %d')} - {activity.period_end.strftime('%B %d, %Y')}"

        if style == SummaryStyle.EXECUTIVE:
            return f"""# {period.value.title()} Activity Summary
**Period:** {period_str}

## Key Metrics
• **Total Activity:** {activity.total_events} events
• **Items Captured:** {activity.inbox_items_added} inbox items
• **Tasks Created:** {activity.tasks_created}
• **Tasks Completed:** {activity.tasks_completed}
• **Projects:** {activity.projects_created} created, {activity.projects_focused} focused
• **Most Active Project:** {activity.most_active_project or 'N/A'}

## Status
Progress ratio: {activity.tasks_completed}/{activity.tasks_created} tasks completed
Clarifications: {activity.clarifications_resolved}/{activity.clarifications_requested} resolved
"""

        else:  # FRIENDLY
            completion_rate = (activity.tasks_completed / max(activity.tasks_created, 1)) * 100

            return f"""# Your {period.value.title()} Progress Summary 🎯

**Period:** {period_str}

Great work this {period.value}! Here's what you accomplished:

## 📊 Your Activity
You've been quite productive with **{activity.total_events} total activities**! You captured {activity.inbox_items_added} new items and created {activity.tasks_created} tasks.

## ✅ Completions
You completed {activity.tasks_completed} tasks ({completion_rate:.1f}% completion rate) - that's excellent progress!

## 🎯 Project Focus
{f"You spent most time on **{activity.most_active_project}**" if activity.most_active_project else "You worked across multiple projects"}
Created {activity.projects_created} new projects and switched focus {activity.projects_focused} times.

## 🤔 Problem Solving
You handled {activity.clarifications_requested} clarifications, resolving {activity.clarifications_resolved} of them.

Keep up the momentum! 🚀"""