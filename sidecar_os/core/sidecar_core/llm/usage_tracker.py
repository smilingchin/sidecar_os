"""LLM usage tracking with persistent statistics."""

import json
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, asdict
import os

@dataclass
class LLMUsageStats:
    """LLM usage statistics for a single day."""
    date: str
    request_count: int = 0
    total_cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    provider: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMUsageStats':
        """Create from dictionary."""
        return cls(**data)


class LLMUsageTracker:
    """Persistent LLM usage tracking."""

    def __init__(self, data_dir: Path = None):
        """Initialize usage tracker.

        Args:
            data_dir: Directory to store usage data (defaults to ./data)
        """
        if data_dir is None:
            data_dir = Path.cwd() / "data"

        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)
        self.usage_file = self.data_dir / "llm_usage.json"

        # Load existing data
        self._usage_data: Dict[str, LLMUsageStats] = self._load_usage_data()

    def _load_usage_data(self) -> Dict[str, LLMUsageStats]:
        """Load usage data from file."""
        if not self.usage_file.exists():
            return {}

        try:
            with open(self.usage_file, 'r') as f:
                data = json.load(f)

            # Convert to LLMUsageStats objects
            return {
                date_str: LLMUsageStats.from_dict(stats)
                for date_str, stats in data.items()
            }
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Warning: Failed to load LLM usage data: {e}")
            return {}

    def _save_usage_data(self) -> None:
        """Save usage data to file."""
        try:
            data = {
                date_str: stats.to_dict()
                for date_str, stats in self._usage_data.items()
            }

            with open(self.usage_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save LLM usage data: {e}")

    def get_today_stats(self) -> LLMUsageStats:
        """Get today's usage statistics."""
        today_str = date.today().isoformat()

        if today_str not in self._usage_data:
            self._usage_data[today_str] = LLMUsageStats(date=today_str)

        return self._usage_data[today_str]

    def track_request(self, cost: float, input_tokens: int = 0, output_tokens: int = 0, provider: str = "unknown") -> None:
        """Track a single LLM request.

        Args:
            cost: Cost of the request in USD
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            provider: LLM provider name
        """
        today_stats = self.get_today_stats()

        today_stats.request_count += 1
        today_stats.total_cost += cost
        today_stats.input_tokens += input_tokens
        today_stats.output_tokens += output_tokens
        today_stats.provider = provider

        self._save_usage_data()

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary for display."""
        today_stats = self.get_today_stats()

        return {
            "daily_requests": today_stats.request_count,
            "daily_cost": today_stats.total_cost,
            "daily_input_tokens": today_stats.input_tokens,
            "daily_output_tokens": today_stats.output_tokens,
            "provider": today_stats.provider,
            "date": today_stats.date
        }

    def reset_daily_stats(self) -> None:
        """Reset today's statistics."""
        today_str = date.today().isoformat()
        if today_str in self._usage_data:
            del self._usage_data[today_str]
            self._save_usage_data()


# Global singleton tracker
_global_tracker: LLMUsageTracker = None

def get_usage_tracker() -> LLMUsageTracker:
    """Get the global usage tracker singleton."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = LLMUsageTracker()
    return _global_tracker