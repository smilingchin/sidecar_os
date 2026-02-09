# Sidecar OS — Claude Code Instructions

## Mission
Build a CLI-first, event-sourced productivity system.

## Core principles
- Events are append-only (JSONL).
- State is derived by replaying events.
- Artifacts (weekly emails, briefs, visualizations) are generated from events + state.

## Non-negotiables
1. Never edit past events; append corrections/reschedules instead.
2. Keep real user data private:
   - `data/` is gitignored
   - commit only synthetic examples in `examples/`
3. Draft-first:
   - generate outputs as files under `outputs/`
   - never auto-send
4. Minimal clarification:
   - ask at most 1–3 questions when ambiguity matters
   - otherwise capture as inbox event

## MVP scope
CLI commands:
- sidecar add "<text>"
- sidecar status
- sidecar weekly --style exec|friendly
- sidecar doc add --title ... --url ... --project ...
- sidecar link task <task_id> --doc <doc_id|url>
- sidecar done <task_id|query>
- sidecar triage

## Tech stack
Python 3.12, uv, typer, pydantic, rich.

## Architecture
- sidecar_core/events: event schemas + append/read
- sidecar_core/state: projection logic
- sidecar_core/router: parsing + clarification gating
- sidecar_core/skills: weekly/daily generation
- sidecar_core/artifacts: artifact models + renderers
- cli: CLI wiring

## Output
Artifacts saved as Markdown with provenance (event IDs).

## Testing
Add unit tests for event store, projection, and weekly summary.
