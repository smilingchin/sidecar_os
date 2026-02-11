# Sidecar OS — Claude Code Instructions

## Project Overview
**Read REQUIREMENTS.md first** for complete project vision, design principles, and MVP scope.

## Core Principles for Claude Code
1. **Event-sourced architecture**: Events are append-only (JSONL), state is derived by replaying events
2. **Never edit past events**: append corrections/reschedules instead
3. **Privacy-first**: keep real user data private (`data/` is gitignored), commit only synthetic examples in `examples/`
4. **Draft-first**: generate outputs as files under `outputs/`, never auto-send
5. **Minimal clarification**: ask at most 1–3 questions when ambiguity matters, otherwise capture as inbox event
6. **Design docs stay local**: REQUIREMENTS.md, CLAUDE.md and other specification documents are gitignored and never committed to GitHub

## Implementation Guidelines

### Tech Stack
- Python 3.12, uv for dependency management
- typer (CLI framework), pydantic (data validation), rich (terminal output)
- **IMPORTANT**: Always use UV commands: `uv add`, `uv sync`, `uv run`, NOT pip

### Architecture (see REQUIREMENTS.md for detailed mental model)
- `sidecar_core/events`: event schemas + append/read functionality
- `sidecar_core/state`: projection logic (events → derived state)
- `sidecar_core/router`: input parsing + clarification gating
- `sidecar_core/skills`: artifact generation (weekly/daily summaries)
- `sidecar_core/artifacts`: artifact models + renderers
- `cli`: CLI command wiring

### MVP Commands (Priority Order)
1. **Slice 1**: `sidecar add "<text>"`, `sidecar status`
2. **Slice 2**: `sidecar done <task_id|query>`, `sidecar list`
3. **Slice 3**: `sidecar weekly --style exec|friendly`
4. **Slice 4**: `sidecar doc add`, `sidecar link task`, `sidecar triage`

### Code Quality Requirements
- All artifacts saved as Markdown with provenance (event IDs)
- Unit tests required for: event store, projection logic, weekly summary generation
- Follow event-sourcing patterns: immutable events, derived state, clear separation

## Development Workflow

### Test-as-You-Go Methodology
**IMPORTANT**: Build incrementally with this cycle for each feature:

1. **Build**: Implement one specific functionality (e.g., event store, add command, status view)
2. **Test**: Write and run unit tests for the functionality using `uv run pytest`
3. **Commit**: Commit the working, tested code with clear commit message
4. **Push**: Push to remote repository
5. **Next**: Move to the next functionality

**Never build multiple features simultaneously**. Each commit should represent one working, tested piece of functionality.

### Git Practices
- **Always check git status before git add**: Run `git status` to review what files will be staged and ensure no unintended files are included
- Commit messages should be clear and describe the specific functionality added
- **Never include Co-Authored-By lines**: Do not add "Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>" or similar attribution lines in commit messages
- Each commit should pass all tests
- Push frequently to maintain backup and enable collaboration
- Only commit code files and synthetic examples, never design documents or real user data

### Session Checkpoints
**IMPORTANT**: Create development checkpoints periodically to maintain context across sessions:

1. **After Major Milestones**: Create checkpoint when significant features are completed
2. **End of Complex Sessions**: Summarize progress, decisions, and next steps
3. **Before Architecture Changes**: Document current state before major refactoring

**Checkpoint Process:**
1. **Update** `claude-checkpoint/current-state.md` with latest architecture and status
2. **Create** `claude-checkpoint/session-YYYY-MM-DD.md` with session summary
3. **Include**: Key implementations, testing results, discoveries, next steps
4. **Focus**: Technical decisions, user experience insights, architectural learnings

**Checkpoint Structure:**
```
claude-checkpoint/
├── README.md              # Checkpoint system explanation
├── current-state.md       # Always up-to-date project status
├── session-YYYY-MM-DD.md  # Individual session summaries
└── architecture-overview.md # High-level system design
```

This enables resuming work efficiently and maintains development history outside of git commits.
