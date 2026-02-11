# Sidecar OS

AI-powered productivity assistant with hybrid pattern recognition and intelligent task management.

## Overview

Sidecar OS is an event-sourced productivity system that combines fast pattern matching with sophisticated LLM analysis to intelligently capture, organize, and analyze your work. It adapts to your workflow, learns from context, and provides AI-powered insights while maintaining complete privacy.

**Key Features:**
- 🧠 **Hybrid Intelligence**: Pattern-first analysis (fast) + LLM fallback (smart)
- 🎯 **Contextual Awareness**: Adapts behavior based on project focus and workflow state
- 📊 **AI-Powered Summaries**: Weekly/daily productivity insights with multiple styles
- 🔐 **Privacy-First**: All processing within AWS Bedrock boundaries, local data storage
- 💰 **Cost Transparent**: Real-time LLM usage monitoring with configurable limits
- ⚡ **Event-Sourced**: Complete audit trail, consistent state, reliable operation

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- AWS account with Bedrock access (for LLM features)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd sidecar
   ```

2. **Install dependencies**
   ```bash
   cd sidecar_os
   uv sync
   ```

3. **Set up AWS credentials** (for LLM features)
   ```bash
   # Configure AWS profile for Bedrock access
   aws configure --profile your-profile-name
   ```

4. **Set up convenient `ss` command** (recommended)
   ```bash
   # Create shortcut script (replace with your actual path)
   mkdir -p ~/.local/bin
   cat > ~/.local/bin/ss << 'EOF'
   #!/bin/bash
   cd "/path/to/your/sidecar/sidecar_os" && uv run sidecar "$@"
   EOF

   # Make executable
   chmod +x ~/.local/bin/ss

   # Add to PATH (add to your ~/.zshrc or ~/.bashrc)
   echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.zshrc
   source ~/.zshrc
   ```

5. **Verify installation**
   ```bash
   ss status
   # Should show: "System ready - no data yet"
   ```

### Basic Usage

```bash
# Add tasks and notes with intelligent interpretation
ss add "Project Alpha: review requirements document"   # → Creates project + task
ss add "had productive meeting with design team"       # → Captures as activity log
ss add "need to follow up on client feedback"          # → May request clarification

# Check system status and LLM usage
ss status

# Set project focus for better context awareness
ss focus "Project Alpha"             # Set focus to project
ss add "schedule requirements call"   # → Automatically linked to focused project

# Interactive triage for ambiguous items
ss triage

# Generate AI-powered productivity summaries
ss weekly --style exec      # Brief executive summary
ss weekly --style friendly  # Detailed encouraging insights
ss daily --style exec       # Concise daily metrics
```

## Core Commands

### Task & Project Management
- `ss add "text"` - Intelligent task/note capture with hybrid interpretation
- `ss status` - System overview with LLM usage statistics
- `ss list` - View all tasks and inbox items
- `ss focus "Project Name"` - Set project focus for better context
- `ss focus --clear` - Clear current project focus

### Interactive Workflows
- `ss triage` - Process unclear items with guided clarification
- `ss task <id>` - Convert inbox item to structured task
- `ss done <task>` - Mark task as completed

### AI-Powered Insights
- `ss weekly --style [exec|friendly]` - Weekly productivity summary
- `ss daily --style [exec|friendly]` - Daily activity analysis

### Project Management
- `ss project-add "Project Name"` - Manually create project
- `ss project-list` - View all projects

## How It Works

### Hybrid Intelligence System

Sidecar OS uses a sophisticated two-tier approach:

1. **Pattern Analysis (Fast Path)**: Sub-millisecond pattern matching handles 90%+ of clear inputs
   ```
   "ACME: review contract" → Instant project + task creation
   "Project Beta: schedule meeting" → Immediate processing
   ```

2. **LLM Analysis (Smart Path)**: AWS Bedrock integration for ambiguous cases
   ```
   "had interesting conversation about requirements" → LLM analyzes context and intent
   ```

### Contextual Awareness

The system adapts based on your current workflow state:

- **With Project Focus**: Ambiguous inputs default to focused project
- **Without Focus**: System requests clarification for unclear items
- **Recent Context**: Learns from your recent activity patterns

### Cost Optimization

- **Hybrid Routing**: Only uses expensive LLM when pattern confidence < 60%
- **Real-time Monitoring**: Track costs, tokens, and request counts
- **Configurable Limits**: Default $10/day limit with professional monitoring
- **Cost Transparency**: See actual costs for all LLM operations

## Configuration

### Environment Variables

```bash
# AWS Configuration (for LLM features)
export AWS_PROFILE="your-profile-name"       # Your AWS profile name
export AWS_REGION="us-east-1"                # Bedrock region

# Privacy Settings (optional)
export DISABLE_TELEMETRY="1"                 # Disable any telemetry
export DISABLE_ERROR_REPORTING="1"           # Disable error reporting
```

### LLM Configuration

Configuration is auto-created on first run. You can modify settings as needed:

```yaml
provider: "bedrock"                    # bedrock | mock
model: "claude-opus-4.6"              # LLM model to use
aws_region: "us-east-1"               # AWS region for Bedrock
aws_profile: "your-profile-name"      # AWS profile name
cost_limit_daily: 10.0                # Daily cost limit (USD)
confidence_threshold: 0.6             # LLM fallback threshold
```

## Data & Privacy

### Local Data Storage
- **Event Log**: `sidecar_os/data/events.log` (JSONL format)
- **LLM Usage**: `sidecar_os/data/llm_usage.json` (cost tracking)
- **Configuration**: User-specific settings in local config files

### Privacy Guarantees
- **AWS Bedrock Boundaries**: All LLM processing stays within AWS infrastructure
- **No External Sharing**: Telemetry and error reporting disabled by default
- **Complete Audit Trail**: Event-sourced architecture tracks all actions
- **Local Control**: All data stored locally, you control retention and sharing

## Architecture

Sidecar OS follows event-sourcing principles with immutable event logs and derived state:

```
Raw Input → Hybrid Interpreter → Events → State Projection → User Interface
    ↑             ↓                ↓            ↓              ↓
Pattern         LLM            Event         Project       Status
Analysis      Analysis         Store         State         Display
```

### Key Components
- **Event Store**: Append-only JSONL persistence
- **Hybrid Interpreter**: Pattern matching + LLM integration
- **State Models**: Project, Task, Clarification management
- **LLM Service**: AWS Bedrock integration with cost tracking
- **Summary Generator**: AI-powered productivity insights

## Development

### Running Tests
```bash
cd sidecar_os
uv run pytest
```

### Development Mode
```bash
# Run directly with uv
cd sidecar_os
uv run sidecar add "test item"

# Or use your ss shortcut (if configured)
ss add "test item"
```

## Examples

### Daily Workflow
```bash
# Morning: Set focus and add tasks
ss focus "Q1 Planning"
ss add "review budget projections for Q1"
ss add "schedule team alignment meeting"

# During day: Quick capture
ss add "great insights from client call - they want mobile-first approach"
ss add "need to follow up with legal on contract terms"

# End of day: Review and summarize
ss triage                           # Process unclear items
ss daily --style friendly          # Get encouraging daily summary
```

### Project-Based Work
```bash
# Switch between projects
ss focus "Product Launch"
ss add "finalize marketing copy"
ss add "coordinate with PR team"

ss focus "Client Onboarding"
ss add "prepare demo environment"
ss add "schedule kickoff call"

# Weekly review
ss weekly --style exec             # Executive summary for reporting
```

## Support & Contributing

- **Issues**: Report bugs and feature requests in GitHub Issues
- **Documentation**: See `claude-checkpoint/` directory for development notes
- **Architecture**: Review `claude-checkpoint/architecture-overview.md`

## License

[License to be determined]

---

**Built with:** Event-sourcing principles, AWS Bedrock, Python 3.12, and a focus on user privacy and productivity intelligence.