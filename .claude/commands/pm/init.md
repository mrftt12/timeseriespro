# PM Init Command

## Purpose
Initialize the Claude Code PM system for project management with GitHub integration.

## Usage
```
/pm:init
```

## What This Command Does
1. Checks for GitHub CLI installation and authentication
2. Installs gh-sub-issue extension if needed
3. Creates required directory structure in `.claude/`
4. Updates .gitignore with PM system entries
5. Sets up CLAUDE.md with project context
6. Prepares the system for PRD creation and epic management

## Directory Structure Created
```
.claude/
├── CLAUDE.md          # Always-on instructions
├── agents/            # Task-oriented agents
├── commands/          # Command definitions
│   ├── context/       # Context management
│   ├── pm/            # PM commands
│   └── testing/       # Testing commands
├── context/           # Project-wide context
├── epics/             # Epic workspace (add to .gitignore)
├── prds/              # Product Requirements Documents
├── rules/             # Rule files
└── scripts/           # Script files
```

## Prerequisites
- GitHub CLI (gh) must be installed
- GitHub authentication must be configured
- Repository must have GitHub remote configured

## Next Steps
After running `/pm:init`:
1. Run `/context:create` to prime the system
2. Create your first PRD with `/pm:prd-new feature-name`
3. Start the structured development workflow