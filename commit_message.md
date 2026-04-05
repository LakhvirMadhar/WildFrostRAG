# Commit Message


Refactor: Poetry package-mode + drop src. prefix from all imports

- Enable package-mode=true with explicit package includes from src/
- Replace all `from src.X` imports with `from X` across src/ and scripts/
- Move prompt files into src/prompts/ with __init__.py
- Redirect prompt_utils imports from utils to prompts package
- Delete src/types/, redirect imports to models.retrieval
- Update mypy_path to "src" for proper module resolution
- Add mypy configuration to pyproject.toml

Do NOT commit: commit_message.md, .claude/
