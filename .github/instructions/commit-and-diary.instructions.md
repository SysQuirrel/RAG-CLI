---
description: "Use when making code or configuration changes in this project. Enforces post-change git commit with a clear message and update of project diary markdown with the same clear summary."
name: "Commit And Diary Rule"
applyTo: "**"
---
# Commit And Diary Rule

- After completing implementation work, always run a git commit.
- Commit messages must be clear, specific, and describe what was changed.
- Use the imperative style in commit messages when possible.
- Do not use vague commit messages like "update" or "fix stuff".

- For every commit, update the project diary markdown file at `FEATURES_AND_PIPELINE.md`.
- Add a concise change note that mirrors the commit intent.
- Include date, problem, what changed, and outcome in the diary entry.
- Keep diary wording consistent with the commit message so history stays traceable.

- If there are no source changes to commit, explicitly state that no commit was needed.
- If commit cannot be made (for example user blocked it), explain the blocker and suggest the exact commit command.
