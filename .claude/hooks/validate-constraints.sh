#!/usr/bin/env bash
# PostToolUse hook for Edit/Write/MultiEdit.
# When constraints.md is touched, sanity-check structure so a malformed file is caught immediately.
# Conservative: we only catch missing required headings. Stricter table parsing belongs in code that loads the file.

input=$(cat)

if command -v jq >/dev/null 2>&1; then
  file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // .tool_input.path // ""')
else
  file=$(printf '%s' "$input" | grep -oE '"file_path"\s*:\s*"[^"]*"' | head -1 | sed -E 's/.*"file_path"\s*:\s*"(.*)".*/\1/')
fi

case "$file" in
  *constraints.md)
    [ ! -f "$file" ] && exit 0

    missing=()
    grep -q '^## Defaults' "$file" || missing+=("## Defaults")
    grep -q '^## Per-car overrides' "$file" || missing+=("## Per-car overrides")

    if [ ${#missing[@]} -gt 0 ]; then
      echo "constraints.md missing required heading(s) after edit: ${missing[*]}" >&2
      exit 1
    fi
    ;;
esac

exit 0
