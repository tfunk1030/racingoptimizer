#!/usr/bin/env bash
# PreToolUse hook for Bash matcher.
# Block destructive shell commands targeting ibtfiles/ or aero-maps/.
# Reads the tool-call JSON from stdin; exits 2 to block, 0 to allow.

input=$(cat)

# Try jq, fall back to a minimal grep if jq isn't installed.
if command -v jq >/dev/null 2>&1; then
  cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // ""')
else
  cmd=$(printf '%s' "$input" | grep -oE '"command"\s*:\s*"[^"]*"' | head -1 | sed -E 's/.*"command"\s*:\s*"(.*)".*/\1/')
fi

# Targets a protected directory?
if printf '%s' "$cmd" | grep -Eq '(ibtfiles|aero-maps)'; then
  # And uses a destructive verb / output redirect?
  if printf '%s' "$cmd" | grep -Eq '(\brm\b|\bmv\b|\btruncate\b|\bshred\b|\bdd\b|>[[:space:]]*[^|]|>>[[:space:]]*[^|])'; then
    cat >&2 <<MSG
BLOCKED: destructive operation against ibtfiles/ or aero-maps/.
These directories hold irreplaceable training data (.ibt sessions, parsed aero maps).
Command was:
  $cmd
If this is intentional, run it manually outside Claude Code.
MSG
    exit 2
  fi
fi

exit 0
