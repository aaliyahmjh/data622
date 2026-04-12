#!/bin/bash
# Quick GitHub update script for data622
# Usage: bash scripts/update.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "📂 Updating: $PROJECT_ROOT"
echo "---"

# Check if git repo
if ! git -C "$PROJECT_ROOT" rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Not a Git repository!"
    exit 1
fi

# Get current branch
BRANCH=$(git -C "$PROJECT_ROOT" branch --show-current)
echo "📌 Branch: $BRANCH"

# Fetch
echo "⬇️  Fetching from GitHub..."
git -C "$PROJECT_ROOT" fetch origin

# Stash if needed
if ! git -C "$PROJECT_ROOT" diff --quiet; then
    echo "⚠️  Stashing local changes..."
    git -C "$PROJECT_ROOT" stash
    STASHED=true
fi

# Pull
echo "🔄 Pulling from origin/$BRANCH..."
git -C "$PROJECT_ROOT" pull origin "$BRANCH"

# Pop stash
if [ "$STASHED" = true ]; then
    echo "📦 Restoring stashed changes..."
    git -C "$PROJECT_ROOT" stash pop || echo "⚠️  Review stash conflicts manually"
fi

# Show latest
echo ""
echo "📜 Latest commits:"
git -C "$PROJECT_ROOT" log --oneline -n 5

echo ""
echo "✨ Update complete!"
