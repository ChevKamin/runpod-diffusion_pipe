#!/usr/bin/env bash
set -euo pipefail

# Set repo URL and branch
REPO_URL="https://github.com/ChevKamin/runpod-diffusion_pipe.git"
BRANCH="${BRANCH:-main}"
DEST="/tmp/runpod-diffusion_pipe"

# Clone if not already present
if [ ! -d "$DEST/.git" ]; then
    echo ">>> Cloning repo..."
    git clone --branch "$BRANCH" "$REPO_URL" "$DEST"
else
    echo ">>> Repo already present, pulling latest changes..."
    cd "$DEST"
    git reset --hard
    git clean -fd
    git pull origin "$BRANCH"
fi

# Continue with the rest of your startup logic
cd "$DEST/src"
bash start.sh
