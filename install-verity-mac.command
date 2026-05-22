#!/usr/bin/env bash
# install-verity-mac.command
# One-click Verity setup for Apple Silicon.
#
# Double-click this file in Finder. (First time: right-click, Open,
# confirm the security prompt.)
#
# Default install location is ~/Verity. Override with INSTALL_PATH:
#   INSTALL_PATH=/wherever ./install-verity-mac.command
#
# To install from a fork:
#   REPO_URL=https://github.com/<you>/Verity.git ./install-verity-mac.command
# The URL must match the github.com pattern check below.

# 2026-05-12: hardened error handling.
#   set -e: exit on first failed command.
#   set -u: treat unset variables as errors (no silent "" expansion).
#   set -o pipefail: a pipeline's exit status is the last non-zero
#                    command, so `npm install | tail` no longer
#                    swallows npm failures.
set -euo pipefail

INSTALL_PATH="${INSTALL_PATH:-$HOME/Verity}"
REPO_URL="${REPO_URL:-https://github.com/johnnyryan/Verity.git}"

# 2026-05-12: validate REPO_URL pattern. `npm install` runs arbitrary
# post-install scripts from whatever code we clone; accepting any URL
# would turn this installer into a remote-code-execution vector. Allow
# only https github.com URLs by default.
if [[ ! "$REPO_URL" =~ ^https://github\.com/[^/[:space:]]+/[^/[:space:]]+\.git$ ]]; then
    echo "[ERROR] REPO_URL must match https://github.com/<owner>/<repo>.git"
    echo "        got: $REPO_URL"
    exit 1
fi

# When double-clicked from Finder, the script's working dir is the
# script's directory. cd to $HOME so install lands sensibly. (The
# old extra `cd "$(dirname "$0")"` was a no-op and has been dropped.)
cd "$HOME"

echo ""
echo "Verity setup"
echo "============"
echo ""

# 0. Apple Silicon check (warn, do not block; this script also runs on Intel
#    macs, just sub-optimally).
arch="$(uname -m)"
if [ "$arch" != "arm64" ]; then
    echo "[WARN] This script targets Apple Silicon (arm64). Detected: $arch"
    echo "       Continuing; expect slower model load times."
    echo ""
fi

# 1. Prerequisite checks.
echo "Checking prerequisites..."
missing=0

require() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "  Missing: $1 ($2)"
        missing=1
    fi
}

require node   "https://nodejs.org or 'brew install node'"
require git    "Xcode CLT or 'brew install git'"
require ollama "https://ollama.com or 'brew install ollama'"

if [ "$missing" -ne 0 ]; then
    echo ""
    echo "Install the missing items and run this script again."
    exit 1
fi
echo "  [OK] Node.js, Git, Ollama present"

# LM Studio is a UI app. Check the standard /Applications path.
if [ -d "/Applications/LM Studio.app" ]; then
    echo "  [OK] LM Studio.app found in /Applications"
else
    echo ""
    echo "[WARN] LM Studio.app not in /Applications."
    echo "       Install LM Studio 0.3.x or newer from https://lmstudio.ai"
    echo "       Continuing; you can install LM Studio after this script."
fi

# 2. Clone or update the Verity repo.
echo ""
echo "Fetching Verity source..."
if [ -d "$INSTALL_PATH" ]; then
    echo "  [INFO] $INSTALL_PATH already exists; pulling latest"
    ( cd "$INSTALL_PATH" && git pull ) || echo "  [WARN] git pull failed; continuing with existing copy"
else
    echo "  [INFO] Cloning into $INSTALL_PATH"
    git clone "$REPO_URL" "$INSTALL_PATH"
fi

# 3. Install dependencies and build.
echo ""
echo "Installing dependencies (npm install)..."
( cd "$INSTALL_PATH/project" && npm install )
echo ""
echo "Building (npm run build)..."
( cd "$INSTALL_PATH/project" && npm run build )

# 4. Pull the two critic models via Ollama.
echo ""
echo "Pulling critic models (this may take a few minutes)..."
ollama pull granite3.2:8b
ollama pull granite3.2:2b

# 5. Apple Silicon has one unified memory pool. The dual-GPU /second path
#    is meaningless here; drop it.
ENV_FILE="$INSTALL_PATH/.verity-env"
cat > "$ENV_FILE" <<EOF
# Apple Silicon: single unified memory pool. Disable dual-GPU /second.
CONSULT_DUAL=0
EOF
echo ""
echo "  [INFO] Wrote $ENV_FILE (CONSULT_DUAL=0 for unified memory)."

# 6. Done. Print the MCP config and next steps.
cat <<EOF

Done.

Next steps:

  1. Open LM Studio. Settings -> Model Context Protocol. Paste:

       {
         "mcpServers": {
           "verity": {
             "url": "http://localhost:8090/mcp",
             "timeout": 240000,
             "retries": 1
           }
         }
       }

  2. Load a chat model in LM Studio. That is your worker.

  3. Start Verity. From a Terminal prompt:

       cd $INSTALL_PATH/project
       node dist/index.js

     (Or write a small launcher script that runs this for you.)

  4. After any answer in LM Studio, type '/verify'.

EOF
