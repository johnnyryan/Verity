#!/usr/bin/env bash
# install-verity-mac.command
# One-click Verity setup for Apple Silicon.
#
# Double-click this file in Finder. (First time: right-click, Open,
# confirm the security prompt.)
#
# Default install location is ~/Verity. Override with INSTALL_PATH:
#   INSTALL_PATH=/wherever ./install-verity-mac.command

set -e

INSTALL_PATH="${INSTALL_PATH:-$HOME/Verity}"
REPO_URL="${REPO_URL:-https://github.com/johnnyryan/Verity.git}"

# When double-clicked from Finder, cd to the script's directory so any
# relative paths work; cd back to HOME so install lands sensibly.
cd "$(dirname "$0")" >/dev/null 2>&1 || true
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
