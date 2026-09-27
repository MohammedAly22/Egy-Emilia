#!/usr/bin/env bash
# Install the bgutil PO-token provider so yt-dlp can download from YouTube on a
# cloud IP (RunPod) WITHOUT cookies. Run once, inside the conda env:
#
#   conda activate egy
#   bash scripts/setup_pot_provider.sh
#
# Installs: Node.js >= 22 (conda-forge), the yt-dlp plugin (pip), and the token
# server built from the SAME version tag into .tools/ (git-ignored).
# The download stage starts/stops the server by itself — nothing else to run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS="$REPO_ROOT/.tools"
SERVER_DIR="$TOOLS/bgutil-ytdlp-pot-provider"

echo "==> Node.js >= 22"
if ! command -v node >/dev/null || [ "$(node -p 'process.versions.node.split(".")[0]')" -lt 22 ]; then
    conda install -c conda-forge "nodejs>=22" -y
fi
node --version

echo "==> yt-dlp + PO-token plugin"
pip install -U "yt-dlp[default]" bgutil-ytdlp-pot-provider
VERSION="$(pip show bgutil-ytdlp-pot-provider | awk '/^Version:/{print $2}')"
echo "plugin version: $VERSION"

echo "==> token server $VERSION (must match the plugin version)"
mkdir -p "$TOOLS"
rm -rf "$SERVER_DIR"
git clone --depth 1 --single-branch --branch "$VERSION" \
    https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git "$SERVER_DIR"
cd "$SERVER_DIR/server"
npm ci
npx tsc

echo
echo "==> done. server: $SERVER_DIR/server/build/main.js"
echo "    python download.py   (or run_pipeline.py) will start it automatically."
