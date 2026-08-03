#!/usr/bin/env bash
# Create XRDmatch.app on the macOS Desktop (and Applications if possible).
# Usage: ./scripts/create_macos_desktop_app.sh [project_root] [env_name]

set -euo pipefail

PROJECT_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${2:-xrdmatch}"
APP_NAME="XRDmatch"
DESKTOP="${HOME}/Desktop"
APP_PATH="${DESKTOP}/${APP_NAME}.app"
ICNS="${PROJECT_ROOT}/assets/XRDmatch.icns"

find_conda_sh() {
  if command -v conda >/dev/null 2>&1; then
    local base
    base="$(conda info --base 2>/dev/null)"
    if [[ -f "${base}/etc/profile.d/conda.sh" ]]; then
      echo "${base}/etc/profile.d/conda.sh"
      return 0
    fi
  fi
  local c
  for c in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/Miniconda3/etc/profile.d/conda.sh" \
    "$HOME/Anaconda3/etc/profile.d/conda.sh" \
    "$HOME/mambaforge/etc/profile.d/conda.sh" \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
  do
    if [[ -f "$c" ]]; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

CONDA_SH="$(find_conda_sh)" || {
  echo "Could not find conda.sh — activate conda and retry."
  exit 1
}

rm -rf "${APP_PATH}" 2>/dev/null || true
mkdir -p "${APP_PATH}/Contents/MacOS"
mkdir -p "${APP_PATH}/Contents/Resources"

if [[ -f "$ICNS" ]]; then
  cp "$ICNS" "${APP_PATH}/Contents/Resources/AppIcon.icns"
fi

cat > "${APP_PATH}/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>
  <string>${APP_NAME}</string>
  <key>CFBundleDisplayName</key>
  <string>XRD Phase Matcher</string>
  <key>CFBundleIdentifier</key>
  <string>org.xrdtools.xrdmatch</string>
  <key>CFBundleVersion</key>
  <string>1.0.0</string>
  <key>CFBundleShortVersionString</key>
  <string>1.0</string>
  <key>CFBundleExecutable</key>
  <string>${APP_NAME}</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleIconFile</key>
  <string>AppIcon</string>
  <key>LSMinimumSystemVersion</key>
  <string>11.0</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
EOF

cat > "${APP_PATH}/Contents/MacOS/${APP_NAME}" <<EOF
#!/bin/bash
# XRDmatch desktop launcher
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:\${PATH}"

PROJECT_ROOT=$(printf '%q' "${PROJECT_ROOT}")
ENV_NAME=$(printf '%q' "${ENV_NAME}")
CONDA_SH=$(printf '%q' "${CONDA_SH}")

cd "\$PROJECT_ROOT" || {
  /usr/bin/osascript -e 'display alert "XRDmatch" message "Project folder not found. Re-run install.sh." as critical'
  exit 1
}

# shellcheck source=/dev/null
source "\$CONDA_SH"
conda activate "\$ENV_NAME" || {
  /usr/bin/osascript -e 'display alert "XRDmatch" message "Could not activate conda env. Re-run install.sh." as critical'
  exit 1
}

exec python "\$PROJECT_ROOT/main.py"
EOF

chmod +x "${APP_PATH}/Contents/MacOS/${APP_NAME}"

# Clear quarantine so double-click works after download/copy
xattr -dr com.apple.quarantine "${APP_PATH}" 2>/dev/null || true

# Also place in Applications if writable
if [[ -d "/Applications" && -w "/Applications" ]]; then
  rm -rf "/Applications/${APP_NAME}.app"
  cp -R "${APP_PATH}" "/Applications/${APP_NAME}.app"
  xattr -dr com.apple.quarantine "/Applications/${APP_NAME}.app" 2>/dev/null || true
  echo "Created: /Applications/${APP_NAME}.app"
fi

echo "Created: ${APP_PATH}"
echo "Double-click XRDmatch on your Desktop to launch."
