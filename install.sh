#!/usr/bin/env bash
# XRDmatch step-by-step installer
# Checks for Anaconda/Miniconda/conda, installs if missing, then creates the app env.

set -euo pipefail

ENV_NAME="xrdmatch"
PYTHON_VERSION="3.11"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- helpers ---

bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
ok()    { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn()  { printf '  \033[33m!\033[0m %s\n' "$*"; }
fail()  { printf '  \033[31m✗\033[0m %s\n' "$*"; }
step()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
pause() {
  echo
  read -r -p "Press Enter to continue (or Ctrl+C to cancel)..." _
}

detect_os() {
  case "$(uname -s)" in
    Darwin) echo "macos" ;;
    Linux)  echo "linux" ;;
    *)      echo "unknown" ;;
  esac
}

detect_arch() {
  case "$(uname -m)" in
    x86_64|amd64) echo "x86_64" ;;
    arm64|aarch64) echo "arm64" ;;
    *) echo "$(uname -m)" ;;
  esac
}

# Try to locate conda even if not on PATH
find_conda() {
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi

  local candidates=(
    "$HOME/anaconda3/bin/conda"
    "$HOME/Anaconda3/bin/conda"
    "$HOME/miniconda3/bin/conda"
    "$HOME/Miniconda3/bin/conda"
    "$HOME/mambaforge/bin/conda"
    "$HOME/miniforge3/bin/conda"
    "/opt/anaconda3/bin/conda"
    "/opt/miniconda3/bin/conda"
    "/usr/local/anaconda3/bin/conda"
    "/usr/local/miniconda3/bin/conda"
  )

  # Apple Silicon / Intel Homebrew paths
  candidates+=(
    "/opt/homebrew/Caskroom/miniconda/base/bin/conda"
    "/usr/local/Caskroom/miniconda/base/bin/conda"
  )

  local c
  for c in "${candidates[@]}"; do
    if [[ -x "$c" ]]; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

init_conda_shell() {
  local conda_bin="$1"
  local conda_base
  conda_base="$("$conda_bin" info --base 2>/dev/null)"
  if [[ -f "$conda_base/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "$conda_base/etc/profile.d/conda.sh"
  else
    export PATH="$(dirname "$conda_bin"):$PATH"
  fi
}

miniconda_installer_url() {
  local os="$1" arch="$2"
  local base="https://repo.anaconda.com/miniconda"
  if [[ "$os" == "macos" && "$arch" == "arm64" ]]; then
    echo "$base/Miniconda3-latest-MacOSX-arm64.sh"
  elif [[ "$os" == "macos" ]]; then
    echo "$base/Miniconda3-latest-MacOSX-x86_64.sh"
  elif [[ "$os" == "linux" && "$arch" == "arm64" ]]; then
    echo "$base/Miniconda3-latest-Linux-aarch64.sh"
  elif [[ "$os" == "linux" ]]; then
    echo "$base/Miniconda3-latest-Linux-x86_64.sh"
  else
    return 1
  fi
}

install_miniconda() {
  local os arch url installer
  os="$(detect_os)"
  arch="$(detect_arch)"
  url="$(miniconda_installer_url "$os" "$arch")" || {
    fail "Unsupported platform: $os / $arch"
    echo "    Download Anaconda manually: https://www.anaconda.com/download"
    return 1
  }

  installer="/tmp/Miniconda3-XRDmatch-install.sh"
  step "Step 1b — Download Miniconda"
  echo "  URL: $url"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail -o "$installer" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$installer" "$url"
  else
    fail "Need curl or wget to download the installer."
    echo "    Install curl, or download Anaconda from:"
    echo "    https://www.anaconda.com/download"
    return 1
  fi

  step "Step 1c — Run Miniconda installer"
  echo "  Installing to: $HOME/miniconda3"
  echo "  (batch mode — no interactive conda prompts)"
  bash "$installer" -b -p "$HOME/miniconda3"
  rm -f "$installer"

  if [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    ok "Miniconda installed at $HOME/miniconda3"
    "$HOME/miniconda3/bin/conda" init bash >/dev/null 2>&1 || true
    "$HOME/miniconda3/bin/conda" init zsh >/dev/null 2>&1 || true
    return 0
  fi
  fail "Miniconda install finished but conda was not found."
  return 1
}

# --- main ---

clear 2>/dev/null || true
bold "=============================================="
bold "  XRDmatch — Step-by-step installer"
bold "=============================================="
echo
echo "This will:"
echo "  1. Check for Anaconda / Miniconda (conda)"
echo "  2. Install Miniconda if conda is missing"
echo "  3. Create a conda env named '${ENV_NAME}' (Python ${PYTHON_VERSION})"
echo "  4. Install XRDmatch dependencies from requirements.txt"
echo "  5. Verify the install"
echo
echo "Project folder: $SCRIPT_DIR"
pause

# ---------- Step 1: conda ----------
step "Step 1 — Looking for Anaconda / conda"

CONDA_BIN=""
if CONDA_BIN="$(find_conda)"; then
  ok "Found conda: $CONDA_BIN"
  init_conda_shell "$CONDA_BIN"
  ok "conda version: $(conda --version 2>/dev/null || echo unknown)"
else
  warn "conda was not found on this computer."
  echo
  echo "  XRDmatch needs Anaconda or Miniconda (both provide 'conda')."
  echo "  Full Anaconda Distribution: https://www.anaconda.com/download"
  echo "  This installer can set up lightweight Miniconda automatically"
  echo "  (same conda tooling; enough for XRDmatch)."
  echo
  read -r -p "  Install Miniconda now? [Y/n] " ans
  ans="${ans:-Y}"
  if [[ "$ans" =~ ^[Yy]$ ]]; then
    install_miniconda
    CONDA_BIN="$(find_conda)" || {
      fail "Still cannot find conda after install."
      exit 1
    }
    init_conda_shell "$CONDA_BIN"
    ok "Using conda: $CONDA_BIN"
  else
    fail "Cannot continue without conda."
    echo "    Install Anaconda from https://www.anaconda.com/download"
    echo "    Then open a new terminal and re-run:  ./install.sh"
    exit 1
  fi
fi

# ---------- Step 2: env ----------
step "Step 2 — Create conda environment '${ENV_NAME}'"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  warn "Environment '${ENV_NAME}' already exists."
  read -r -p "  Recreate it from scratch? [y/N] " ans
  if [[ "${ans:-N}" =~ ^[Yy]$ ]]; then
    conda env remove -y -n "$ENV_NAME"
    conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}"
    ok "Recreated env '${ENV_NAME}'"
  else
    ok "Keeping existing env '${ENV_NAME}'"
  fi
else
  conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}"
  ok "Created env '${ENV_NAME}' with Python ${PYTHON_VERSION}"
fi

# ---------- Step 3: deps ----------
step "Step 3 — Install Python packages"

# Prefer conda for scientific stack when possible, then pip for the rest
echo "  Activating ${ENV_NAME}…"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

ok "Python: $(python --version 2>&1)"
ok "pip:    $(python -m pip --version 2>&1)"

echo "  Upgrading pip…"
python -m pip install --upgrade pip

if [[ -f "$SCRIPT_DIR/requirements.txt" ]]; then
  echo "  Installing from requirements.txt…"
  python -m pip install -r "$SCRIPT_DIR/requirements.txt"
  ok "Dependencies installed"
else
  fail "requirements.txt not found in $SCRIPT_DIR"
  exit 1
fi

# ---------- Step 4: verify ----------
step "Step 4 — Verify installation"

VERIFY_OK=1
python - <<'PY' || VERIFY_OK=0
import importlib
pkgs = [
    ("PyQt5", "PyQt5"),
    ("matplotlib", "matplotlib"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("requests", "requests"),
    ("bs4", "beautifulsoup4"),
    ("lxml", "lxml"),
    ("gemmi", "gemmi"),
    ("pymatgen", "pymatgen"),
]
failed = []
for mod, name in pkgs:
    try:
        importlib.import_module(mod)
        print(f"  ✓ {name}")
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        failed.append(name)
if failed:
    raise SystemExit(1)
PY

if [[ "$VERIFY_OK" -ne 1 ]]; then
  fail "Some packages failed to import."
  exit 1
fi

# Quick import of app entry (may need display — only import modules)
python - <<'PY' || true
import importlib
for mod in ("gui.theme", "gui.session", "matplotlib_config"):
    importlib.import_module(mod)
    print(f"  ✓ {mod}")
print("  ✓ Core XRDmatch modules import OK")
PY

# ---------- Step 5: desktop shortcut ----------
step "Step 5 — Desktop shortcut"

OS_NAME="$(detect_os)"
if [[ "$OS_NAME" == "macos" ]]; then
  chmod +x "$SCRIPT_DIR/scripts/create_macos_desktop_app.sh"
  if "$SCRIPT_DIR/scripts/create_macos_desktop_app.sh" "$SCRIPT_DIR" "$ENV_NAME"; then
    ok "Desktop app: ~/Desktop/XRDmatch.app"
  else
    warn "Could not create macOS Desktop app (you can run scripts/create_macos_desktop_app.sh later)."
  fi
elif [[ "$OS_NAME" == "linux" ]]; then
  DESKTOP_DIR="${XDG_DESKTOP_DIR:-$HOME/Desktop}"
  if [[ -d "$DESKTOP_DIR" ]]; then
    cat > "$DESKTOP_DIR/XRDmatch.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=XRD Phase Matcher
Comment=XRD phase identification
Exec=bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ${ENV_NAME} && cd "${SCRIPT_DIR}" && python main.py'
Path=${SCRIPT_DIR}
Terminal=false
Categories=Science;
EOF
    chmod +x "$DESKTOP_DIR/XRDmatch.desktop"
    ok "Desktop launcher: $DESKTOP_DIR/XRDmatch.desktop"
  else
    warn "No Desktop folder found — skipped Linux .desktop shortcut."
  fi
else
  warn "Desktop shortcuts for this OS: use install.bat on Windows."
fi

# ---------- Done ----------
step "Done"
ok "XRDmatch is ready."
echo
bold "How to run"
if [[ "$OS_NAME" == "macos" ]]; then
  echo "  Double-click XRDmatch on your Desktop (or in Applications)"
  echo "  — or —"
fi
echo "  conda activate ${ENV_NAME}"
echo "  cd \"$SCRIPT_DIR\""
echo "  python main.py"
echo
echo "Optional check:"
echo "  python test_install.py"
echo
warn "If 'conda activate' fails in a new terminal, run:  conda init zsh"
warn "(or conda init bash), then open a new terminal window."
echo
