#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — provision a Python venv for vllm-gaudi profiling on a Gaudi dev
# machine where the full npu-stack sources + builds already exist (typically
# assembled by `build_and_insmod_habanalabs`).
#
# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCY LAYERS (deepest → highest)
#
#   Layer 0 — Specs / headers (compile-time only, no runtime install):
#     1.  specs_external            — HCL/SoC register and IP specs headers
#
#   Layer 1 — Core runtime / synapse stack (loaded via LD_LIBRARY_PATH):
#     2.  synapse                   → synapse_release_build/            libSynapse.so
#     3.  synapse_utils             → synapse_utils_release_build/      helpers
#     4.  synapse_profiler          → synapse_profiler_release_build/   hl-prof-config, trace lib
#     5.  hcl                       → hcl_release_build/                collectives runtime
#     6.  scal                      → scal_release_build/               scheduler abstraction
#     7.  aeon                      → aeon_release_build/               data loader backend
#
#   Layer 2 — TPC toolchain & kernels (compiled then loaded at graph compile):
#     8.  tpc_llvm10                → tpc_llvm_release_build/           TPC compiler (tpc-clang)  — TPC_COMPILER_PATH
#     9.  tpc_kernels               → tpc_kernels_release_build/        libtpc_kernels.so         — GC_KERNEL_PATH
#     10. tpc_scalar_kernels        → tpc_scalar_kernels_release_build/ scalar kernels            — PYTHONPATH
#     11. tpc_fuser                 → tpc_fuser_release_build/          graph fuser + py bindings — PYTHONPATH + LD
#     12. complex_guid_lib          → complex_guid_lib_release_build/   complex ops (softmax etc) — HABANA_PLUGINS_LIB_PATH
#     13. engines_fw                → engines_fw_release_build/         engines FW + headers
#
#   Layer 3 — PyTorch integration:
#     14. pytorch-fork              → pytorch_fork_release_build/pkgs/  torch wheel (+hpu patches) — pip install
#     15. pytorch-integration       → pytorch_modules_release_build/    habana_frameworks bridge  — pip install -e python_packages/
#
#   Layer 4 — Inference engine:
#     16. vllm                      (upstream vLLM, editable)           — pip install -e (VLLM_TARGET_DEVICE=empty)
#     17. vllm-gaudi                (HPU platform plugin)               — pip install -e
#
#   Layer 5 — Workload-specific:
#     18. torchaudio                (CPU wheel, required by Qwen3.5)    — pip install --no-deps
#
#   Layer 6 — Tooling (not on runtime import path):
#     19. automation/habana_scripts (habana_env — wires env vars)       — source
#     20. habana_py                 (HW-level debugging helpers)        — on PYTHONPATH
#
# ─────────────────────────────────────────────────────────────────────────────
# Why this is not a simple `pip install`:
#
# habana_env (sourced below) wires all those builds into runtime env vars.
# The habana_torch_plugin wheel in pytorch_modules_release_build/pkgs/ has the
# WRONG .so layout for runtime (puts C extensions under fork_pybind/); the
# canonical install is `pip install -e $PYTORCH_MODULES_ROOT_PATH/python_packages`
# which triggers a custom InstallCMakeLibs step that lays the .so files flat
# in habana_frameworks/torch/lib/.
#
# Usage:
#   ./setup_env.sh                 # idempotent; installs only missing pieces
#   FORCE_REINSTALL=1 ./setup_env.sh   # wipe and rebuild the venv
#
# Env overrides:
#   VENV_DIR        (default $HOME/.venvs/vllm-gaudi)
#   NPU_STACK_ROOT  (default $SCRIPT_DIR/../..)
#   BUILDS_ROOT     (default $HOME/builds)
#   PY              (default python3.12)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
LOG="$SCRIPT_DIR/setup_env.log"
: > "$LOG"
echo "(also logging to $LOG)"

# Enable colors only when stdout is a TTY (and NO_COLOR not set). This way
# the on-disk log stays ANSI-free automatically and the terminal stays pretty.
if [[ -t 1 && "${NO_COLOR:-}" != "1" ]]; then
    C_STEP=$'\033[1;34m'; C_OK=$'\033[1;32m'; C_WARN=$'\033[1;33m'
    C_ERR=$'\033[1;31m';  C_RST=$'\033[0m'
else
    C_STEP= C_OK= C_WARN= C_ERR= C_RST=
fi

# Each helper prints the colored line to stdout and a plain copy to $LOG so
# `cat setup_env.log` is readable.
step() { printf '\n%s==> %s%s\n' "$C_STEP" "$*" "$C_RST"; printf '==> %s\n' "$*" >> "$LOG"; }
ok()   { printf '   %sOK%s %s\n'  "$C_OK"   "$C_RST" "$*"; printf '   OK %s\n'   "$*" >> "$LOG"; }
warn() { printf '   %sWARN%s %s\n' "$C_WARN" "$C_RST" "$*"; printf '   WARN %s\n' "$*" >> "$LOG"; }
die()  { printf '   %sERR%s %s\n'  "$C_ERR"  "$C_RST" "$*" >&2; printf '   ERR %s\n'  "$*" >> "$LOG"; exit 1; }

echo "=== setup_env.sh started at $(date) ===" | tee -a "$LOG"

# ---------- 0. Resolve paths ---------------------------------------------
NPU_STACK_ROOT="${NPU_STACK_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/vllm-gaudi}"
BUILDS_ROOT="${BUILDS_ROOT:-$HOME/builds}"
PY="${PY:-python3.12}"
FORCE_REINSTALL="${FORCE_REINSTALL:-0}"

echo "SCRIPT_DIR=$SCRIPT_DIR" | tee -a "$LOG"
echo "NPU_STACK_ROOT=$NPU_STACK_ROOT" | tee -a "$LOG"
echo "VENV_DIR=$VENV_DIR" | tee -a "$LOG"
echo "BUILDS_ROOT=$BUILDS_ROOT" | tee -a "$LOG"
echo "PY=$PY" | tee -a "$LOG"

# ---------- 1. System Python 3.12 ----------------------------------------
step "Check Python 3.12"
if ! command -v "$PY" >/dev/null; then
    die "$PY not found. Install with: sudo apt-get install -y python3.12 python3.12-venv python3.12-dev"
fi
if ! "$PY" -c 'import venv' 2>/dev/null; then
    die "$PY lacks 'venv' module. Install: sudo apt-get install -y python3.12-venv"
fi
ok "$(command -v "$PY"): $("$PY" --version)"

# ---------- 2. Gaudi hardware sanity -------------------------------------
step "Check Gaudi hardware"
if command -v hl-smi >/dev/null; then
    # Lightweight driver-version query (the full table invocation has hung
    # under ssh/nohup redirection in the past).
    if drv="$(hl-smi --query-driver=version --format=csv,noheader 2>/dev/null | head -1)"; then
        ok "hl-smi: driver $drv"
    else
        ok "hl-smi present"
    fi
else
    warn "hl-smi NOT found — run_profile.sh will fail without Gaudi driver"
fi

# ---------- 3. Source repos + build outputs ------------------------------
step "Check npu-stack repos (deepest → highest dependency layers)"
# Layer 0–5 sources we depend on. Listed in topological order.
REPOS_NEEDED=(
    # Layer 1 — synapse runtime
    synapse synapse_utils synapse_profiler hcl scal aeon
    # Layer 2 — TPC toolchain (tpc_scalar_kernels and engines_fw are
    # consumed as pre-built artefacts, not source repos here)
    tpc_llvm10 tpc_kernels tpc_fuser complex_guid_lib
    # Layer 3 — PyTorch
    pytorch-fork pytorch-integration
    # Layer 4 — inference
    vllm vllm-gaudi
    # Layer 6 — tooling
    automation
    # Note: specs_external is compile-time only (headers baked into synapse
    # builds). It is documented in the layer diagram above but we don't check
    # for it here because it is not needed for a runtime venv.
)
missing=()
for r in "${REPOS_NEEDED[@]}"; do
    [[ -d "$NPU_STACK_ROOT/$r" ]] || missing+=("$r")
done
if (( ${#missing[@]} > 0 )); then
    die "missing npu-stack repos: ${missing[*]} (repo sync with: cd $NPU_STACK_ROOT && repo sync)"
fi
ok "${#REPOS_NEEDED[@]} repos present in $NPU_STACK_ROOT"

step "Check required builds (matching the layers above)"
BUILDS_NEEDED=(
    # Layer 1
    synapse_release_build synapse_utils_release_build synapse_profiler_release_build
    hcl_release_build scal_release_build aeon_release_build
    # Layer 2 (tpc_scalar_kernels + engines_fw come from pre-built artifacts)
    tpc_llvm_release_build tpc_kernels_release_build
    tpc_fuser_release_build complex_guid_lib_release_build
    # Layer 3
    pytorch_fork_release_build pytorch_modules_release_build
    # Aggregated symlinks
    latest
)
missing=()
for b in "${BUILDS_NEEDED[@]}"; do
    [[ -e "$BUILDS_ROOT/$b" ]] || missing+=("$b")
done
if (( ${#missing[@]} > 0 )); then
    die "missing builds: ${missing[*]} — run build_and_insmod_habanalabs first"
fi
# Check critical runtime artifacts
[[ -f "$BUILDS_ROOT/tpc_kernels_release_build/src/libtpc_kernels.so" ]] \
    || die "libtpc_kernels.so missing — rebuild tpc_kernels"
[[ -x "$BUILDS_ROOT/tpc_llvm_release_build/bin/tpc-clang" ]] \
    || warn "tpc-clang not found in tpc_llvm_release_build/bin"
[[ -f "$BUILDS_ROOT/latest/libhl_logger.so" ]] \
    || warn "libhl_logger.so not in builds/latest — habana_frameworks import will fail"
ok "${#BUILDS_NEEDED[@]} builds present under $BUILDS_ROOT"

# Locate torch wheel (its version dictates PT_WHEEL_VERS for pytorch-integration)
step "Locate torch wheel in pytorch_fork_release_build/pkgs"
TORCH_WHEEL="$(ls -1t "$BUILDS_ROOT/pytorch_fork_release_build/pkgs"/torch-*cp312*linux_x86_64.whl 2>/dev/null | head -1)"
[[ -f "$TORCH_WHEEL" ]] || die "no cp312 torch wheel in $BUILDS_ROOT/pytorch_fork_release_build/pkgs"
# torch-2.11.0a0+git3d7d1c1-cp312-cp312-linux_x86_64.whl → 2.11
TORCH_FULL_VER="$(basename "$TORCH_WHEEL" | sed -E 's/^torch-([^-+]+).*/\1/')"
PT_WHEEL_VERS="${TORCH_FULL_VER%.*}"      # 2.11.0a0 → 2.11
ok "torch wheel: $(basename "$TORCH_WHEEL")  (PT_WHEEL_VERS=$PT_WHEEL_VERS)"

# ---------- 4. Create / reuse venv ---------------------------------------
step "Provision venv at $VENV_DIR"
if [[ "$FORCE_REINSTALL" == "1" && -d "$VENV_DIR" ]]; then
    warn "FORCE_REINSTALL=1 → removing $VENV_DIR"
    rm -rf "$VENV_DIR"
fi
if [[ ! -d "$VENV_DIR" ]]; then
    mkdir -p "$(dirname "$VENV_DIR")"
    "$PY" -m venv "$VENV_DIR"
    ok "created venv"
else
    ok "venv already exists"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
ok "activated: $(python --version)"

# ---------- 5. Source habana_env (wires builds -> runtime env) -----------
step "Source habana_env"
HABANA_ENV="$NPU_STACK_ROOT/automation/habana_scripts/habana_env"
[[ -f "$HABANA_ENV" ]] || die "habana_env missing at $HABANA_ENV"
# habana_env references $1..$3 unguarded and also has path concatenations that
# use unset vars; relax `set -u` for the duration of the source.
set +u
# shellcheck disable=SC1090
source "$HABANA_ENV" >/dev/null 2>&1 || true
set -u
# Sanity: after sourcing we MUST have these
for v in BUILD_ROOT BUILD_ROOT_LATEST GC_KERNEL_PATH TPC_COMPILER_PATH PYTORCH_MODULES_ROOT_PATH LD_LIBRARY_PATH; do
    [[ -n "${!v:-}" ]] || die "habana_env did not set $v"
done
[[ -f "$GC_KERNEL_PATH" ]] || die "GC_KERNEL_PATH points to missing file: $GC_KERNEL_PATH"
ok "BUILD_ROOT_LATEST=$BUILD_ROOT_LATEST"
# H1: du may fail on dangling symlinks or permission errors; make it safe.
ok "GC_KERNEL_PATH=$GC_KERNEL_PATH ($(du -h "$GC_KERNEL_PATH" 2>/dev/null | awk '{print $1}' || echo '?'))"
ok "PYTORCH_MODULES_ROOT_PATH=$PYTORCH_MODULES_ROOT_PATH"

# ---------- 6. Helper funcs ----------------------------------------------
have_py() { python -c "import $1" >/dev/null 2>&1; }
pip_version() { python -c "import importlib.metadata as m; print(m.version('$1'))" 2>/dev/null || true; }

# Install pip requirements from <file>, stripping lines that would clobber
# our habana torch stack (torch / torchaudio / torchvision) or our pinned
# pip/setuptools. Missing file → warn + continue. No `-q` and no `|| true`
# on pip install: runtime-dep errors must be visible, not masked.
install_reqs() {
    local file="$1" label="$2"
    if [[ ! -f "$file" ]]; then
        warn "$label: requirements file not found at $file"
        return 0
    fi
    local filtered
    filtered="$(mktemp)"
    # Keep lines that don't start with a conflicting package name followed by
    # whitespace or a version operator. This preserves our habana torch and
    # our already-pinned pip/setuptools.
    grep -vE '^[[:space:]]*(torch|torchaudio|torchvision|pip|setuptools)[[:space:]=<>!]' \
        "$file" > "$filtered" || true
    local nlines
    nlines="$(grep -cvE '^[[:space:]]*(#|$)' "$filtered" || true)"
    if (( nlines > 0 )); then
        python -m pip install -r "$filtered"
        ok "$label: installed $nlines requirements from $(basename "$file")"
    else
        ok "$label: no requirements to install from $(basename "$file") (after filtering)"
    fi
    rm -f "$filtered"
}

# ---------- 7. Base tooling ----------------------------------------------
step "Upgrade pip/wheel/setuptools"
# pytorch-fork wheels pin setuptools<82 — respect that.
python -m pip install -q --upgrade pip wheel "setuptools<82"
ok "pip=$(python -m pip --version | awk '{print $2}'), setuptools=$(pip_version setuptools), wheel=$(pip_version wheel)"

step "Install pytorch-integration build requirements"
# PTI's requirements.txt has build-time deps (cmake, pybind11, numpy,
# symengine, packaging, ...). symengine in particular is required by
# python_packages/setup.py at import time for install_requires pinning.
install_reqs "$PYTORCH_MODULES_ROOT_PATH/requirements.txt" "pytorch-integration"

step "Install torch runtime-only deps (not covered by PTI req file)"
# torch imports these at runtime. Listed explicitly rather than letting the
# torch wheel resolve because we install torch with --no-deps (next step) to
# keep pip from yanking habana-compatible pins.
python -m pip install typing_extensions filelock sympy networkx jinja2 fsspec
ok "torch runtime deps installed"

# ---------- 8. torch+hpu from local wheel --------------------------------
# H5: verify successful import, not just that metadata exists. `import torch`
# can fail even when pip_version returns a string (corrupt wheel, broken
# extension loader, etc.). M1: removed duplicated torch-deps pip install —
# step 7 now installs them before torch.
step "Install torch+hpu from $(basename "$TORCH_WHEEL")"
current_torch="$(pip_version torch)"
if [[ "$current_torch" == "$TORCH_FULL_VER"* ]] && have_py torch; then
    ok "torch already installed and importable: $current_torch"
else
    python -m pip install -q --no-deps "$TORCH_WHEEL"
    have_py torch || die "torch installed but `import torch` fails"
    ok "installed torch $(pip_version torch)"
fi

# ---------- 9. habana_frameworks (editable install from pytorch-integration) -
# `habana_torch_plugin` is installed editable from pytorch-integration/python_packages.
# The .so files (_core_C.so, _hpu_C.so, ...) live at
#   $PYTORCH_MODULES_BUILD/habana_frameworks/torch/lib/fork_pybind/
# and the Python wrappers under habana_frameworks/torch/utils/_<name>_C.py
# dispatch to them via is_torch_fork (for torch+hpu forks) or upstream_pybind
# (for upstream torch). The editable install exposes the sources in the venv
# and the .so files are loaded directly from the build tree via the shim
# modules, with LD_LIBRARY_PATH (wired by habana_env) providing libSynapse etc.
step "Install habana_frameworks (editable from pytorch-integration/python_packages)"
current_htp="$(pip_version habana-torch-plugin)"
# Consider install good iff python -c import works for the fork_pybind shim.
htp_good=0
if [[ -n "$current_htp" ]] \
   && python -c "import habana_frameworks.torch.lib.fork_pybind._core_C" >/dev/null 2>&1; then
    htp_good=1
fi
# Env needed by setup.py AND by the symlink step below (used unconditionally).
export PYTORCH_MODULES_BUILD="$BUILDS_ROOT/pytorch_modules_release_build"
export PYTORCH_MODULES_WHL_BUILD_DIR="$PYTORCH_MODULES_BUILD/pkgs"
export PT_WHEEL_VERS

if (( htp_good )); then
    ok "habana-torch-plugin already editable: $current_htp"
else
    # H2: use a subshell (parens) instead of pushd/popd so that if pip fails
    # we don't leave the caller's cwd pointing into python_packages/.
    mkdir -p "$PYTORCH_MODULES_WHL_BUILD_DIR"
    ( cd "$PYTORCH_MODULES_ROOT_PATH/python_packages" \
      && python -m pip install -q --no-build-isolation --no-deps --force-reinstall -e . )
    ok "installed habana-torch-plugin $(pip_version habana-torch-plugin) (editable)"
fi

# Editable install leaves habana_frameworks/torch/lib/ empty; all .so files
# live in $PYTORCH_MODULES_BUILD (flat). __init__.py tries to dlopen
# lib/libhabana_pytorch2_plugin.so, and the _*_C.py shims import from
# lib/fork_pybind/_*_C.so. Expose both via symlinks into the build tree.
step "Link habana_frameworks .so libraries into source tree"
HF_LIB_DIR="$PYTORCH_MODULES_ROOT_PATH/python_packages/habana_frameworks/torch/lib"
if [[ -L "$HF_LIB_DIR" ]]; then
    target="$(readlink -f "$HF_LIB_DIR")"
    if [[ "$target" == "$PYTORCH_MODULES_BUILD" ]]; then
        ok "lib → $PYTORCH_MODULES_BUILD (already linked)"
    else
        warn "lib points to unexpected target $target; re-linking"
        ln -sfn "$PYTORCH_MODULES_BUILD" "$HF_LIB_DIR"
        ok "lib → $PYTORCH_MODULES_BUILD (relinked)"
    fi
elif [[ -e "$HF_LIB_DIR" ]]; then
    # H3: a plain directory here would silently shadow our intended symlink.
    # Fail loudly so the user can decide whether to delete it or keep.
    die "$HF_LIB_DIR exists and is not a symlink. Expected a symlink to
       $PYTORCH_MODULES_BUILD. Remove or rename it and re-run, or set
       FORCE_REINSTALL=1 (venv rebuild does not touch this path)."
else
    ln -sfn "$PYTORCH_MODULES_BUILD" "$HF_LIB_DIR"
    ok "lib → $PYTORCH_MODULES_BUILD"
fi
# fork_pybind subdir is where _*_C.py shims look for .so (same files, same
# dir). Since HF_LIB_DIR is a symlink to PYTORCH_MODULES_BUILD, fork_pybind
# must be created *inside* that build dir — a self-link pointing back to its
# own parent so the shims can resolve `lib/fork_pybind/_*_C.so`.
# H6: guard against a non-symlink entry already existing at that path.
fp_path="$PYTORCH_MODULES_BUILD/fork_pybind"
if [[ -L "$fp_path" ]]; then
    ok "fork_pybind symlink already present in build dir"
elif [[ -e "$fp_path" ]]; then
    die "$fp_path exists and is not a symlink (directory or file). Remove it and rerun."
else
    ln -sfn "$PYTORCH_MODULES_BUILD" "$fp_path"
    ok "fork_pybind → $PYTORCH_MODULES_BUILD (self-link)"
fi

# habana-torch-dataloader — separate package, same source tree? Actually it's
# a pure-python wheel from pkgs/. Install from there (layout is fine for it).
step "Install habana-torch-dataloader"
if [[ -n "$(pip_version habana-torch-dataloader)" ]]; then
    ok "already: $(pip_version habana-torch-dataloader)"
else
    dlr_whl="$(ls -1t "$BUILDS_ROOT/pytorch_modules_release_build/pkgs"/habana_torch_dataloader-*.whl 2>/dev/null | head -1)"
    if [[ -f "$dlr_whl" ]]; then
        python -m pip install -q --no-deps "$dlr_whl"
        ok "installed $(basename "$dlr_whl")"
    else
        warn "habana_torch_dataloader wheel not found — skipping (only needed for training dataloaders)"
    fi
fi

# ---------- 10. habana utility packages (from pip index) -----------------
step "Install habana-pyhlml / habana-media-loader / neural_compressor_pt"
# neural_compressor_pt is referenced by vllm_gaudi config guards; if missing,
# finalize_config() fails with PackageNotFoundError at model init time.
for pkg in habana-pyhlml habana-media-loader neural_compressor_pt; do
    if [[ -n "$(pip_version "$pkg")" ]]; then
        ok "$pkg: $(pip_version "$pkg")"
    elif python -m pip install -q "$pkg" 2>/dev/null; then
        ok "installed $pkg: $(pip_version "$pkg")"
    else
        warn "$pkg not resolvable — skipping (may affect media/monitoring only)"
    fi
done

# ---------- 11. Verify HPU import ----------------------------------------
# M2: Single cheap check — does habana_frameworks import + how many HPU
# devices are visible. The full-stack smoke (vllm/vllm_gaudi/LLM) happens
# at step 16 after those two are installed; duplicating here adds no value.
step "Smoke-test habana_frameworks import"
if ! python - <<'PY'
import habana_frameworks.torch  # noqa: F401
import torch
print(f'   torch {torch.__version__}, hpu_count={torch.hpu.device_count()}')
PY
then
    warn "habana_frameworks import FAILED."
    warn "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"
    warn "GC_KERNEL_PATH=${GC_KERNEL_PATH:-<unset>}"
    die "fix habana_frameworks provisioning before proceeding"
fi
if python -c "import torch; exit(0 if torch.hpu.device_count() > 0 else 1)" 2>/dev/null; then
    ok "HPU backend responsive"
else
    warn "HPU backend loaded but 0 devices visible (no Gaudi on this VM — OK for dev only)"
fi

# ---------- 12. vLLM runtime deps + editable install ---------------------
# C1: install vllm's runtime deps from requirements/common.txt BEFORE the
# editable install. Without this, pip leaves 40+ packages out and `import
# vllm` only works until something lazy-loads aiohttp, openai, etc.
step "Install vllm build + runtime requirements"
VLLM_DIR="$NPU_STACK_ROOT/vllm"
[[ -d "$VLLM_DIR" ]] || die "vllm sources missing at $VLLM_DIR"
# build.txt has setuptools-scm, cmake, ninja, build — required by vllm's
# pyproject.toml hooks during editable install (prepare_metadata_for_build_editable).
# common.txt has runtime deps (aiohttp, openai, transformers, fastapi, ...).
# Both filtered to preserve our habana torch + pinned setuptools/pip.
install_reqs "$VLLM_DIR/requirements/build.txt"  "vllm/build"
install_reqs "$VLLM_DIR/requirements/common.txt" "vllm/common"

step "Install vllm (editable, VLLM_TARGET_DEVICE=empty)"
# Idempotency via the .pth file, not `pip show vllm`. `pip show` reports the
# leftover dist-info from a previously failed editable install even after the
# .pth is gone — we'd then skip re-install and the venv stays broken.
# Similarly `have_py vllm` can pass via PYTHONPATH shadowing the source tree.
vllm_pth="$(ls "$VENV_DIR/lib/python3.12/site-packages"/__editable__.vllm-*.pth 2>/dev/null | head -1)"
if [[ -n "$vllm_pth" ]]; then
    ok "vllm already editable: $(basename "$vllm_pth")"
else
    # H2: subshell instead of pushd/popd so cwd never leaks on failure.
    # H4: no `|| true` — a build failure must surface, not be hidden.
    ( cd "$VLLM_DIR" \
      && VLLM_TARGET_DEVICE=empty python -m pip install --no-build-isolation -e . )
    ok "installed vllm $(pip_version vllm)"
fi

# ---------- 13. vllm-gaudi runtime deps + editable install ---------------
step "Install vllm-gaudi runtime requirements"
VLLM_GAUDI_DIR="$NPU_STACK_ROOT/vllm-gaudi"
[[ -d "$VLLM_GAUDI_DIR" ]] || die "vllm-gaudi sources missing at $VLLM_GAUDI_DIR"
install_reqs "$VLLM_GAUDI_DIR/requirements.txt" "vllm-gaudi"

step "Install vllm-gaudi (editable)"
if [[ -n "$(pip_version vllm-gaudi)" ]] && have_py vllm_gaudi; then
    ok "vllm-gaudi already: $(pip_version vllm-gaudi)"
else
    python -m pip install -e "$VLLM_GAUDI_DIR"
    ok "installed vllm-gaudi $(pip_version vllm-gaudi)"
fi

# ---------- 14. torchaudio + torchvision (Qwen3.5 / Qwen2.5-VL deps) ----
# vLLM imports Qwen2_5_VL model at registry time even when running text-only,
# which pulls in torchvision. Install CPU-only wheels (HPU path doesn't use
# them for compute — only for data transforms on CPU).
step "Install torchaudio + torchvision (CPU-only)"
for p in torchaudio torchvision; do
    if [[ -n "$(pip_version "$p")" ]]; then
        ok "$p: $(pip_version "$p")"
    else
        python -m pip install -q --no-deps "$p" \
            --extra-index-url https://download.pytorch.org/whl/cpu
        ok "installed $p $(pip_version "$p")"
    fi
done

# ---------- 15. Profiling helpers ---------------------------------------
step "Install profiling helpers"
for p in tensorboard torch-tb-profiler perfetto; do
    if [[ -n "$(pip_version "$p")" ]]; then
        ok "$p: $(pip_version "$p")"
    else
        python -m pip install -q "$p" && ok "installed $p" || warn "failed $p"
    fi
done

# ---------- 16. Final end-to-end smoke test ------------------------------
step "End-to-end import smoke test (with PYTHONPATH fix)"
# Strip bare NPU_STACK_ROOT from PYTHONPATH — otherwise `import vllm` resolves
# to the empty namespace dir npu-stack/vllm (no __init__.py) before the
# editable install at npu-stack/vllm/vllm/.
NEW_PP="$(python -c "
import os,sys
root=os.path.abspath(sys.argv[1]).rstrip('/')
pp=os.environ.get('PYTHONPATH','')
parts=[p for p in pp.split(':') if p and os.path.abspath(p).rstrip('/')!=root]
print(':'.join(parts))" "$NPU_STACK_ROOT")"
PYTHONPATH="$NEW_PP" python - <<PY
import vllm, vllm_gaudi, torch, habana_frameworks.torch  # noqa: F401
from vllm import LLM  # noqa: F401
print(f'   vllm:       {vllm.__file__}')
print(f'   vllm_gaudi: {vllm_gaudi.__file__}')
print(f'   torch:      {torch.__version__}')
print(f'   hpu count:  {torch.hpu.device_count()}')
PY
ok "all imports OK from $VENV_DIR"

# ---------- 17. Component versions --------------------------------------
# Print versions + git HEADs so a log captures the exact sources used.
# Uses `git -C` to avoid changing cwd. Falls back to '?' on any failure.
step "Component versions"
git_head() { git -C "$1" rev-parse --short HEAD 2>/dev/null || echo '?'; }
git_desc() { git -C "$1" describe --always --tags --dirty 2>/dev/null || echo '?'; }
git_msg()  { git -C "$1" log -1 --pretty=%s 2>/dev/null | head -c 70 || echo '?'; }
print_component() {
    local name="$1" dir="$2" pkg="$3"
    local ver head desc msg
    ver="$(pip_version "$pkg")"
    if [[ -d "$dir/.git" ]]; then
        head="$(git_head "$dir")"
        desc="$(git_desc "$dir")"
        msg="$(git_msg "$dir")"
        printf '   %-14s %-30s  git:%s  (%s)\n      └─ %s\n' \
            "$name" "${ver:-<not installed>}" "$head" "$desc" "$msg"
    else
        printf '   %-14s %-30s  (no .git at %s)\n' "$name" "${ver:-<not installed>}" "$dir"
    fi
}
print_component "vllm"          "$NPU_STACK_ROOT/vllm"                 vllm
print_component "vllm-gaudi"    "$NPU_STACK_ROOT/vllm-gaudi"           vllm-gaudi
print_component "pytorch-int."  "$PYTORCH_MODULES_ROOT_PATH"           habana-torch-plugin
print_component "torch"         "$NPU_STACK_ROOT/pytorch-fork"         torch

# ---------- 18. Summary --------------------------------------------------
step "Summary"
cat <<EOF
   Venv:        $VENV_DIR
   Activate:    source $VENV_DIR/bin/activate
                source $HABANA_ENV
   Next step:   ./run_profile.sh
   Setup log:   $LOG
EOF
echo "=== setup_env.sh finished at $(date) ===" | tee -a "$LOG"
