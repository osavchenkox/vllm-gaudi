#!/usr/bin/env bash
# =============================================================================
# run_profile.sh — profile vllm-gaudi inference on a Gaudi VM.
#
# Portable (no hardcoded usernames / absolute $HOME paths). All outputs land
# in a timestamped run dir next to this script: $SCRIPT_DIR/runs/<timestamp>/.
#
# WHAT THIS SCRIPT DOES
#   1. Activates the Python venv prepared by setup_env.sh.
#   2. Sources Habana's environment (builds/latest on LD_LIBRARY_PATH, kernels,
#      TPC compiler, etc.) so that habana_frameworks can dlopen its libraries.
#   3. Strips the bare $NPU_STACK_ROOT entry from PYTHONPATH to avoid an
#      "import vllm" namespace-package collision.
#   4. Sanity-checks the driver (hl-smi) and the Python imports.
#   5. Runs hl-prof-config to enable either the full (profile_api_with_nics +
#      fuser + trace-analyzer + .hltv) or basic trace template.
#   6. Optionally caches the model weights to /tmp (local disk), so the first
#      run pays the NFS cost once and subsequent runs load from local disk.
#   7. Exports the vLLM-gaudi tuning env (buckets, lazy mode, MoE chunks, …)
#      plus the profiling env (VLLM_PROFILER_ENABLED=full, HABANA_PROFILE=1,
#      FUSER_DEBUG_DATA=1, FUSER_DEBUG_PATH).
#   8. Runs tests/test.py against $MODEL and intentionally asserts once the
#      profiler has collected the requested trace (vllm_gaudi does this by
#      design when VLLM_PROFILE_PROMPT is set).
#   9. On exit (trap), copies ~/.habana_logs into the run dir and prints a
#      summary listing every produced artefact.
#
# USAGE
#   cd /path/to/vllm-gaudi/tests
#   ./run_profile.sh                        # defaults (Qwen3.5-35B, prompt 2048)
#   MODEL=/path/to/other-model ./run_profile.sh
#   PROFILE_PROMPT=1,1024,0 ./run_profile.sh
#   DETAIL_LEVEL=basic ./run_profile.sh     # no .hltv (lighter trace)
#   MODEL_CACHE_DIR= ./run_profile.sh       # disable /tmp cache (reads NFS)
#   NO_COLOR=1 ./run_profile.sh             # disable colored stage banners
#   PROFILER_MODE=full ./run_profile.sh     # merge high-level into pt trace
#                                           # (side effect: no server_events
#                                           #  file — see sec. 14 comment)
#   ./run_profile.sh --help                 # print help + exit
#
# ENV OVERRIDES
#   MODEL              HF snapshot dir (default: Qwen3.5-35B-A3B on /software)
#   PROMPT             Prompt string passed to test.py
#   MAX_MODEL_LEN      default 8192
#   MAX_BATCHED_TOKENS auto-bumped to >= MAX_MODEL_LEN (vLLM requires this)
#   PROFILE_PROMPT     VLLM_PROFILE_PROMPT triplet (default "1,2048,0")
#   DETAIL_LEVEL       "full" (default) | "basic"
#   RUN_DIR            output dir       (default $SCRIPT_DIR/runs/<ts>)
#   VENV_DIR           venv to activate (default $HOME/.venvs/vllm-gaudi)
#   MODEL_CACHE_DIR    local cache root (default /tmp/hf_cache;
#                      set to empty to disable and read directly from $MODEL)
#   NPU_STACK_ROOT     path to npu-stack repo root (default derived from
#                      $SCRIPT_DIR/../..)
#   VLLM_BUILD         build tag for vllm_gaudi (default "1.24.1.0";
#                      must match regex ^\d+\.\d+\.\d+\.\d+$)
#   PROFILER_MODE      value for VLLM_PROFILER_ENABLED; "true" (default) keeps
#                      server_events_*.json and adds pt.trace.json.gz;
#                      "full" merges high-level into pt.trace but drops
#                      server_events_*.json (due to how boolean() parses it)
#
# ARTEFACTS PRODUCED (under $RUN_DIR)
#   run.log                                 # stdout/stderr, ANSI stripped
#   traces/<host>_<pid>.*.pt.trace.json.gz  # PyTorch+HPU trace for Perfetto
#   <run_dir>/*.hltv                        # device-level trace (full only)
#   <run_dir>/server_events_*.json          # high-level vLLM events
#   habana_logs/                            # snapshot of ~/.habana_logs
#   fuser_debug/                            # fuser graph/schedule dumps
# =============================================================================
set -euo pipefail

# ---------- 1. --help flag --------------------------------------------------
# Cheap arg handling: we only accept -h/--help. Everything else is configured
# via environment variables (documented above).
for arg in "$@"; do
    case "$arg" in
        -h|--help)
            sed -n '2,/^# ====/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
    esac
done

# ---------- 2. Colored banners ---------------------------------------------
# Emit ANSI colors to stdout only when it's a TTY AND NO_COLOR is not set to
# "1". Set NO_COLOR=1 to force plain output.
if [[ -t 1 && "${NO_COLOR:-}" != "1" ]]; then
    C_STEP=$'\033[1;34m'; C_OK=$'\033[1;32m'; C_WARN=$'\033[1;33m'
    C_ERR=$'\033[1;31m';  C_DIM=$'\033[2m';   C_RST=$'\033[0m'
else
    C_STEP= C_OK= C_WARN= C_ERR= C_DIM= C_RST=
fi
step() { printf '\n%s==> %s%s\n' "$C_STEP" "$*" "$C_RST"; }
ok()   { printf '   %sOK%s %s\n'  "$C_OK"   "$C_RST" "$*"; }
warn() { printf '   %sWARN%s %s\n' "$C_WARN" "$C_RST" "$*"; }
info() { printf '   %s%s%s\n'      "$C_DIM"  "$*"    "$C_RST"; }
die()  { printf '   %sERR%s %s\n'  "$C_ERR"  "$C_RST" "$*" >&2; exit 1; }

# ---------- 3. Locate script + create output directories -------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$SCRIPT_DIR/runs/$TS}"
mkdir -p "$RUN_DIR"

LOG_FILE="$RUN_DIR/run.log"
TRACE_DIR="$RUN_DIR/traces"
FUSER_DEBUG_DIR="$RUN_DIR/fuser_debug"
HABANA_LOGS_SAVE_DIR="$RUN_DIR/habana_logs"
mkdir -p "$TRACE_DIR" "$FUSER_DEBUG_DIR"

# Dual output: write everything (stdout + stderr) to the terminal AND to
# $LOG_FILE. tee forks a small sed filter to strip ANSI escape codes out of
# the on-disk log file so `cat run.log` is readable.
#
# NOTE: `set -o pipefail` combined with `exec > >(tee ...)` can cause the
# cleanup trap's output to be lost if the child process dies with SIGABRT
# (e.g. when synapse profiler trips an assertion). We therefore disable
# pipefail specifically for this pipeline redirection — `set -e` still
# catches other failures, but broken-pipe writes from the trap won't kill
# us. We also flush after every write with `stdbuf` to minimize lost
# output on abrupt exits.
set +o pipefail
exec > >(stdbuf -oL -eL tee >(sed -u $'s/\x1b\\[[0-9;]*m//g' > "$LOG_FILE")) 2>&1
set -o pipefail
step "run_profile.sh started at $(date)"
info "SCRIPT_DIR=$SCRIPT_DIR"
info "RUN_DIR=$RUN_DIR"

# ---------- 4. cleanup trap (always runs on exit) -------------------------
# Called on any exit (success, failure, or CTRL-C). Captures the framework
# logs and prints a final summary with a list of produced artefacts and
# links to the viewers.
cleanup() {
    local ec=$?
    step "Collecting post-run artefacts"

    # (a) snapshot $HABANA_LOGS (populated by the habana runtime at
    # $HOME/.habana_logs after habana_env is sourced). When the run succeeds
    # this captures graph_compiler.log / synapse_runtime.log / mme_stack.log
    # etc.; on failure it also captures any crash-time diagnostics.
    if [[ -n "${HABANA_LOGS:-}" && -d "${HABANA_LOGS:-/nonexistent}" ]]; then
        if cp -a "$HABANA_LOGS" "$HABANA_LOGS_SAVE_DIR" 2>/dev/null; then
            local sz
            sz="$(du -sh "$HABANA_LOGS_SAVE_DIR" 2>/dev/null | awk '{print $1}')"
            ok "habana_logs saved to $HABANA_LOGS_SAVE_DIR ($sz)"
        else
            warn "failed to copy $HABANA_LOGS -> $HABANA_LOGS_SAVE_DIR"
        fi
    fi

    # (b) confirm fuser_debug produced anything useful. If the run aborted
    # before the fuser ran, the directory will be empty — note that.
    if [[ -d "$FUSER_DEBUG_DIR" ]]; then
        local nfiles
        nfiles="$(find "$FUSER_DEBUG_DIR" -type f 2>/dev/null | wc -l)"
        if (( nfiles > 0 )); then
            ok "fuser debug data: $FUSER_DEBUG_DIR ($nfiles files)"
        else
            info "no fuser debug files produced (expected if the run aborted early)"
        fi
    fi

    # (c) metrics summary — runs here (not after python3) so we get it even
    # when test.py exits via `AssertionError: Finished profiling`. WALL_SEC
    # is set only after inference; fall back to epoch delta if we aborted
    # before that.
    step "Metrics summary"
    local wall
    if [[ -n "${WALL_SEC:-}" ]]; then
        wall="$WALL_SEC"
    elif [[ -n "${START_SEC:-}" ]]; then
        wall=$(( $(date +%s) - START_SEC ))
    else
        wall="?"
    fi
    info "test.py exit code: $ec  (non-zero typically means \"Finished profiling\" — expected with VLLM_PROFILE_PROMPT set)"
    info "wall-clock: ${wall} s"

    # Parse the latest server_events_*.json file (it's often truncated when
    # the run ends on an AssertionError; we parse events one-by-one instead
    # of json.load()'ing the whole file).
    if ls server_events_*.json >/dev/null 2>&1; then
        python3 - <<'PY' || true
import glob, json, sys
files = sorted(glob.glob('server_events_*.json'))
path = files[-1]
print(f'   source: {path}')
raw = open(path).read()
def iter_events(text):
    depth = 0; start = None
    for i, c in enumerate(text):
        if c == '{':
            if depth == 0: start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try: yield json.loads(text[start:i+1])
                except Exception: pass
                start = None
wanted = [
    'prompt_real_gen_throughput',
    'prompt_real_in_throughput',
    'prompt_bucket_gen_throughput',
    'prompt_bucket_in_throughput',
    'decode_real_gen_throughput',
    'decode_bucket_gen_throughput',
    'average_real_throughput',
    'cache_computed_utilization',
    'cache_num_blocks_used',
    'prompt_batch_block_utilization',
    'cumulative_graph_compilations',
    'engine_iteration',
]
last = {}
total = counters = 0
for e in iter_events(raw):
    total += 1
    if e.get('ph') != 'C':
        continue
    counters += 1
    for k, v in (e.get('args') or {}).items():
        if k in wanted:
            ts = e.get('ts', 0)
            if k not in last or ts >= last[k][1]:
                last[k] = (v, ts)
print(f'   events parsed: {total} total, {counters} counter samples')
if not last:
    print('   (no named counters found)')
else:
    for k in wanted:
        if k in last:
            v, _ = last[k]
            if isinstance(v, float):
                print(f'   {k:35s} = {v:.4f}')
            else:
                print(f'   {k:35s} = {v}')
PY
    else:
        info "(no server_events_*.json found in $RUN_DIR)"
    fi

    step "Summary"
    echo "   Run dir:    $RUN_DIR"
    echo "   Log:        $LOG_FILE"
    # List trace artefacts the rest of the tooling cares about.
    find "$RUN_DIR" -maxdepth 3 \
        \( -name '*.pt.trace.json.gz' -o -name 'server_events_*.json' -o -name '*.hltv' \) \
        -printf '   %p (%s bytes)\n' 2>/dev/null || true
    echo
    echo "View traces at: https://perfetto.habana.ai"
    echo "  -> Open trace file: *.pt.trace.json.gz  (PyTorch+HPU)"
    echo "  -> Or:              server_events_*.json (high-level vLLM)"
    echo "  *.hltv files (device-level, fused ops) require hltv-viewer"
    step "run_profile.sh finished at $(date) (exit=$ec)"
}
trap cleanup EXIT

# ---------- 5. Activate Python venv ----------------------------------------
step "Activate venv"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/vllm-gaudi}"
[[ -f "$VENV_DIR/bin/activate" ]] \
    || die "venv not found at $VENV_DIR; run ./setup_env.sh first"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
ok "$(which python) ($(python --version))"

# ---------- 6. Source habana_env ------------------------------------------
# habana_env wires every build under ~/builds into runtime env vars:
#   - LD_LIBRARY_PATH: adds ~/builds/latest (libSynapse, libhl_logger, ...)
#   - GC_KERNEL_PATH : points to libtpc_kernels.so
#   - TPC_COMPILER_PATH, HABANA_PLUGINS_LIB_PATH, PYTORCH_MODULES_ROOT_PATH,
#     HABANA_LOGS = $HOME/.habana_logs, and dozens of other *_ROOT / *_BUILD
#     variables used by test/debug scripts in the stack.
step "Source habana_env"
NPU_STACK_ROOT="${NPU_STACK_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
HABANA_ENV="$NPU_STACK_ROOT/automation/habana_scripts/habana_env"
if [[ -f "$HABANA_ENV" ]]; then
    # Two gotchas when sourcing habana_env:
    #   1) It references $1..$3 without guards, so it trips `set -u`.
    #   2) It has a side-effect that overwrites SCRIPT_DIR (a variable name it
    #      picks from sival/embedded sub-scripts). Save ours around the source.
    _OUR_SCRIPT_DIR="$SCRIPT_DIR"
    set +u
    # shellcheck disable=SC1090
    source "$HABANA_ENV" >/dev/null 2>&1 || true
    set -u
    SCRIPT_DIR="$_OUR_SCRIPT_DIR"
    unset _OUR_SCRIPT_DIR
    ok "sourced $HABANA_ENV"
else
    warn "$HABANA_ENV not found; expecting habana_frameworks already on PATH"
fi

# ---------- 7. Fix PYTHONPATH (namespace-package collision) ---------------
# habana_env prepends $NPU_STACK_ROOT to PYTHONPATH. With the editable install
# of vllm at $NPU_STACK_ROOT/vllm/, Python resolves `import vllm` to the empty
# namespace directory $NPU_STACK_ROOT/vllm (which has no top-level
# __init__.py) instead of the real editable install at
# $NPU_STACK_ROOT/vllm/vllm/__init__.py. Strip only the bare root; keep every
# other habana_py / synapse / builds path intact.
step "Adjust PYTHONPATH"
if [[ -n "${PYTHONPATH:-}" ]]; then
    PYTHONPATH="$(python3 -c "
import os,sys
root=os.path.abspath(sys.argv[1]).rstrip('/')
pp=os.environ.get('PYTHONPATH','')
parts=[p for p in pp.split(':') if p and os.path.abspath(p).rstrip('/')!=root]
print(':'.join(parts))" "$NPU_STACK_ROOT")"
    export PYTHONPATH
fi
ok "PYTHONPATH stripped (first 3 entries):"
echo "$PYTHONPATH" | tr ':' '\n' | sed -n '1,3p' | sed 's/^/     /'

# ---------- 8. Hardware check ---------------------------------------------
# Run `hl-smi` and print enough lines to cover up to 8 devices (each device
# block is ~3 lines; header is 7). If the driver is broken the tool itself
# will fail — we want to stop early rather than find out deep in vLLM init.
step "Check Gaudi hardware"
command -v hl-smi >/dev/null || die "hl-smi not found. Is habanalabs driver installed?"
hl-smi | head -40 | sed 's/^/   /' || true
ok "hl-smi OK"

# ---------- 9. Python import check ----------------------------------------
# Fail fast if the venv is missing anything: habana_frameworks, HPU count > 0,
# or the editable vllm/vllm_gaudi installs. This avoids a 40+ minute wait on
# model loading only to trip over a missing import.
step "Verify Python imports"
python3 - <<'PY'
import habana_frameworks.torch as _htorch
import torch
assert torch.hpu.device_count() > 0, 'No HPU devices visible to PyTorch'
print(f'   hpu_count: {torch.hpu.device_count()}')
import vllm
if not hasattr(vllm, 'LLM'):
    raise RuntimeError(f'vllm.LLM missing; __file__={vllm.__file__} __path__={list(vllm.__path__)}')
import vllm_gaudi  # noqa
print('   imports OK')
PY

# ---------- 10. Configure Habana profiler --------------------------------
# Writes $HOME/.habana/prof_config.json; the file is read on next
# HABANA_PROFILE=1 run. Two templates:
#   - full:  profile_api_with_nics + fuser + trace-analyzer => .hltv files
#            with fused-op graph structure and node names. Largest output.
#   - basic: profile_api --hw-trace off. Lighter, no .hltv. Use when the
#            full trace is too big to load in a browser.
DETAIL_LEVEL="${DETAIL_LEVEL:-full}"
step "Configure Habana profiler (DETAIL_LEVEL=$DETAIL_LEVEL)"
if ! command -v hl-prof-config >/dev/null; then
    warn "hl-prof-config not in PATH; adding \$HOME/builds/latest"
    export PATH="$HOME/builds/latest:$PATH"
fi
command -v hl-prof-config >/dev/null \
    || die "hl-prof-config still not found (check \$HOME/builds/latest)"
case "$DETAIL_LEVEL" in
    full)
        hl-prof-config --use-template profile_api_with_nics --fuser on --trace-analyzer on
        export HABANA_PROFILE_WRITE_HLTV=1
        ok "template=profile_api_with_nics + fuser + trace-analyzer; HABANA_PROFILE_WRITE_HLTV=1"
        ;;
    basic)
        hl-prof-config --use-template profile_api --hw-trace off
        ok "template=profile_api (hw-trace off)"
        ;;
    *)
        die "DETAIL_LEVEL must be 'full' or 'basic' (got '$DETAIL_LEVEL')"
        ;;
esac

# ---------- 11. Model path + local /tmp cache ----------------------------
# Loading Qwen3.5-35B over NFS took ~42 minutes in our measurements. Caching
# it to local disk (default /tmp/hf_cache) turns subsequent runs into
# seconds. Set MODEL_CACHE_DIR= (empty) to disable and read straight off NFS.
# We use `${VAR-default}` (single dash) so that an explicitly empty value
# does NOT trigger the default — that's the user's "disable cache" knob.
step "Resolve model path (optional local cache)"
MODEL_DEFAULT="/software/data/pytorch/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307"
MODEL_SRC="${MODEL:-$MODEL_DEFAULT}"
[[ -d "$MODEL_SRC" ]] || die "MODEL dir does not exist: $MODEL_SRC"
info "source: $MODEL_SRC"

MODEL_CACHE_DIR="${MODEL_CACHE_DIR-/tmp/hf_cache}"
if [[ -n "$MODEL_CACHE_DIR" ]]; then
    # Build a stable cache key. The standard HF layout under /software is:
    #   .../hub/models--<owner>--<name>/snapshots/<sha>/
    # which we map to "<owner>--<name>/<sha>" so different snapshots of the
    # same model don't collide.
    if [[ "$MODEL_SRC" == *"/snapshots/"* ]]; then
        cache_key="$(echo "$MODEL_SRC" \
            | sed -E 's|.*/models--([^/]+)/snapshots/([^/]+)/?$|\1/\2|')"
    else
        cache_key="$(basename "$MODEL_SRC")"
    fi
    MODEL_LOCAL="$MODEL_CACHE_DIR/$cache_key"

    # Consider the cache valid iff model.safetensors.index.json exists AND
    # the total byte size of *.safetensors matches the source. This is a
    # quick sanity check; a stronger one would be sha256, but that doubles
    # the first-run time and adds little for a dev/tmp cache.
    need_copy=1
    if [[ -f "$MODEL_LOCAL/model.safetensors.index.json" ]]; then
        src_sz="$(du -sb "$MODEL_SRC"/*.safetensors 2>/dev/null | awk '{s+=$1} END {print s+0}')"
        dst_sz="$(du -sb "$MODEL_LOCAL"/*.safetensors 2>/dev/null | awk '{s+=$1} END {print s+0}')"
        [[ "$src_sz" != "0" && "$src_sz" == "$dst_sz" ]] && need_copy=0
    fi

    if (( need_copy )); then
        # Require ~10% headroom on the cache filesystem before starting the
        # (potentially 70+ GB) copy; otherwise fall back to NFS.
        src_mb="$(du -sBM "$MODEL_SRC" 2>/dev/null | awk '{print $1+0}')"
        cache_parent="$(dirname "$MODEL_CACHE_DIR")"
        free_mb="$(df -BM "$cache_parent" 2>/dev/null | tail -1 | awk '{print $4+0}')"
        info "source ~${src_mb} MB; free on cache mount ~${free_mb} MB"
        if (( free_mb < src_mb * 11 / 10 )); then
            warn "insufficient space; falling back to $MODEL_SRC"
            MODEL="$MODEL_SRC"
        else
            mkdir -p "$MODEL_LOCAL"
            info "copying model to $MODEL_LOCAL (first-time cost, then reused)..."
            # -a: archive (perms, symlinks, timestamps).
            # -L: resolve symlinks — the HF layout points snapshots/<sha>/*.
            #     safetensors at ../blobs/<sha>, and we want the blob data
            #     on the local disk, not a dangling symlink.
            # --info=progress2: live progress bar.
            # Trailing slash on MODEL_SRC/ matters: it copies the contents of
            # the dir, not the dir itself.
            time rsync -aL --info=progress2 "$MODEL_SRC"/ "$MODEL_LOCAL"/
            ok "cached at $MODEL_LOCAL"
            MODEL="$MODEL_LOCAL"
        fi
    else
        ok "cache hit: $MODEL_LOCAL (reusing)"
        MODEL="$MODEL_LOCAL"
    fi
else
    info "MODEL_CACHE_DIR is empty -> reading directly from $MODEL_SRC"
    MODEL="$MODEL_SRC"
fi

# ---------- 12. Runtime params -------------------------------------------
PROMPT="${PROMPT:-Nebius stock rose by double-digits on Wednesday. Nvidia will invest \$2 billion in Nebius. tell me more about nvidia}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
# vLLM (pydantic) rejects max_num_batched_tokens < max_model_len. The
# original q35s.sh had 4096/8192 and crashed at config validation; auto-bump.
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-$MAX_MODEL_LEN}"
if (( MAX_BATCHED_TOKENS < MAX_MODEL_LEN )); then
    warn "bumping MAX_BATCHED_TOKENS $MAX_BATCHED_TOKENS -> $MAX_MODEL_LEN"
    MAX_BATCHED_TOKENS="$MAX_MODEL_LEN"
fi

# ---------- 13. vLLM-gaudi tuning env (from q35s.sh) ---------------------
# Kept identical to q35s.sh so that performance comparisons stay apples-to-
# apples. See docs/configuration/performance_tuning.md for details on each.
step "Export vLLM-gaudi tuning env"
export VLLM_PROMPT_BS_BUCKET_MIN=4
export VLLM_PROMPT_BS_BUCKET_STEP=4
export VLLM_PROMPT_BS_BUCKET_MAX=8
export PT_HPU_LAZY_MODE=0
export VLLM_EXPONENTIAL_BUCKETING=true
export VLLM_PROMPT_QUERY_BUCKET_MIN=128
export VLLM_PROMPT_QUERY_BUCKET_STEP=128
export VLLM_PROMPT_QUERY_BUCKET_MAX=128
export VLLM_PROMPT_CTX_BUCKET_MIN=0
export VLLM_PROMPT_CTX_BUCKET_STEP=1
export VLLM_PROMPT_CTX_BUCKET_MAX=64
export VLLM_FUSED_BLOCK_SOFTMAX_ADJUSTMENT=False
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export EXPERIMENTAL_WEIGHT_SHARING=0
export FUSER_ENABLE_LOW_UTILIZATION=true
export ENABLE_FUSION_BEFORE_NORM=true
export VLLM_SKIP_WARMUP=true
export VLLM_FP32_SOFTMAX=false
export VLLM_USE_HYBRID_CACHE=true
export VLLM_USE_NAIVE_MAMBA_CACHE_SHARING=false
export VLLM_COMPACT_GDN=1
export VLLM_GRAPH_RESERVED_MEM=0.2
export VLLM_PROFILE_STEPS=0,5
export PT_HPU_ENABLE_EAGER_CACHE=true
export VLLM_MOE_CHUNK="64,128,256"
export VLLM_MOE_TOKEN_BOUNDARY="16,64,99999"
ok "tuning env set"

# ---------- 14. Profiling + fuser-debug env ------------------------------
# VLLM_BUILD regex check: vllm_gaudi validates the detected build against
# ^\d+\.\d+\.\d+\.\d+$. habana's auto-detected "1.24.1+git0756d47a4" fails
# that regex, so we pin a pattern-matching synthetic version here.
step "Export profiling + fuser-debug env"
export VLLM_BUILD="${VLLM_BUILD:-1.24.1.0}"
export LOG_LEVEL_PERF_LIB=0
# NOTE on VLLM_PROFILER_ENABLED values and what they actually enable:
#   - =true : high-level profiler is active (vllm_gaudi/extension/features.py
#             parses via `boolean()`, which only matches true/1/yes/on). That
#             profiler writes `server_events_<instance>.json` with per-step
#             counters: prompt/decode throughput (real + bucketed), cache
#             utilization, graph compilations, engine_iteration, etc. This
#             is what we want for "tokens per second" metrics.
#   - =full : PyTorch profiler's trace handler switches to
#             `full_trace_handler` (merges high-level events INTO the
#             .pt.trace.json.gz) — BUT `boolean('full') == False`, so the
#             high-level profiler itself is disabled and NO server_events
#             file is written. Confusing but that's the current behavior.
#   - (unset or =false): both profilers are off.
#
# Conclusion: we set =true here so we reliably get server_events_*.json. The
# PyTorch trace still fires because VLLM_TORCH_PROFILER_DIR is set (it's the
# real switch for the torch profiler — VLLM_PROFILER_ENABLED only affects
# the handler choice). The downside vs =full is slightly less merged info
# inside the .pt.trace.json.gz; server_events_*.json carries it separately.
# Set PROFILER_MODE=full to opt into the merged-trace-only behavior.
export VLLM_PROFILER_ENABLED="${PROFILER_MODE:-true}"
# Respect an explicitly empty PROFILE_PROMPT (disables the warmup-time
# `AssertionError: Finished profiling` and lets inference run end-to-end).
# We use ${VAR-default} (single dash) so `PROFILE_PROMPT=` stays empty.
VLLM_PROFILE_PROMPT_DEFAULT="1,2048,0"
export VLLM_PROFILE_PROMPT="${PROFILE_PROMPT-$VLLM_PROFILE_PROMPT_DEFAULT}"
if [[ -z "$VLLM_PROFILE_PROMPT" ]]; then
    unset VLLM_PROFILE_PROMPT
fi
export HABANA_PROFILE=1
export VLLM_TORCH_PROFILER_DIR="$TRACE_DIR"
# Fuser debug: dump graph/schedule/tensor-shape artefacts for every fuser
# invocation into $FUSER_DEBUG_PATH. Useful when diagnosing why an op was
# (or wasn't) fused into a larger TPC kernel.
export FUSER_DEBUG_DATA=1
export FUSER_DEBUG_PATH="$FUSER_DEBUG_DIR"
ok "VLLM_PROFILER_ENABLED=$VLLM_PROFILER_ENABLED, VLLM_PROFILE_PROMPT=$VLLM_PROFILE_PROMPT"
ok "FUSER_DEBUG_DATA=1, FUSER_DEBUG_PATH=$FUSER_DEBUG_PATH"

# Synapse IR graph dumps (pre/post optimization passes).
# Controlled by DUMP_SYNAPSE_GRAPHS env var (default: off — large output).
if [[ "${DUMP_SYNAPSE_GRAPHS:-}" == "1" ]]; then
    SYNAPSE_GRAPH_DIR="$RUN_DIR/synapse_graphs"
    mkdir -p "$SYNAPSE_GRAPH_DIR"
    export ENABLE_EXPERIMENTAL_FLAGS=true
    export GRAPH_VISUALIZATION=1
    export DUMP_PRE_GRAPHS="$SYNAPSE_GRAPH_DIR"
    export DUMP_POST_GRAPHS="$SYNAPSE_GRAPH_DIR"
    ok "Synapse graph dumps: $SYNAPSE_GRAPH_DIR"
fi

# Clean stale $HABANA_LOGS so the snapshot we collect at exit contains ONLY
# this run's framework output. Only touch paths under $HOME to avoid wiping
# a shared system location by accident.
if [[ -n "${HABANA_LOGS:-}" && "${HABANA_LOGS}" == "$HOME/"* ]]; then
    info "cleaning $HABANA_LOGS (will be collected into run dir on exit)"
    rm -rf "$HABANA_LOGS"
    mkdir -p "$HABANA_LOGS"
fi

# ---------- 15. Launch inference ----------------------------------------
# test.py is the workload runner shared with q35s.sh. We chdir into RUN_DIR
# so that files written relative to cwd (server_events_*.json, *.hltv)
# land inside the run directory automatically.
TEST_PY="$SCRIPT_DIR/test.py"
[[ -f "$TEST_PY" ]] || die "$TEST_PY not found"

step "Launch inference at $(date)"
info "MODEL=$MODEL"
info "MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_BATCHED_TOKENS=$MAX_BATCHED_TOKENS"
info "TRACE_DIR=$TRACE_DIR"
info "HABANA_LOGS=${HABANA_LOGS:-<not set>}"
info "FUSER_DEBUG_PATH=$FUSER_DEBUG_PATH"

# Time the whole inference so we report a wall-clock number even when the run
# ends with vllm_gaudi's intentional `AssertionError: "Finished profiling"`
# (which is expected when VLLM_PROFILE_PROMPT is set). `time -p` writes
# `real/user/sys` in seconds to stderr in a stable format we can grep later.
# We don't want `set -e` to kill the script the moment test.py asserts — we
# still want the EXIT trap to collect habana_logs — so we allow a non-zero
# return here by OR-ing with `true` and capturing the exit code ourselves.
cd "$RUN_DIR"
# START_SEC is global so the cleanup trap can use it to fall back to a rough
# wall-clock estimate even if the python call below aborts via an uncaught
# exception/signal that skips the WALL_SEC assignment.
START_SEC="$(date +%s)"
TEST_EC=0
# `time -p` writes the real/user/sys timings to stderr on its own lines.
# `|| TEST_EC=$?` keeps `set -e` from killing us on the expected
# AssertionError("Finished profiling") so the cleanup trap can still run.
{ time -p python3 "$TEST_PY" \
    --model "$MODEL" \
    --mode text \
    --text-api generate \
    --prompt "$PROMPT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" ; } || TEST_EC=$?
WALL_SEC=$(( $(date +%s) - START_SEC ))

# Exit with 0 so cleanup() sees `ec=0` for clean finishes; the EXIT trap
# runs regardless and prints the metrics summary using TEST_EC / WALL_SEC.
exit "$TEST_EC"
