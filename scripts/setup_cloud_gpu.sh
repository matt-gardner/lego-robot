#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_cloud_gpu.sh --storage /mnt/training [--skip-system-packages] [--dry-run]

Bootstrap Ubuntu 24.04 x86_64 with an NVIDIA driver >= 580 already installed.
--storage must be an existing, writable mount of a persistent data disk.
Does not provision VMs, format/mount disks, upload data, or start training.
Creates an isolated CUDA 13.0 environment; leaves the robot/MPS .venv unchanged.
EOF
}

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

storage=
skip_system=0
dry_run=0
while (($#)); do
    case "$1" in
        --storage)
            (($# >= 2)) || fail "--storage needs a path"
            storage=$2
            shift 2
            ;;
        --skip-system-packages) skip_system=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) fail "Unknown argument: $1" ;;
    esac
done
[[ "$storage" == /* && "$storage" != / ]] || fail "Pass an absolute, non-root --storage mount"
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[[ -f "$repo/uv.lock" && -f "$repo/pyproject.toml" ]] || fail "Run from a complete project checkout"

if ((dry_run)); then
    usage
    printf '\nPlan: verify mounted storage %s and NVIDIA driver; install system libraries;\n' "$storage"
    printf 'install uv 0.11.18 and Python 3.12; export locked training/simulation dependencies;\n'
    printf 'create %s/so101-cloud/venv with CUDA wheels; check CUDA and EGL without loading a policy.\n' "$storage"
    printf 'Write %s/so101-cloud/training.env and installed-requirements.txt.\n' "$storage"
    exit 0
fi

[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || fail "Requires Linux x86_64"
source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 24.04 ]] || fail "Requires Ubuntu 24.04"
command -v mountpoint >/dev/null || fail "Install util-linux (mountpoint)"
storage=$(realpath -e -- "$storage")
[[ "$storage" != / ]] || fail "--storage resolves to the filesystem root"
mountpoint -q -- "$storage" || fail "--storage is not mounted; refusing to put state on the OS disk"
[[ "$(findmnt -n -o MAJ:MIN --target "$storage")" != "$(findmnt -n -o MAJ:MIN --target /)" ]] \
    || fail "--storage must be on a different filesystem from the OS disk"
[[ -w "$storage" ]] || fail "--storage must be writable by this user"
command -v nvidia-smi >/dev/null || fail "Use a GPU image with NVIDIA driver >= 580; driver installation/reboot is not automated"
drivers=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
[[ -n "$drivers" ]] || fail "No NVIDIA GPU found"
while IFS= read -r driver; do
    [[ "${driver%%.*}" =~ ^[0-9]+$ ]] || fail "Cannot parse NVIDIA driver version: $driver"
    (( ${driver%%.*} >= 580 )) || fail "CUDA 13.0 requires driver >= 580; found $driver"
done <<< "$drivers"

if ((!skip_system)); then
    as_root=()
    if ((EUID != 0)); then
        command -v sudo >/dev/null || fail "Install system packages as root or use --skip-system-packages"
        as_root=(sudo -n)
    fi
    "${as_root[@]}" apt-get update -qq
    "${as_root[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip ca-certificates git curl rsync ffmpeg \
        libegl1 libgl1 libglvnd0 libglfw3 libosmesa6 libglib2.0-0t64
fi
command -v python3 >/dev/null || fail "Python 3 is required"
state="$storage/so101-cloud"
mkdir -p "$state"
if [[ ! -x "$state/bootstrap/bin/python" ]]; then
    python3 -m venv "$state/bootstrap"
fi
"$state/bootstrap/bin/python" -m pip install --disable-pip-version-check --quiet 'uv==0.11.18'
uv="$state/bootstrap/bin/uv"
export UV_CACHE_DIR="$state/cache/uv"
export UV_PYTHON_INSTALL_DIR="$state/python"
export UV_CONCURRENT_DOWNLOADS=2
export UV_CONCURRENT_INSTALLS=2
"$uv" python install 3.12
"$uv" export --project "$repo" --locked --extra training --extra simulation \
    --no-dev --no-hashes --no-annotate --no-header --no-emit-project \
    --output-file "$state/requirements.cpu.txt"
python3 "$repo/src/so101/cloud_requirements.py" \
    "$state/requirements.cpu.txt" "$state/requirements.cuda.txt"
if [[ ! -x "$state/venv/bin/python" ]]; then
    "$uv" venv --python 3.12 "$state/venv"
fi
"$uv" --no-config pip install --python "$state/venv/bin/python" \
    --torch-backend cu130 --requirement "$state/requirements.cuda.txt"
"$uv" --no-config pip install --python "$state/venv/bin/python" --no-deps --editable "$repo"
"$uv" --no-config pip check --python "$state/venv/bin/python"
"$uv" --no-config pip freeze --python "$state/venv/bin/python" > "$state/installed-requirements.txt"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HOME="$state/cache/huggingface"
"$state/venv/bin/python" - <<'PY'
import torch
import mujoco
import so101.train_trajectory_rl

if not torch.cuda.is_available() or torch.version.cuda is None:
    raise RuntimeError("CUDA PyTorch is not usable; refusing CPU fallback")
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("The frozen visual/text prefix requires a BF16-capable GPU")
value = torch.ones(8, 8, device="cuda", requires_grad=True)
(value @ value).sum().backward()
torch.cuda.synchronize()
if not torch.isfinite(value.grad).all():
    raise RuntimeError("CUDA gradient smoke test failed")
model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><geom type="box" size=".1 .1 .1"/></worldbody></mujoco>'
)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, height=64, width=64)
try:
    renderer.update_scene(data)
    image = renderer.render()
    if image.shape != (64, 64, 3):
        raise RuntimeError("Unexpected EGL render output")
finally:
    renderer.close()
print(f"CUDA/EGL/import checks passed: {torch.cuda.get_device_name(0)}, Torch {torch.__version__}")
PY

{
    printf 'export SO101_ROOT=%q\n' "$repo"
    printf 'export SO101_PYTHON=%q\n' "$state/venv/bin/python"
    printf 'export HF_HOME=%q\n' "$HF_HOME"
    printf 'export MUJOCO_GL=egl\nexport PYOPENGL_PLATFORM=egl\n'
} > "$state/training.env"
printf 'Ready. Source %s/training.env, cd "$SO101_ROOT", and use "$SO101_PYTHON".\n' "$state"
printf 'Do not use plain uv run/sync in this CUDA environment: the root project selects CPU wheels on Linux.\n'
