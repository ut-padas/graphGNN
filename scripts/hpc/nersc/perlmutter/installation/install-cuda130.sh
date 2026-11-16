#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../../../" && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-${PWD}/GrainGNN-Installation-Perlmutter-CUDA130}"
VENV_PATH="${VENV_PATH:-${INSTALL_ROOT}/graingnn_venv_cuda130}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PYTORCH_CUDA_INDEX_URL="${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
TORCH_VERSION="${TORCH_VERSION:-2.14.0}"
TOTAL_STAGES=5
STAGE_NUMBER=0
CURRENT_STAGE="startup"
STAGE_STARTED=${SECONDS}

start_stage() {
    STAGE_NUMBER=$((STAGE_NUMBER + 1))
    CURRENT_STAGE="$1"
    STAGE_STARTED=${SECONDS}
    printf '\n[%s] Stage %d/%d: %s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "${STAGE_NUMBER}" "${TOTAL_STAGES}" "${CURRENT_STAGE}"
}

finish_stage() {
    local elapsed=$((SECONDS - STAGE_STARTED))
    printf '[%s] Finished stage %d/%d: %s (elapsed: %02d:%02d)\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "${STAGE_NUMBER}" "${TOTAL_STAGES}" \
        "${CURRENT_STAGE}" "$((elapsed / 60))" "$((elapsed % 60))"
}

report_failure() {
    local exit_code="$1"
    local line_number="$2"
    local command="$3"
    printf '[%s] FAILED during stage %d/%d (%s), line %s, exit code %s: %s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "${STAGE_NUMBER}" "${TOTAL_STAGES}" \
        "${CURRENT_STAGE}" "${line_number}" "${exit_code}" "${command}" >&2
    exit "${exit_code}"
}

trap 'report_failure "$?" "$LINENO" "$BASH_COMMAND"' ERR

printf '[%s] Starting GrainGNN Perlmutter CUDA 13.0 installation.\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')"
printf 'Environment prefix: %s\n' "${VENV_PATH}"

start_stage "Initialize and load the Perlmutter module stack"
if ! command -v module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/modules.sh ]]; then
        source /etc/profile.d/modules.sh
    elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
        source /usr/share/lmod/lmod/init/bash
    elif [[ -f /usr/share/Modules/init/bash ]]; then
        source /usr/share/Modules/init/bash
    fi
fi

if ! command -v module >/dev/null 2>&1; then
    echo "Could not initialize the module command; run this script on NERSC Perlmutter." >&2
    exit 1
fi

module reset
module load cpe/26.03
module load PrgEnv-gnu/8.7.0
module load craype-accel-nvidia80
module load cudatoolkit/13.0
module load conda/Miniforge3-26.9.4-1

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda was not found after loading the conda module." >&2
    exit 1
fi
export PYTHONNOUSERSITE=1
finish_stage

start_stage "Create or reuse the Conda environment"
mkdir -p "${INSTALL_ROOT}"
if [[ ! -d "${VENV_PATH}/conda-meta" ]]; then
    conda create --yes --prefix "${VENV_PATH}" "python=${PYTHON_VERSION}" pip
else
    printf 'Reusing existing environment: %s\n' "${VENV_PATH}"
fi
conda env config vars set --prefix "${VENV_PATH}" PYTHONNOUSERSITE=1
finish_stage

export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${INSTALL_ROOT}/.cache/pip}"
mkdir -p "${PIP_CACHE_DIR}"

start_stage "Install PyTorch ${TORCH_VERSION} for CUDA 13.0"
conda run --prefix "${VENV_PATH}" python -m pip install --progress-bar on \
    --index-url "${PYTORCH_CUDA_INDEX_URL}" \
    --extra-index-url https://pypi.org/simple \
    "torch==${TORCH_VERSION}"
finish_stage

start_stage "Verify PyTorch CUDA version"
conda run --no-capture-output --prefix "${VENV_PATH}" python - <<'PY'
import torch

assert torch.version.cuda is not None, f"Expected a CUDA PyTorch build, got {torch.__version__}"
assert torch.version.cuda.startswith("13.0"), f"Expected CUDA 13.0, got {torch.version.cuda}"
print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}")
PY
finish_stage

start_stage "Install GrainGNN core dependencies"
conda run --prefix "${VENV_PATH}" bash "${REPO_ROOT}/install_dependencies.sh"
finish_stage

cat <<EOF
[$(date '+%Y-%m-%d %H:%M:%S')] GrainGNN Perlmutter CUDA 13.0 installation complete.
Environment: ${VENV_PATH}

To activate it in a new shell, load the same module stack and run:
  conda activate ${VENV_PATH}
EOF
