#!/usr/bin/env bash
set -Eeuo pipefail

# Derived from HydraGNN's scripts/hpc/alcf/aurora/installation/install.sh, keeping
# only what GrainGNN needs. PyTorch for Intel XPU comes from the ALCF `frameworks`
# module and must not be replaced by a pip wheel.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../../../" && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-${PWD}/GrainGNN-Installation-Aurora-XPU}"
VENV_PATH="${VENV_PATH:-${INSTALL_ROOT}/graingnn_venv_xpu}"
FRAMEWORKS_MODULE="${FRAMEWORKS_MODULE:-frameworks}"
RECREATE_ENV="${RECREATE_ENV:-0}"
TOTAL_STAGES=6
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

printf '[%s] Starting GrainGNN Aurora (Intel XPU) installation.\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')"
printf 'Environment prefix: %s\n' "${VENV_PATH}"

start_stage "Initialize and load the Aurora frameworks module"
# Lmod init scripts and the frameworks module reference unset variables.
set +u
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
    echo "Could not initialize the module command; run this script on ALCF Aurora." >&2
    exit 1
fi

module reset
module load "${FRAMEWORKS_MODULE}"
module -t list 2>&1
set -u

PYTHON_BIN="$(command -v python3 || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "python3 was not found after loading ${FRAMEWORKS_MODULE}." >&2
    exit 1
fi
printf 'Frameworks Python: %s\n' "${PYTHON_BIN}"
finish_stage

start_stage "Configure outbound network access"
# Aurora compute nodes reach the internet only through the ALCF proxy.
if ! curl --silent --head --max-time 10 https://pypi.org/simple/ >/dev/null 2>&1; then
    export HTTP_PROXY="${HTTP_PROXY:-http://proxy.alcf.anl.gov:3128}"
    export HTTPS_PROXY="${HTTPS_PROXY:-${HTTP_PROXY}}"
    export http_proxy="${HTTP_PROXY}"
    export https_proxy="${HTTPS_PROXY}"
    export no_proxy="${no_proxy:-admin,*.hostmgmt.cm.aurora.alcf.anl.gov,*.alcf.anl.gov,localhost}"
    printf 'Using proxy: %s\n' "${HTTPS_PROXY}"
else
    echo "Direct access to PyPI is available; no proxy configured."
fi
finish_stage

start_stage "Create or reuse the Python venv (inherits frameworks site-packages)"
mkdir -p "${INSTALL_ROOT}"
if [[ -d "${VENV_PATH}" && "${RECREATE_ENV}" -eq 1 ]]; then
    printf 'Removing existing environment: %s\n' "${VENV_PATH}"
    rm -rf "${VENV_PATH}"
fi
if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_PATH}"
else
    printf 'Reusing existing environment: %s\n' "${VENV_PATH}"
fi
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${INSTALL_ROOT}/.cache/pip}"
mkdir -p "${PIP_CACHE_DIR}"
python -m pip install --upgrade pip
finish_stage

start_stage "Verify frameworks PyTorch XPU build"
python - <<'PY'
import torch

assert hasattr(torch, "xpu"), f"PyTorch {torch.__version__} has no XPU backend"
print(f"PyTorch {torch.__version__}; file {torch.__file__}")
print(f"torch.xpu.is_available() = {torch.xpu.is_available()} (False is expected on login nodes)")
PY
finish_stage

start_stage "Install GrainGNN core dependencies"
# Pin the frameworks torch/numpy so no dependency can pull a CUDA wheel or a new NumPy ABI.
CONSTRAINTS_FILE="${INSTALL_ROOT}/pip-constraints.txt"
python - > "${CONSTRAINTS_FILE}" <<'PY'
from importlib.metadata import version

for name in ("torch", "numpy"):
    print(f"{name}=={version(name)}")
PY
cat "${CONSTRAINTS_FILE}"
PIP_CONSTRAINT="${CONSTRAINTS_FILE}" bash "${REPO_ROOT}/install_dependencies.sh"
finish_stage

start_stage "Verify torch and mpi4py come from frameworks, not the venv"
VENV_PATH="${VENV_PATH}" python - <<'PY'
import os

import mpi4py
import torch
import torch_geometric

venv = os.path.realpath(os.environ["VENV_PATH"])
for module in (torch, mpi4py):
    print(f"{module.__name__} {module.__version__}: {module.__file__}")
    if os.path.realpath(module.__file__).startswith(venv):
        raise SystemExit(f"{module.__name__} is shadowed by a venv install; pip uninstall it from {venv}")
print(f"torch_geometric {torch_geometric.__version__}: {torch_geometric.__file__}")
PY
finish_stage

cat <<EOF
[$(date '+%Y-%m-%d %H:%M:%S')] GrainGNN Aurora XPU installation complete.
Environment: ${VENV_PATH}

To activate it in a new shell:
  module reset
  module load ${FRAMEWORKS_MODULE}
  source ${VENV_PATH}/bin/activate

Then, inside a compute-node allocation, verify the Intel GPUs with:
  python scripts/check_gpu.py --require-xpu
EOF
