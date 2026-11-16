# GrainGNN Installation

Create one Python environment for GrainGNN on each machine. The laptop uses Python's `venv`; TACC uses Conda; Frontier uses the ROCm 7.2 Conda installer included in this repository. Follow only the section for the machine you are using.

## Laptop

From the repository root, create and activate a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch for the laptop's hardware using the [official PyTorch selector](https://pytorch.org/get-started/locally/). For example, for a CPU-only environment:

```sh
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
```

Then install GrainGNN's PyTorch Geometric and scientific Python dependencies:

```sh
./install_dependencies.sh
```

The script expects PyTorch to already be installed, so it will not replace a selected CPU, CUDA, or ROCm build. It installs only the core dependencies listed in `requirements-core.txt`; VTK is not included because it is only needed by the optional ExaCA VTK reader. Install it separately with `python -m pip install vtk` if using that reader.

## TACC Vista and Stampede3

The TACC environment guide has the complete installation and sample workflow. It uses the existing Conda installation under `$WORK`. In a login shell, initialize Conda, create the environment once, and activate it:

```sh
source "$WORK/anaconda3/bin/activate"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
conda create -n GNN --override-channels -c conda-forge python=3.12 pip
conda activate GNN
```

Then continue with the PyTorch and GrainGNN dependency installation in [the TACC guide](env_tacc.md). In a new shell or batch job, initialize Conda and activate the existing environment with:

```sh
source "$WORK/anaconda3/bin/activate" GNN
```

## OLCF Frontier

From the repository root on Frontier, run the installer. It creates a Conda environment at `GrainGNN-Installation-Frontier-ROCm72/graingnn_venv_rocm72` by default, installs PyTorch for ROCm 7.2, then installs GrainGNN's core dependencies:

```sh
bash scripts/hpc/olcf/frontier/installation/install-rocm72.sh
```

To activate the environment in a later shell, first change to the repository root (the default environment path is relative to it), then load the same module stack and activate its Conda prefix:

```sh
cd /path/to/graphGNN
module reset
module load cpe/24.07 cce/18.0.0 rocm/7.2.0 amd-mixed/7.2.0
module load craype-accel-amd-gfx90a PrgEnv-gnu miniforge3/23.11.0-0
conda activate "$PWD/GrainGNN-Installation-Frontier-ROCm72/graingnn_venv_rocm72"
```

The installer configures this Conda environment to set `PYTHONNOUSERSITE=1` on activation. This prevents packages installed in your account's `~/.local` from shadowing the environment's ROCm PyTorch. Verify the active interpreter and PyTorch build with:

```sh
python -c 'import sys, torch; print(sys.executable); print(torch.__version__, torch.version.hip); print(torch.__file__)'
```

The executable and Torch path should be inside the `graingnn_venv_rocm72` directory, and `torch.version.hip` should report ROCm 7.2. The installer accepts `INSTALL_ROOT`, `VENV_PATH`, and `PYTHON_VERSION` overrides; see the script header for defaults. It does not install HydraGNN's additional packages or optional PyG compiled extensions.

The Frontier installer prints timestamped stage start and finish messages, elapsed time per stage, and the failing stage if a command exits with an error. To run it in the background and follow its progress from the repository root:

```sh
nohup bash scripts/hpc/olcf/frontier/installation/install-rocm72.sh > frontier-install.log 2>&1 &
tail -f frontier-install.log
```

The log reports the current installation stage; package downloads may also show pip's progress bar. Press `Ctrl+C` to stop following the log without stopping the installer.

## NERSC Perlmutter

From the repository root on Perlmutter, run the installer. It creates a Conda environment at `GrainGNN-Installation-Perlmutter-CUDA130/graingnn_venv_cuda130` by default, installs PyTorch for CUDA 13.0, then installs GrainGNN's core dependencies:

```sh
bash scripts/hpc/nersc/perlmutter/installation/install-cuda130.sh
```

To activate the environment in a later shell, change to the repository root, load the same module stack and activate its Conda prefix:

```sh
cd /path/to/graphGNN
module reset
module load cpe/26.03 PrgEnv-gnu/8.7.0 craype-accel-nvidia80 cudatoolkit/13.0
module load conda/Miniforge3-26.9.4-1
conda activate "$PWD/GrainGNN-Installation-Perlmutter-CUDA130/graingnn_venv_cuda130"
```

As on Frontier, the environment sets `PYTHONNOUSERSITE=1` on activation, accepts `INSTALL_ROOT`, `VENV_PATH`, and `PYTHON_VERSION` overrides, and does not install HydraGNN's additional packages or optional PyG compiled extensions. Run the GPU smoke test inside a GPU allocation with `python scripts/check_gpu.py`.

## Verify

After installation, activate the environment and run the focused tests:

```sh
python -m pip install -r requirements-test.txt
python -m pytest tests/
```

The published training-dataset test is opt-in because its archive is about 1 GB compressed and expands to several GB. The test needs a writable cache directory with enough free space for both files. By default it uses `~/.cache/graingnn`. To choose a different cache directory, create it and set `GRAINGNN_DATA_DIR` before running the test. For example, from the repository root:

```sh
mkdir -p "$HOME/graingnn-data-cache"
export GRAINGNN_DATA_DIR="$HOME/graingnn-data-cache"
```

Keep that environment variable set in later shells when you want to reuse this cache. If `dataset_train.pkl` is already in the directory, the test uses it. If not, but `dataset_train.pkl.gz` is present, the test extracts and caches the dataset. If neither file is present, it downloads the archive from the Google Drive link in the README, then extracts it. You can also put an archive you already downloaded at `$GRAINGNN_DATA_DIR/dataset_train.pkl.gz`. The cache is outside the repository in this example, so the large data files will not be included in Git.

Run the opt-in test with the cache configured:

```sh
GRAINGNN_RUN_TRAINING_TEST=1 python -m pytest -m integration tests/test_training_integration.py
```

The training check uses two samples for one epoch and writes model/output files only to pytest's temporary directory. For a dataset archive already downloaded to the repository root, you can instead set `GRAINGNN_DATA_DIR="$PWD"` before running the command; those dataset filenames are ignored by Git.

### GPU smoke test

To verify accelerator visibility and actual PyTorch and PyTorch Geometric execution, run the standalone smoke test from inside a GPU allocation. It fails rather than silently falling back to CPU if PyTorch cannot see a GPU:

```sh
python scripts/check_gpu.py
```

On Frontier, load the documented ROCm 7.2 modules and activate the GrainGNN environment as above, then run:

```sh
python scripts/check_gpu.py --require-rocm
```

The check reports the PyTorch/PyG versions and visible GPU, performs a synchronized PyTorch tensor operation, transfers a heterogeneous graph to the GPU, runs a PyG `HeteroConv` with `SAGEConv`, and checks the output and gradients. Run it inside a scheduler-provided GPU allocation; it is not part of the CPU-only pytest suite.

For the TACC end-to-end sample workflow, use the validation commands in [docs/env_tacc.md](env_tacc.md).
