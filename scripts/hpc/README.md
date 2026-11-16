# GrainGNN HPC Installation

HPC installation scripts are organized by facility and system. Run an installer from the repository root; each installer creates its environment in a separate installation directory so it does not replace another setup.

| Facility / system | Installer or guide | Accelerator stack |
| --- | --- | --- |
| OLCF Frontier | [install-rocm72.sh](olcf/frontier/installation/install-rocm72.sh) | ROCm 7.2, PyTorch 2.14.0 |
| ALCF Aurora | [install-xpu.sh](alcf/aurora/installation/install-xpu.sh) | Intel XPU, PyTorch from the ALCF `frameworks` module |
| TACC Vista and Stampede3 | [docs/env_tacc.md](../../docs/env_tacc.md) | Existing CUDA setup; unchanged |

The shared [install_dependencies.sh](../../install_dependencies.sh) installs PyTorch Geometric and only GrainGNN's core requirements into the active Python environment. HPC installers install their facility-compatible PyTorch build first, then invoke this shared script. VTK and other optional tools are not installed by default.