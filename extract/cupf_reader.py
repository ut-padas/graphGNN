"""cuPF readers, kept for validation only: the seed-10020 cuPF stack is the one dataset where the legacy
extractor provides ground truth (junction counts, positions, events, the 72/75 and 0.1128 rollout), and cuPF 3D
fields are how additional phase-field test cases enter the same pipeline. Not needed to process ExaCA data.
"""
from pathlib import Path
import numpy as np
from .grain_field import Field


def read_cupf_alpha(path):
    """cuPF 3D field file (*_time*.h5): dataset `alpha` in Fortran order with a one-cell halo. ids[z, y, x], cut
    below the first plane that still holds liquid (id 0), so only fully solidified cross-sections remain."""
    import h5py
    f = h5py.File(path, 'r')
    x, y, z = (np.asarray(f[k]) for k in ('x_coordinates', 'y_coordinates', 'z_coordinates'))
    A = np.asarray(f['alpha']).reshape((len(x), len(y), len(z)), order='F')[1:-1, 1:-1, 1:-1]   # x, y, z interior
    ids = np.ascontiguousarray(np.transpose(A, (2, 1, 0))).astype(np.int32)                  # z, y, x
    liquid = np.flatnonzero((ids == 0).any(axis=(1, 2)))
    if liquid.size:
        ids = ids[:liquid[0]]
    cell = float(x[1] - x[0])
    return Field(ids=ids, cell=cell, dz=float(z[1] - z[0]), source=str(path), z0=float(z[1]))


def read_cupf_cross_sec(path, dz=0.4, z0=2.0):
    """cuPF QoI file: `cross_sec` frames (the solidification-front cross-section every dz um). ids[frame, y, x]."""
    import h5py, re
    f = h5py.File(path, 'r')
    x = np.asarray(f['x_coordinates'])
    fnx = len(x)
    frames = int(re.search(r'frames(\d+)', Path(path).name).group(1)) + 1
    A = np.asarray(f['cross_sec']).reshape((fnx, fnx, frames), order='F')[1:-1, 1:-1, :]      # x, y, frame
    ids = np.ascontiguousarray(np.transpose(A, (2, 1, 0))).astype(np.int32)                   # frame, y, x
    return Field(ids=ids, cell=float(x[1] - x[0]), dz=dz, source=str(path), z0=z0)


def cupf_theta(angles):
    """cuPF `angles` dataset (length 2N+1): [0, theta_x[1..N], theta_z[1..N]] -> dict grain id -> (theta_x, theta_z),
    folded into [0, pi/2) as in the GrainGNN training data (newer cuPF drivers store them modulo pi; the cubic
    anisotropy is invariant under the quarter turn)."""
    a = np.asarray(angles) % (np.pi / 2)
    n = (len(a) - 1) // 2
    return {g: (float(a[g]), float(a[n + g])) for g in range(1, n + 1)}
