"""Readers for ExaCA output: the GrainID field (legacy VTK) and the orientation table."""
import numpy as np
from .grain_field import Field


def read_vtk(path, array='GrainID'):
    """Legacy VTK STRUCTURED_POINTS written by ExaCA (ASCII or BINARY). Returns a Field with ids[z, y, x] in cells
    of `cell` um. Spacing is converted from metres when the file uses them."""
    from vtk import vtkDataSetReader
    from vtk.util.numpy_support import vtk_to_numpy
    reader = vtkDataSetReader()
    reader.SetFileName(str(path))
    reader.ReadAllScalarsOn()
    reader.Update()
    data = reader.GetOutput()
    nx, ny, nz = data.GetDimensions()
    if nx * ny * nz == 0:
        raise IOError(f'{path}: not a structured-points VTK file or empty')
    arr = data.GetPointData().GetArray(array)
    if arr is None:
        names = [data.GetPointData().GetArrayName(i) for i in range(data.GetPointData().GetNumberOfArrays())]
        raise KeyError(f'{path}: no point array {array!r}; found {names}')
    ids = vtk_to_numpy(arr).reshape(nz, ny, nx).astype(np.int32)          # VTK lists x fastest
    sx, sy, sz = data.GetSpacing()
    to_um = 1e6 if sx < 1e-3 else 1.0                                      # ExaCA writes metres, cuPF conversions um
    return Field(ids=ids, cell=float(sx * to_um), dz=float(sz * to_um), source=str(path))


def exaca_theta(ids, orientation_csv, n_orient=None, theta_x='tilted'):
    """theta_x, theta_z per grain id from ExaCA's orientation table (rows = the three <100> unit vectors), in radians
    as in graph_datastruct.  theta_z = angle between the <100> axis nearest to the build direction and z (0..54.7 deg).
    theta_x = azimuth modulo pi/2 of that tilted axis ('tilted', default; in cuPF this coincides with the azimuth of
    the lateral <100> axes) or of the <100> axis nearest to the plane ('lateral', the in-plane rotation that sets the
    octahedron's lateral reach; trained no better on the 0.1 um pilot, 2026-11-03)."""
    ori = np.loadtxt(orientation_csv, delimiter=',', skiprows=1).reshape(-1, 3, 3)
    n_orient = n_orient or len(ori)
    theta = {}
    for g in np.unique(ids):
        if g == 0:
            continue
        axes = ori[(abs(int(g)) - 1) % n_orient]            # 3 unit vectors
        a = axes[np.argmax(np.abs(axes[:, 2]))]
        if a[2] < 0:
            a = -a
        b = a if theta_x == 'tilted' else axes[np.argmin(np.abs(axes[:, 2]))]
        theta[int(g)] = (float(np.arctan2(b[1], b[0]) % (np.pi / 2)), float(np.arccos(min(1.0, a[2]))))
    return theta

