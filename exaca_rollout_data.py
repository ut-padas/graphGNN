"""ExaCA stack -> GrainGNN rollout test data, the two files test.py reads from --truth_dir:
    seed<S>_G<G>_R<R>_span<N>.pkl   [hg0], the initial GrainHeterograph
    traj<S>.pkl                     graph_trajectory holding the ExaCA truth (areas, eliminations, cross-sections)
The stack is tracked with extract (layers 1-3 view), then resampled onto the legacy frame of reference: 0.08 um / 501 px
images (nearest neighbour) and 0.4 um frames from z = 2 um, so test.py, GNN_update and the QoIs run unchanged.
usage: python exaca_rollout_data.py test.vtk --orientations csv --G 1.9 --R 0.558 --seed 28 --save_dir graphs/exaca/
"""
import argparse, os, re
from collections import defaultdict
import numpy as np, dill
from scipy.interpolate import griddata
from extract.exaca_reader import read_vtk, exaca_theta
from extract.stack_tracking import track_stack
from extract.graingnn_view import track_view
from graph_trajectory import graph_trajectory

MESH, LXD, Z0, DZ, FRAMES = 0.08, 40.0, 2.0, 0.4, 121          # legacy test geometry: 501 px, frames at 2 + 0.4 j um


def resample(img, size):
    """Nearest-neighbour resize of a periodic (H, W) image to (size, size); pixel i covers [i*W/size, (i+1)*W/size)."""
    iy = (np.arange(size) * img.shape[0] / size).astype(int) % img.shape[0]
    ix = (np.arange(size) * img.shape[1] / size).astype(int) % img.shape[1]
    return img[np.ix_(iy, ix)]


def build(vtk, orientations, G, R, seed, theta_x='tilted', verbose=False):
    field = read_vtk(vtk)
    field.theta = exaca_theta(field.ids, orientations, theta_x=theta_x)
    raw = track_stack(field, verbose=verbose)
    view, vfield, schedule, painted = track_view(field, raw=raw)
    K, H, W = vfield.ids.shape
    assert H == W and field.periodic == (True, True), 'periodic square stacks only'
    size = int(LXD / MESH) + 1                                                      # 501
    layers = np.minimum(np.round((Z0 + DZ * np.arange(FRAMES) - field.z0) / field.dz).astype(int), K - 1)
    gids = view.gids
    row_of = np.full(int(gids.max()) + 1, 0, int); row_of[gids] = np.arange(1, len(gids) + 1)   # id -> 1..G
    G_n = len(gids)
    frames = np.stack([resample(row_of[vfield.ids[l]], size) for l in layers], axis=-1)       # (y, x, frame)
    totalV = np.zeros((G_n, FRAMES), int)
    counts = []
    for j in range(FRAMES):
        ids, n = np.unique(frames[:, :, j], return_counts=True)
        totalV[ids - 1, j] = n
        counts.append(dict(zip(ids.tolist(), n.tolist())))
    events = [set()] + [set(counts[j - 1]) - set(counts[j]) for j in range(1, FRAMES)]

    traj = graph_trajectory(lxd=LXD, randInit=False, seed=seed, frames=FRAMES, BC='periodic',
                            physical_params={'G': float(G), 'R': float(R)})
    k0 = int(layers[0])
    scale = size / W / (size - 1)                                   # cells of the stack -> legacy normalised units
    slots = np.flatnonzero(view.track[k0] >= 0)
    for j, s in enumerate(slots):
        traj.vertices[j] = np.array(view.pos[k0, s]) * scale
        traj.vertex2joint[j] = set(int(row_of[g]) for g in view.triple[k0, s])
    traj.joint2vertex = dict((tuple(sorted(v)), k) for k, v in traj.vertex2joint.items())
    traj.theta_x, traj.theta_z = np.zeros(1 + G_n), np.zeros(1 + G_n)
    th = np.nan_to_num(view.theta)
    traj.theta_x[1:], traj.theta_z[1:] = th[:, 0], th[:, 1]
    traj.num_regions, traj.num_vertices = G_n, len(slots)
    traj.alpha_pde_frames = np.ascontiguousarray(np.transpose(frames, (1, 0, 2)))     # (x, y, frame), as load_pde_data
    traj.alpha_pde = traj.alpha_pde_frames[:, :, 0].T
    traj.alpha_field = traj.alpha_pde.copy()
    traj.totalV_frames, traj.extraV_frames = totalV, np.zeros_like(totalV)
    traj.area_counts = counts[0]
    traj.area_traj = [counts[0]]
    traj.grain_events = events
    traj.period = np.array([size, size]) * MESH / LXD
    traj.raise_err = False
    traj.source = dict(vtk=os.path.abspath(vtk), cell=field.cell, dz=field.dz, layers=layers.tolist(),
                       schedule_painted=int(painted.sum()))
    slot_to_j = {int(s): j for j, s in enumerate(slots)}             # true boundaries from layer 2, both directions, as the
    for a, b in view.edge_slots[view.edges_of(k0)].tolist():        # legacy init would list them; its angular sort of each
        traj.edges += [[slot_to_j[a], slot_to_j[b]], [slot_to_j[b], slot_to_j[a]]]   # grain's vertices fails on non-convex grains
    traj.update(init=False)                                          # regions, centres, vertex_neighbor from the given edges
    traj.plot_polygons()                                             # polygon image and error_layer against the ExaCA slice
    traj.num_edges = len(traj.edges)
    bad = [v for v, n in traj.vertex_neighbor.items() if len(n) != 3]
    if bad:
        raise RuntimeError(f'{len(bad)} junctions without three neighbours: {bad[:5]}')
    traj.form_states_tensor(0)
    hg0 = traj.states[0]
    return traj, hg0, dict(layers=K, grains=G_n, junctions=len(slots), edges=len(traj.edges),
                           eliminations=sum(len(e) for e in events), error_layer=float(traj.error_layer))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('vtk'); p.add_argument('--orientations', required=True)
    p.add_argument('--G', type=float, required=True); p.add_argument('--R', type=float, required=True)
    p.add_argument('--seed', type=int, required=True); p.add_argument('--save_dir', default='./graphs/exaca/')
    p.add_argument('--theta-x', choices=('tilted', 'lateral'), default='tilted')
    p.add_argument('--span', type=int, help='frames per step (default: the training grid value at G, R)')
    p.add_argument('-v', '--verbose', action='store_true')
    a = p.parse_args()
    os.makedirs(a.save_dir, exist_ok=True)
    traj, hg0, info = build(a.vtk, a.orientations, a.G, a.R, a.seed, a.theta_x, a.verbose)
    if a.span:
        hg0.span = a.span
    else:
        grid = dill.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GR_train_grid.pkl'), 'rb'))
        G_ = (a.G - grid['G_min']) / (grid['G_max'] - grid['G_min']); R_ = (a.R - grid['R_min']) / (grid['R_max'] - grid['R_min'])
        hg0.span = int(griddata(np.array([grid['G'], grid['R']]).T, np.array(grid['span']), (G_, R_), method='nearest'))
    hg0.form_gradient(prev=None, nxt=None, event_list=None, elim_list=None)
    hg0.append_history([])
    name = f'seed{a.seed}_G{round(a.G, 3)}_R{round(a.R, 3)}_span{hg0.span}.pkl'
    with open(os.path.join(a.save_dir, name), 'wb') as f:
        dill.dump([hg0], f)
    with open(os.path.join(a.save_dir, f'traj{a.seed}.pkl'), 'wb') as f:
        dill.dump(traj, f)
    print(f'{info}; span {hg0.span}; grain features {hg0.feature_dicts["grain"].shape}, joint {hg0.feature_dicts["joint"].shape}; wrote {name}, traj{a.seed}.pkl in {a.save_dir}')


if __name__ == '__main__':
    main()
