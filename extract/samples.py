"""GrainGNN training samples from a tracked stack (layer 3, part 2).

A sample is the state of one layer k with the labels of layer k + span, in the attribute layout of
graph_datastruct.GrainHeterograph so that train.py and data_loader.py read it unchanged:
  features   grain: x, y, z, area, cosx, sinx, cosz, sinz, span, darea     joint: x, y, z, G, R, span, dx, dy
  targets    grain: darea (x20)                                            joint: dx, dy (x5)
  events     grain_event: the grain is gone at k + span.  edge_event: the two junctions are still joined at
             k + span by a boundary between a different pair of grains while both grains of the old pair still
             exist (1, a neighbour switch), still joined otherwise (0: same pair, or the pair changed because one
             of its grains was eliminated and a neighbour took over the boundary, which the grain event covers),
             or no longer joined (-100).  edge: signed change of the straight edge length (x2.5), negative
             through a switch.
  masks      grain: has cells at k.  joint: the slot holds the same junction with the same triple at k and
             k + span.  edge: edge_event >= 0.
  edges      grain -> joint (push) and joint -> grain (pull) for the three grains of every junction, weighted by
             the junction-to-centroid distance; joint -> joint (connect) in both directions, weighted by the
             straight length.
Units: positions in the unit square (cells / image size), area as a fraction of the slice, z and span as
fractions of Z_RANGE_UM (the height of the paper's trajectories: 120 frames of 0.4 um), G and R as 1 - G/10 and
R/2 as in the paper. Displacements are the shortest periodic difference and positions stay in the unit square.
Grain ids inside a sample are 1..G in the row order of the Trajectory; `gids` keeps the original ids. The history
columns (darea, dx, dy) hold the change over the previous span, zero for a junction that was not there. A junction
that is gone at k + span carries the legacy signature target -5 * (x, y), i.e. a zero next row, and is masked.
One span per stack, chosen by the paper's rule unless given: the largest choice for which a window of that
height holds on average fewer than 9 switches and 4.5 eliminations.
"""
import numpy as np
from dataclasses import dataclass
from types import SimpleNamespace
from .periodic_grid import wrap_delta

GRAIN_FEATURES = ['x', 'y', 'z', 'area', 'cosx', 'sinx', 'cosz', 'sinz', 'span']
JOINT_FEATURES = ['x', 'y', 'z', 'G', 'R', 'span']
HISTORY = {'grain': ['darea'], 'joint': ['dx', 'dy']}
TARGETS = {'grain': ['darea'], 'joint': ['dx', 'dy']}
SCALING = {'grain': 20, 'joint': 5}
EDGE_TYPES = [('grain', 'push', 'joint'), ('joint', 'pull', 'grain'), ('joint', 'connect', 'joint')]
Z_RANGE_UM = 48.0
SPAN_CHOICES_UM = (2.4, 3.2, 4.0, 4.8, 6.0, 8.0, 9.6, 12.0, 16.0, 24.0, 48.0)     # the paper's 6..120 frames
NO_EDGE = -100


def new_sample(span):
    """One training sample: a plain namespace (so the pickle loads without this package) whose attribute names
    follow graph_datastruct.GrainHeterograph."""
    return SimpleNamespace(
        features={'grain': GRAIN_FEATURES + HISTORY['grain'], 'joint': JOINT_FEATURES + HISTORY['joint']},
        targets={k: list(v) for k, v in TARGETS.items()},
        events={'grain_event': 'elimination', 'edge_event': 'rotation'},
        edge_type=list(EDGE_TYPES), targets_scaling=dict(SCALING),
        feature_dicts={}, target_dicts={}, edge_index_dicts={}, edge_weight_dicts={},
        mask={}, additional_features={}, physical_params={}, vertex2joint={}, edges=[], gids=None, span=int(span))


def unit_delta(b, a, periodic):
    """Shortest difference b - a in the unit square, per axis where periodic."""
    d = np.asarray(b, float) - np.asarray(a, float)
    d[..., 0] = wrap_delta(d[..., 0], 1.0, periodic[0])
    d[..., 1] = wrap_delta(d[..., 1], 1.0, periodic[1])
    return d


def span_choices(dz, n_layers, choices_um=SPAN_CHOICES_UM):
    """The paper's span choices converted to layers of spacing dz, those shorter than the stack."""
    layers = sorted({max(1, int(round(um / dz))) for um in choices_um})
    return [s for s in layers if s < n_layers] or [1]


def calibrate_span(traj, choices):
    """Largest choice whose window holds on average fewer than 9 switches and 4.5 eliminations, else the smallest.
    The legacy rule (graph_trajectory.py train mode), with its event counts per 120 frames turned into rates."""
    kinds = traj.events['kind']
    n_switch, n_elim = int((kinds == 'switch').sum()), int((kinds == 'eliminate').sum())
    T = max(traj.n_layers - 1, 1)
    ok = [s for s in choices if n_switch * s / T < 9 and n_elim * s / T < 4.5]
    return max(ok) if ok else choices[0]


@dataclass
class Prepared:
    """A Trajectory in sample units: positions and centroids in the unit square, areas as fractions of the slice."""
    traj: object
    pos: np.ndarray            # (K, J, 2), smoothed when available
    cen: np.ndarray            # (K, G, 2)
    area: np.ndarray           # (K, G)
    occupied: np.ndarray       # (K, J) bool
    alive: np.ndarray          # (K, G) bool
    orient: np.ndarray         # (G, 4) cosx, sinx, cosz, sinz; zeros where the orientation is unknown
    z: np.ndarray              # (K,) height as a fraction of z_range_um
    row_of: dict               # grain id -> row


def prepare(traj, z_range_um, use_smooth):
    H, W = traj.shape
    size = np.array([W, H], float)
    P = traj.pos_smooth if use_smooth and traj.pos_smooth is not None else traj.pos
    tx, tz = traj.theta[:, 0], traj.theta[:, 1]
    return Prepared(traj, P / size, traj.centroid / size, traj.area / float(H * W), traj.occupied(), traj.alive(),
                    np.nan_to_num(np.column_stack([np.cos(tx), np.sin(tx), np.cos(tz), np.sin(tz)])),
                    np.arange(traj.n_layers) * traj.dz / z_range_um, {int(g): r for r, g in enumerate(traj.gids.tolist())})


def grain_state(P, k, p, span_feature):
    """(G, 10) grain features of layer k; p is the history layer or None."""
    live = P.alive[k]
    grain = np.zeros((len(P.row_of), len(GRAIN_FEATURES) + len(HISTORY['grain'])))
    grain[live, 0:2] = P.cen[k, live]
    grain[:, 2], grain[:, 3], grain[:, 8] = P.z[k], P.area[k], span_feature
    grain[live, 4:8] = P.orient[live]
    if p is not None:
        grain[:, 9] = SCALING['grain'] * (P.area[k] - P.area[p])
    return grain


def joint_state(P, k, p, G, R, span_feature):
    """(J, 8) junction features of layer k: position, z, G, R, span and the displacement over the previous span
    for a junction that was already there at p."""
    occ, t = P.occupied[k], P.traj.track
    joint = np.zeros((P.pos.shape[1], len(JOINT_FEATURES) + len(HISTORY['joint'])))
    joint[occ, 0:2] = P.pos[k, occ]
    joint[:, 2], joint[:, 3], joint[:, 4], joint[:, 5] = P.z[k], 1 - G / 10, R / 2, span_feature
    if p is not None:
        same = occ & P.occupied[p] & (t[k] == t[p])
        joint[same, 6:8] = SCALING['joint'] * unit_delta(P.pos[k, same], P.pos[p, same], P.traj.periodic)
    return joint


def joint_targets(P, k, n):
    """Displacement targets (J, 2), the junction mask (J,), and `stays`: the slots holding the same junction at k
    and n. A junction gone at n gets the legacy signature -5 * (x, y), a zero next row."""
    t, tri = P.traj.track, P.traj.triple
    occ = P.occupied[k]
    stays = occ & P.occupied[n] & (t[k] == t[n])
    target = np.zeros((P.pos.shape[1], 2))
    target[stays] = SCALING['joint'] * unit_delta(P.pos[n, stays], P.pos[k, stays], P.traj.periodic)
    gone = occ & ~stays
    target[gone] = -SCALING['joint'] * P.pos[k, gone]
    mask = stays & np.all(tri[k] == tri[n], axis=1)
    return target, mask, stays


def edge_labels(P, k, n, stays):
    """Connect edges of layer k in both directions: index (2, 2E), weights (2E, 1), and per edge the event, the
    signed length-change target and the mask (rules in the module docstring)."""
    traj = P.traj
    e = traj.edges_of(k)
    slots, pairs = traj.edge_slots[e], traj.edge_pair[e]
    joined = {}
    for (a, b), pr in zip(traj.edge_slots[traj.edges_of(n)].tolist(), traj.edge_pair[traj.edges_of(n)].tolist()):
        joined.setdefault(frozenset((a, b)), set()).add(tuple(pr))
    E = len(slots)
    event, target, mask = np.full(E, NO_EDGE, int), np.zeros(E), np.zeros(E, int)
    length = np.hypot(*unit_delta(P.pos[k, slots[:, 1]], P.pos[k, slots[:, 0]], traj.periodic).T) if E else np.zeros(0)
    for i, ((a, b), pr) in enumerate(zip(slots.tolist(), pairs.tolist())):
        key = frozenset((a, b))
        if stays[a] and stays[b] and key in joined:
            switched = tuple(pr) not in joined[key] and bool(P.alive[n, P.row_of[pr[0]]] and P.alive[n, P.row_of[pr[1]]])
            length_n = float(np.hypot(*unit_delta(P.pos[n, b], P.pos[n, a], traj.periodic)))
            event[i], mask[i] = int(switched), 1
            target[i] = 0.5 * SCALING['joint'] * ((-length_n if switched else length_n) - length[i])
    both = lambda x: np.concatenate([x, x])
    index = np.vstack([np.concatenate([slots[:, 0], slots[:, 1]]), np.concatenate([slots[:, 1], slots[:, 0]])]).astype(np.int64)
    return index, both(length)[:, None], both(event), both(target), both(mask)


def grain_joint_edges(P, k):
    """Push edges, grain row -> slot, for the three grains of every junction of layer k, weighted by the
    junction-to-centroid distance. The pull edges are the same rows reversed."""
    js = np.flatnonzero(P.occupied[k])
    g_rows = np.array([P.row_of[int(g)] for j in js for g in P.traj.triple[k, j]], int)
    j_rows = np.repeat(js, 3)
    w = np.hypot(*unit_delta(P.pos[k, j_rows], P.cen[k, g_rows], P.traj.periodic).T) if len(js) else np.zeros(0)
    return np.vstack([g_rows, j_rows]).astype(np.int64), w[:, None]


def build_samples(traj, G, R, span=None, stride=None, seed=0, z_range_um=Z_RANGE_UM, use_smooth=True):
    """Samples of every snapshot k = 0, stride, 2 stride, ... with k + span inside the stack (see module docstring).

    span      layers (default: calibrate_span);  stride   layers between snapshots (default span // 2)
    A snapshot is skipped when layer k, k + span or the history layer k - span is not valid.
    """
    K = traj.n_layers
    span = int(span) if span else calibrate_span(traj, span_choices(traj.dz, K))
    stride = max(1, int(stride) if stride else span // 2)
    span_feature = span * traj.dz / z_range_um
    P = prepare(traj, z_range_um, use_smooth)
    samples = []
    for k in range(0, K - span, stride):
        n, p = k + span, (k - span if k - span >= 0 else None)
        if not (traj.valid[k] and traj.valid[n]) or (p is not None and not traj.valid[p]):
            continue
        target, mask, stays = joint_targets(P, k, n)
        connect, connect_w, edge_event, edge_target, edge_mask = edge_labels(P, k, n, stays)
        push, push_w = grain_joint_edges(P, k)
        live = P.alive[k]
        s = new_sample(span)
        s.feature_dicts = {'grain': grain_state(P, k, p, span_feature), 'joint': joint_state(P, k, p, G, R, span_feature)}
        s.target_dicts = {'grain': (SCALING['grain'] * (P.area[n] - P.area[k]))[:, None], 'joint': target,
                          'grain_event': (live & ~P.alive[n]).astype(int), 'edge_event': edge_event, 'edge': edge_target}
        s.mask = {'grain': live.astype(int)[:, None], 'joint': mask.astype(int)[:, None], 'edge': edge_mask}
        s.edge_index_dicts = {EDGE_TYPES[0]: push, EDGE_TYPES[1]: push[::-1].copy(), EDGE_TYPES[2]: connect}
        s.edge_weight_dicts = {EDGE_TYPES[0]: push_w, EDGE_TYPES[1]: push_w, EDGE_TYPES[2]: connect_w}
        s.vertex2joint = {int(j): tuple(P.row_of[int(g)] + 1 for g in traj.triple[k, j]) for j in np.flatnonzero(P.occupied[k])}
        s.edges = connect.T.tolist()
        s.gids = traj.gids.copy()
        s.physical_params = {'G': float(G), 'R': float(R), 'seed': int(seed), 'height': int(k)}
        samples.append(s)
    return samples


def summary(samples, traj=None):
    """One paragraph of counts and label statistics (pixels where the trajectory is given)."""
    if not samples:
        return 'no samples'
    jm = np.concatenate([s.mask['joint'][:, 0] for s in samples]).astype(bool)
    present = np.concatenate([np.array([j in s.vertex2joint for j in range(len(s.mask['joint']))]) for s in samples])
    T = np.concatenate([s.target_dicts['joint'] for s in samples])[jm]
    ev = np.concatenate([s.target_dicts['edge_event'] for s in samples])
    ge = np.concatenate([s.target_dicts['grain_event'] for s in samples])
    lines = [f'{len(samples)} samples, span {samples[0].span} layers; junction rows present {present.sum()}, '
             f'masked in {jm.sum()} ({100 * jm.sum() / max(present.sum(), 1):.1f}%); grain eliminations {int(ge.sum())}; '
             f'edges {len(ev) // 2}: switching {int((ev == 1).sum()) // 2}, kept {int((ev == 0).sum()) // 2}, gone {int((ev < 0).sum()) // 2}']
    if traj is not None:
        px = SCALING['joint'] / traj.shape[1]
        d = np.abs(T) / px
        lines.append(f'|dx|,|dy| of masked-in junctions: RMS {np.sqrt((d ** 2).mean()):.2f} px, median {np.median(d):.1f}, '
                     f'p99 {np.percentile(d, 99):.1f}, max {d.max():.1f}')
    return '\n'.join(lines)

