"""python -m extract <source> [<source> ...] <out.pkl> [options]      (graingnn-extract ... after pip install)

GrainGNN training samples from solidified grain-id stacks. A source is
  an ExaCA run directory, or its input json    G, R, seed, the VTK file and the orientation csv from the json
  an ExaCA GrainID VTK file                    --G --R [--seed], --orientations
  a cuPF h5 file (alpha or cross_sec)          G, R, seed from the file name
  a pickle written by this command             included as it is
Each stack is tracked (stack_tracking), reduced to what GrainGNN represents (graingnn_view) and cut into samples
(samples); several sources become one dataset in a shuffled stack order (dataset).
"""
import argparse, os, sys
from multiprocessing import Pool
import numpy as np
from .exaca_reader import read_vtk, exaca_theta, read_run
from .cupf_reader import read_cupf, cupf_params
from .stack_tracking import track_stack
from .trajectory import smooth_tracks
from .graingnn_view import track_view, schedule_kinds
from .samples import build_samples, calibrate_span, span_choices, summary, Z_RANGE_UM
from .dataset import save_samples, load_samples, combine_samples


def load_source(src, a):
    """One source -> (Field with theta, G, R, seed, name), or the sample list of a pickle."""
    if src.endswith('.pkl'):
        return load_samples(src)
    G, R, seed, orientations, name = a.G, a.R, a.seed, a.orientations, os.path.splitext(os.path.basename(src))[0]
    if os.path.isdir(src) or src.endswith('.json'):
        run = read_run(src)
        G, R, seed, orientations, name, src = run['G'], run['R'], run['seed'], orientations or run['orientations'], run['name'], run['vtk']
    if src.endswith('.h5'):
        field = read_cupf(src)
        G, R, seed = (v if v is not None else d for v, d in zip(cupf_params(src), (G, R, seed)))
    else:
        if orientations is None:
            sys.exit(f'{src}: --orientations is required (or GrainOrientationFile found from the run)')
        field = read_vtk(src)
        field.theta = exaca_theta(field.ids, orientations, theta_x=a.theta_x)
    if G is None or R is None:
        sys.exit(f'{src}: --G and --R are required')
    return field, float(G), float(R), int(seed), name


def extract_stack(task):
    """Samples of one source (a (source, args) pair, for Pool.map)."""
    src, a = task
    loaded = load_source(src, a)
    if isinstance(loaded, list):
        return loaded
    field, G, R, seed, name = loaded
    raw = track_stack(field, verbose=a.verbose)
    view, _, schedule, painted = track_view(field, raw=raw)
    K, kinds = raw.n_layers, schedule_kinds(raw, schedule)
    ev = {str(k): int(v) for k, v in zip(*np.unique(view.events['kind'], return_counts=True))}
    print(f'{name}: {K} layers, {len(raw.gids)} grains; dead by lens/enclosed onset {kinds["onset"]}, by first gap {kinds["gap"]}, '
          f'absent from layer 0 {kinds["absent"]}, alive throughout {kinds["never"]}; painted cells {int(painted.sum())} '
          f'(max {int(painted.max())} in one layer)\n{name}: view {len(view.gids)} grains, {view.n_slots} slots, '
          f'valid {int(view.valid.sum())}/{K}, events {ev}')
    if a.smooth > 1:
        smooth_tracks(view, a.smooth)
    span = a.span or calibrate_span(view, span_choices(view.dz, K))
    samples = build_samples(view, G, R, span=span, seed=seed, z_range_um=a.z_range)
    print(f'{name}: span {span} layers = {span * view.dz:.2f} um; G {G} R {R} seed {seed}\n{name}: ' + summary(samples, view).replace('\n', f'\n{name}: '))
    if a.save_traj:
        os.makedirs(a.save_traj, exist_ok=True)
        raw.save(os.path.join(a.save_traj, name + '_raw.npz'))
        view.save(os.path.join(a.save_traj, name + '_view.npz'))
    return samples


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m extract', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('source', nargs='+', help='ExaCA run directory or input json, ExaCA VTK, cuPF h5, or sample pickle')
    p.add_argument('out', help='sample pickle (train.py --dataset NAME reads dataset_NAME.pkl)')
    p.add_argument('--orientations', help='ExaCA orientation csv (default: the GrainOrientationFile of the run)')
    p.add_argument('--theta-x', choices=('tilted', 'lateral'), default='tilted', help='ExaCA theta_x definition (exaca_theta)')
    p.add_argument('--G', type=float, help='K/um, for a VTK source')
    p.add_argument('--R', type=float, help='m/s, for a VTK source')
    p.add_argument('--seed', type=int, default=0, help='for a VTK source')
    p.add_argument('--span', type=int, help='layers (default: calibrated from the event rates of each stack)')
    p.add_argument('--smooth', type=int, default=0, help='smooth_tracks window in layers')
    p.add_argument('--z-range', type=float, default=Z_RANGE_UM, help='um, the height z and span are scaled by')
    p.add_argument('--save-traj', metavar='DIR', help='also write <DIR>/<name>_raw.npz and _view.npz per stack')
    p.add_argument('--shuffle-seed', type=int, default=35, help='stack order in the dataset (-1: the given order)')
    p.add_argument('--jobs', type=int, default=1, help='stacks extracted in parallel')
    p.add_argument('-v', '--verbose', action='store_true')
    a = p.parse_args(argv)
    tasks = [(src, a) for src in a.source]
    if a.jobs > 1:
        with Pool(a.jobs) as pool:
            stacks = pool.map(extract_stack, tasks)
    else:
        stacks = [extract_stack(t) for t in tasks]
    samples = combine_samples(stacks, seed=None if len(stacks) == 1 or a.shuffle_seed < 0 else a.shuffle_seed)
    save_samples(samples, a.out)
    print(f'{len(samples)} samples from {len(stacks)} sources -> {a.out}')


if __name__ == '__main__':
    main()
