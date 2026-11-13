"""Tests for extract.dataset, extract.exaca_reader.read_run and the command line (python -m extract).
Run with pytest or `python tests/test_cli.py`."""
import sys, os, json, pickle, types, tempfile, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from test_stack_tracking import moving_seeds_stack
from extract.grain_field import Field
from extract.stack_tracking import track_stack
from extract.samples import build_samples
from extract.dataset import save_samples, load_samples, combine_samples
from extract.exaca_reader import read_run
from extract.__main__ import main as cli


def samples_of(seed):
    traj = track_stack(Field(ids=moving_seeds_stack(speed=0.5, seed=seed), cell=1.0, dz=1.0))
    return build_samples(traj, 2.0, 1.0, span=3, seed=seed)


def write_exaca_run(d, ids, seed, G=1.9, V=0.558):
    """A synthetic ExaCA run directory: input.json, test.vtk (ASCII, spacing in metres) and an orientation csv."""
    os.makedirs(d, exist_ok=True)
    K, H, W = ids.shape
    with open(os.path.join(d, 'test.vtk'), 'w') as f:
        f.write('# vtk DataFile Version 3.0\nsynthetic\nASCII\nDATASET STRUCTURED_POINTS\n'
                f'DIMENSIONS {W} {H} {K}\nORIGIN 0 0 0\nSPACING 2.5e-07 2.5e-07 2.5e-07\nPOINT_DATA {ids.size}\n'
                'SCALARS GrainID int 1\nLOOKUP_TABLE default\n')
        np.savetxt(f, ids.reshape(-1), fmt='%d')
    from scipy.spatial.transform import Rotation
    np.savetxt(os.path.join(d, 'orient.csv'), Rotation.random(10, random_state=0).as_matrix().reshape(10, 9), delimiter=',', header='10', comments='')
    with open(os.path.join(d, 'input.json'), 'w') as f:
        json.dump({'RandomSeed': seed, 'GrainOrientationFile': 'orient.csv', 'TemperatureData': {'G': G * 1e6, 'R': G * 1e6 * V},
                   'Printing': {'PathToOutput': './', 'OutputFile': 'test'}}, f)
    return d


def test_sample_pickle_needs_no_package(tmp_path=None):
    d = str(tmp_path or tempfile.mkdtemp())
    S = samples_of(1)
    assert all(type(s) is types.SimpleNamespace for s in S)
    assert b'extract' not in pickle.dumps(S)                       # no reference to this package inside the pickle
    save_samples(S, os.path.join(d, 'a.pkl')); T = load_samples(os.path.join(d, 'a.pkl'))
    assert len(T) == len(S) and T[0].span == S[0].span and T[0].features == S[0].features
    assert np.array_equal(T[0].feature_dicts['joint'], S[0].feature_dicts['joint'])


def test_combine_shuffles_stacks_with_a_fixed_seed():
    stacks = [samples_of(seed) for seed in (1, 2, 3)]
    D = combine_samples(stacks)
    seeds = [s.physical_params['seed'] for s in D]
    assert len(D) == sum(map(len, stacks)) and sorted(set(seeds)) == [1, 2, 3]
    assert seeds == [s.physical_params['seed'] for s in combine_samples(stacks)]                 # deterministic
    assert [s.physical_params['seed'] for s in combine_samples(stacks, seed=None)] == sorted(seeds)   # given order kept


def test_read_run_converts_exaca_units(tmp_path=None):
    d = write_exaca_run(os.path.join(str(tmp_path or tempfile.mkdtemp()), 'run7'), moving_seeds_stack(speed=0.5), seed=7)
    run = read_run(d)
    assert run['seed'] == 7 and abs(run['G'] - 1.9) < 1e-9 and abs(run['R'] - 0.558) < 1e-9 and run['name'] == 'run7'
    assert os.path.isfile(run['vtk']) and run['orientations'] == os.path.join(d, 'orient.csv')
    assert read_run(os.path.join(d, 'input.json')) == run


def test_cli_exaca_runs_and_pickles_to_one_dataset(tmp_path=None):
    d = str(tmp_path or tempfile.mkdtemp())
    runs = [write_exaca_run(os.path.join(d, f'run{seed}'), moving_seeds_stack(speed=0.5, seed=seed), seed=seed) for seed in (7, 8)]
    cli([runs[0], os.path.join(d, 'one.pkl'), '--span', '3'])
    S = load_samples(os.path.join(d, 'one.pkl'))
    assert len(S) > 0 and S[0].physical_params['seed'] == 7 and abs(S[0].physical_params['R'] - 0.558) < 1e-9
    assert all(np.isfinite(s.feature_dicts['grain']).all() for s in S)       # orientations were found and applied
    save_samples(samples_of(3), os.path.join(d, 'three.pkl'))
    cli([runs[0], runs[1], os.path.join(d, 'three.pkl'), os.path.join(d, 'all.pkl'), '--span', '3', '--jobs', '2'])
    A = load_samples(os.path.join(d, 'all.pkl'))
    seeds = [s.physical_params['seed'] for s in A]
    assert sorted(set(seeds)) == [3, 7, 8] and seeds.count(7) == len(S)
    cli([runs[0], runs[1], os.path.join(d, 'three.pkl'), os.path.join(d, 'again.pkl'), '--span', '3'])
    assert [s.physical_params['seed'] for s in load_samples(os.path.join(d, 'again.pkl'))] == seeds   # same order, jobs or not


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
