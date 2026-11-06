"""Tests for extract.slice_cleanup and extract.periodic_grid. Run with pytest or `python tests/test_slice_cleanup.py`."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from extract.periodic_grid import connected_components
from extract.slice_cleanup import remove_islands, split_bad_corners, corner_blocks, corner_degree


def test_components_join_across_periodic_seams_only():
    S = np.full((10, 10), 2, np.int32); S[:, :2] = 1; S[:, -1] = 1          # grain 1 wraps across the x seam
    lab, n = connected_components(S, (True, True)); assert n == 2 and lab[0, 0] == lab[0, -1]
    lab, n = connected_components(S, (False, False)); assert n == 3 and lab[0, 0] != lab[0, -1]


def test_remove_islands_keeps_largest_piece_and_single_cell_grains():
    S = np.ones((12, 12), np.int32); S[:, 6:] = 2
    S[2:4, 8:10] = 1                                                          # 4-cell piece of grain 1 inside grain 2
    S[9, 1] = 3                                                               # a grain that is one cell: not an island
    C, n = remove_islands(S, (False, False))
    assert n == 4 and (C[2:4, 8:10] == 2).all() and C[9, 1] == 3
    assert (C[:, :6] != 2).all() and (C[:, 6:] == 2).all()


def test_island_across_periodic_seam_is_one_piece():
    S = np.full((10, 10), 2, np.int32); S[:, :2] = 1; S[:, -1] = 1
    C, n = remove_islands(S, (True, True)); assert n == 0 and np.array_equal(C, S)
    C, n = remove_islands(S, (False, False)); assert n == 10 and (C[:, -1] == 2).all() and (C[:, :2] == 1).all()


def test_split_bad_corners_does_not_wrap_on_a_walled_axis():
    A, B, C, D = 1, 2, 3, 4
    S = np.full((6, 6), D, np.int32); S[0, :] = B; S[1:5, 0] = C; S[0, 0] = A; S[5, :] = A
    # corner (0, 0) has degree 4. Clipped windows: A at (0, 0) has no support and is the cell to change.
    # A wrapped window would count row 5 for cell (0, 0) and change cell (1, 0) instead.
    out, n = split_bad_corners(S, (False, False))
    assert n == 1 and out[0, 0] != A and out[1, 0] == C and (out[5, :] == A).all()
    blk = corner_blocks(out, (False, False)); assert not ((corner_degree(blk) == 4) & (blk.min(-1) >= 0)).any()


def test_split_bad_corners_periodic_four_ids():
    S = np.zeros((8, 8), np.int32); S[:4, :4] = 1; S[:4, 4:] = 2; S[4:, :4] = 3; S[4:, 4:] = 4
    out, n = split_bad_corners(S, (True, True))
    assert n >= 1 and not ((corner_degree(corner_blocks(out, (True, True))) == 4)).any()


def test_split_bad_corners_does_not_ping_pong():
    # 6x6 crop of an ExaCA slice (campaign stack G3.02_V2.0_s1133, layer 57): the support rule alone pushed the
    # degree-4 corner back and forth between two corners, one change per sweep, and left it unresolved.
    S = np.array([[5, 4, 4, 4, 3, 3], [5, 4, 4, 3, 3, 6], [5, 5, 4, 6, 6, 6],
                  [5, 5, 2, 1, 6, 6], [5, 2, 2, 1, 1, 6], [2, 2, 2, 1, 1, 1]], np.int32)
    out, n = split_bad_corners(S, (False, False))
    blk = corner_blocks(out, (False, False))
    assert n == 1 and not ((corner_degree(blk) == 4) & (blk.min(-1) >= 0)).any()


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
