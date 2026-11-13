"""Pixel cleanup of one grain-id slice: the smallest edits that make it representable as a trivalent polygon network.

remove_islands      every grain keeps its largest 4-connected piece; the other pieces are repainted from the
                    outside in, each cell taking the id of the surrounding grain it touches most
split_bad_corners   a corner where four boundary segments meet (four ids, or a grain touching itself across the
                    diagonal) cannot be a trivalent junction; one cell of its 2x2 block is changed to another id of
                    the block, the change that leaves the fewest such corners and has the least support against it
Both return (image, number of changed cells). No other cell is touched: this is not a smoothing filter.
On the ExaCA runs seen so far the two together change about 0.05% of the cells of a slice.
"""
import numpy as np
from .periodic_grid import roll, window, connected_components

SIDES = ((1, 0), (-1, 0), (0, 1), (0, -1))             # (dy, dx) offsets of the four 4-neighbours
DIAGONALS = ((1, 1), (1, -1), (-1, 1), (-1, -1))      # (dy, dx) offsets of the four diagonal neighbours


def island_mask(S, periodic):
    """True for every cell that is not in the largest 4-connected piece of its grain.
    Among equal-sized pieces the one whose first cell comes first in raster order is kept."""
    labels, n = connected_components(S, periodic)
    size = np.bincount(labels.ravel(), minlength=n)
    grain_of = np.empty(n, S.dtype)
    grain_of[labels.ravel()] = S.ravel()                       # the grain id of every piece
    by_grain_then_size = np.lexsort((-size, grain_of))          # pieces grouped by grain, largest first (stable on ties)
    sorted_grains = grain_of[by_grain_then_size]
    first_of_grain = np.r_[True, sorted_grains[1:] != sorted_grains[:-1]]
    keep = np.zeros(n, bool)
    keep[by_grain_then_size[first_of_grain]] = True
    return ~keep[labels]


def flood_islands(S, island, periodic):
    """Repaint the island cells from the outside in.

    In each pass every island cell that touches a stable cell (one that is not an island, or was repainted in an
    earlier pass) takes an id from its stable 4-neighbours: the id held by most of them; on a tie the id also
    held by most stable diagonal neighbours; on a second tie the smaller id. Cells deeper inside a piece wait for
    the next pass, so a multi-cell piece is shared between the grains around it in proportion to contact.
    """
    S = S.copy()
    island = island.copy()
    n_ids = int(S.max()) + 1                                   # ids are 0..n_ids-1; used to pack the tie-break key

    def neighbour_values(A, offsets):
        """A at the given (dy, dx) offsets, taken at the island cells: shape (n_island, len(offsets)).
        Positions outside a non-periodic image read -1."""
        shifted = []
        for dy, dx in offsets:
            R = roll(A, dy, 0, periodic) if dy else A
            shifted.append(roll(R, dx, 1, periodic) if dx else R)
        return np.stack(shifted, -1)[island]

    while island.any():
        stable = (~island).astype(np.int8)                     # 1 where the id is final
        sides = neighbour_values(S, SIDES)
        sides = np.where(neighbour_values(stable, SIDES) == 1, sides, -1)            # only stable ids may vote
        diagonals = neighbour_values(S, DIAGONALS)
        diagonals = np.where(neighbour_values(stable, DIAGONALS) == 1, diagonals, -1)
        # For each of the four side slots: how many stable sides and how many stable diagonals hold the same id.
        side_votes = (sides[:, :, None] == sides[:, None, :]).sum(-1)
        diagonal_votes = (sides[:, :, None] == diagonals[:, None, :]).sum(-1)
        # One integer per slot that sorts by (side votes, diagonal votes, smaller id): both vote counts are at
        # most 4, so side_votes * 5 + diagonal_votes orders them lexicographically; multiplying by n_ids and
        # adding (n_ids - 1 - id) then prefers the smaller id among equal votes.
        score = (side_votes * 5 + diagonal_votes) * n_ids + (n_ids - 1 - sides)
        score[sides < 0] = -1                                  # empty or unstable slots never win
        best_slot = score.argmax(1)
        ready = score.max(1) >= 0                              # island cells with at least one stable 4-neighbour
        if not ready.any():
            break
        yy, xx = np.where(island)
        new_ids = sides[np.arange(len(best_slot)), best_slot]
        S[yy[ready], xx[ready]] = new_ids[ready]
        island[yy[ready], xx[ready]] = False
    return S


def remove_islands(S, periodic, max_iter=10):
    """Keep the largest component of every grain and repaint the other pieces from the surrounding grains.

    Repainting joins every island cell to a kept piece, so new islands cannot appear; the loop normally runs
    twice (repaint, then confirm that nothing is left). max_iter only guards against a pathological image.
    Returns (cleaned image, number of repainted cells).
    """
    S = S.copy()
    changed = 0
    for _ in range(max_iter):
        island = island_mask(S, periodic)
        if not island.any():
            break
        changed += int(island.sum())
        S = flood_islands(S, island, periodic)
    return S, changed


def corner_blocks(S, periodic):
    """The 2x2 block of ids around every corner.

    Corner (i, j) is the grid vertex shared by cells (i, j), (i+1, j), (i, j+1), (i+1, j+1). The result has shape
    (H, W, 4) with those four ids along the last axis, -1 where the block reaches outside a non-periodic image.
    """
    here = S
    below = roll(S, -1, 0, periodic)
    right = roll(S, -1, 1, periodic)
    below_right = roll(below, -1, 1, periodic)
    return np.stack([here, below, right, below_right], -1)


def corner_degree(blocks):
    """Number of boundary segments meeting at a corner: 0 inside a grain, 2 on a boundary, 3 at a junction, 4 at a
    defect (four ids, or two ids in a checkerboard). blocks holds (here, below, right, below_right) on its last
    axis: the (H, W, 4) array of corner_blocks, or the four ids of one corner."""
    here, below, right, below_right = (blocks[..., k] for k in range(4))
    return (here != below).astype(int) + (below != below_right) + (below_right != right) + (right != here)


def split_bad_corners(S, periodic, max_pass=8):
    """Resolve every corner of degree 4 by changing one cell of its 2x2 block.

    The candidates are the changes of one block cell to another id of the block. The change kept is the one that
    leaves the fewest degree-4 corners around the changed cell (so a defect is not pushed back and forth between
    two corners, which happened on about 2% of ExaCA stacks); among those, the cell whose own id has the least
    support (fewest cells of that id) in its 3x3 neighbourhood, taking the id with the most support there.
    Support windows wrap only on periodic axes. A change can still create a new degree-4 corner next door, hence
    up to max_pass sweeps; one or two are needed in practice. Returns (image, number of changed cells).
    """
    S = S.copy()
    H, W = S.shape
    changed = 0

    def wrap(i, j):
        """Cell (i, j) brought into the image on the periodic axes; None when it lies outside on a walled axis."""
        i = i % H if periodic[1] else i
        j = j % W if periodic[0] else j
        return (i, j) if 0 <= i < H and 0 <= j < W else None

    def block(i, j):
        """The cells (here, below, right, below_right) of corner (i, j); None when one lies outside the image."""
        cells = [wrap(i, j), wrap(i + 1, j), wrap(i, j + 1), wrap(i + 1, j + 1)]
        return None if None in cells else cells

    def degree(cells):
        return corner_degree(np.array([S[c] for c in cells]))

    def support(cell, g):
        """Cells of id g among the neighbours of `cell` in its 3x3 window."""
        ys, xs = window(cell[0], H, periodic[1]), window(cell[1], W, periodic[0])
        return int((S[np.ix_(ys, xs)] == g).sum()) - int(S[cell] == g)

    def defects_around(cell):
        """Degree-4 corners among the four corners of `cell`."""
        i, j = cell
        blocks = (block(i + di, j + dj) for di in (-1, 0) for dj in (-1, 0))
        return sum(degree(b) == 4 for b in blocks if b is not None)

    def cost(cell, g):
        """Rank of changing `cell` to g: defects left around the cell, support of its own id, -support of g, g."""
        old, S[cell] = S[cell], g
        left = defects_around(cell)
        S[cell] = old
        return left, support(cell, old), -support(cell, g), g

    for _ in range(max_pass):
        blocks = corner_blocks(S, periodic)
        bad = np.argwhere((corner_degree(blocks) == 4) & (blocks.min(-1) >= 0))
        if not len(bad):
            break
        for i, j in bad:
            cells = block(i, j)
            if degree(cells) < 4:
                continue                                       # resolved by an earlier change in this sweep
            ids = {int(S[c]) for c in cells}
            cell, g = min(((c, g) for c in cells for g in ids - {int(S[c])}), key=lambda cg: cost(*cg))
            S[cell] = g
            changed += 1
    return S, changed
