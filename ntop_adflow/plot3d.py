"""Formatted PLOT3D multiblock surface I/O.

pyHyp reads a structured surface as a formatted PLOT3D file and needs
`fileType: PLOT3D` set explicitly. Without that option it tries to parse the file as
CGNS and dies with "Bad integer for item 1 in list input".

Layout written here, which is what pyHyp expects:

    nblocks
    ni nj nk        (one line per block)
    <all x for block 0> <all y> <all z>  <block 1 ...>

Each coordinate block is written with i varying fastest.
"""
import numpy as np


def write(path, patches):
    """Write a list of (ni, nj, 3) arrays as a formatted PLOT3D surface."""
    with open(path, "w") as f:
        f.write("%d\n" % len(patches))
        for P in patches:
            f.write("%d %d 1\n" % (P.shape[0], P.shape[1]))
        for P in patches:
            for c in range(3):
                vals = P[..., c].T.ravel(order="C")
                for k in range(0, len(vals), 6):
                    f.write(" ".join("%.10e" % v for v in vals[k:k + 6]) + "\n")


def read(path):
    """Read a formatted PLOT3D surface into a list of (ni, nj, 3) arrays."""
    tok = open(path).read().split()
    n = int(tok[0])
    k = 1
    dims = []
    for _ in range(n):
        dims.append((int(tok[k]), int(tok[k + 1]), int(tok[k + 2])))
        k += 3
    out = []
    for (ni, nj, nk) in dims:
        cnt = ni * nj * nk
        P = np.empty((ni, nj, 3))
        for c in range(3):
            P[..., c] = np.array(tok[k:k + cnt], dtype=float).reshape(nj, ni).T
            k += cnt
        out.append(P)
    return out


def read_nodes(path):
    """Read a PLOT3D surface as one flat (nnodes, 3) array, patches concatenated."""
    return np.concatenate([P.reshape(-1, 3) for P in read(path)])
