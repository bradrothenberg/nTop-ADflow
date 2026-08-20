"""Cubed-sphere structured surface for a CLOSED slender body, and its derivative.

Why a cubed sphere. ADflow reads structured multiblock or overset CGNS only, and pyHyp
needs a structured surface seed. A closed body cannot use a single polar patch: the nose
and the tail collapse to a point, which gives zero-area cells. Two earlier topologies
failed on exactly this. A half body with collapsed nose and tail edges produced a
"free corner" topology error. A periodic single patch segfaulted with 12,384 zero-area
cells and infinite metrics. The cubed sphere has six patches built as rotations of one
base face, so shared edges coincide exactly, and with an EVEN number of points per edge
no grid point lands on a pole.

Parameterization. The body is a stack of ellipses along x:

    xi = (1 - dx) / 2            fixed per node, from the cubed-sphere direction
    x  = x_nose + L * xi
    rz = Sz * rz0(xi)
    ry = Sy * ratio0 * rz0(xi)

rz0 is a spline through a normalized radius table, so the profile stretches with L instead
of being truncated. Node (patch, i, j) always maps to the same sphere direction, so node
correspondence between two parameter values is exact by construction. That is what makes
dX_surf/dp well defined node by node, and it is the property the adjoint chain depends on.

SCOPE. This generator covers bodies of revolution with elliptical cross-sections: a
fuselage, a store, a nacelle centrebody. It does NOT make a wing, a fin or a full
aircraft. For those, build one near-body mesh per component and assemble an overset grid.
See docs/GOTCHAS.md.
"""
import numpy as np
from scipy.interpolate import CubicSpline

# Six rotations of one base face. Using rotations of a single face is what makes the
# shared patch edges coincide to machine precision instead of merely nearly.
ROT = [
    np.eye(3),
    np.diag([1.0, -1.0, -1.0]),
    np.array([[0, 0, 1.0], [0, 1, 0], [-1, 0, 0]]),
    np.array([[0, 0, -1.0], [0, 1, 0], [1, 0, 0]]),
    np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]]),
    np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]]),
]

DVS = ("L", "Sy", "Sz")


class Body:
    """Parametric closed body on a fixed cubed-sphere index grid.

    Parameters
    ----------
    xi, rz0 : arrays
        Normalized station in [0, 1] and the vertical radius at that station, in metres.
    x_nose : float
        x of the nose, in metres.
    length : float
        Baseline body length, in metres.
    ratio : float
        Baseline ry / rz of the cross-section.
    n : int
        Points per patch edge. MUST be even.
    """

    def __init__(self, xi, rz0, x_nose, length, ratio, n):
        if n % 2:
            raise ValueError("n must be EVEN, so no grid point lands on a pole; got %d" % n)
        self.x_nose = float(x_nose)
        self.L0 = float(length)
        self.ratio0 = float(ratio)
        self.n = int(n)
        self.rz0 = CubicSpline(np.asarray(xi, float), np.asarray(rz0, float))

        a = np.linspace(-np.pi / 4, np.pi / 4, self.n)
        U, V = np.meshgrid(a, a, indexing="ij")
        base = np.stack([np.tan(U), np.tan(V), np.ones_like(U)], axis=-1)
        base /= np.linalg.norm(base, axis=-1, keepdims=True)
        self.sph = [base @ R.T for R in ROT]

        # The outward-normal flip is decided ONCE at the baseline and then frozen.
        # Recomputing it per perturbation can invert one patch at p+h but not at p-h,
        # which silently scrambles node correspondence and poisons dX_surf/dp.
        self.flips = None
        self.flips = self._decide_flips()

    @classmethod
    def from_npz(cls, path, n=None):
        """Build from the table that profile.save_table writes."""
        d = np.load(path)
        return cls(d["xi"], d["rz0"], float(d["x_nose"]), float(d["length"]),
                   float(d["ratio"]), int(n if n is not None else d["n"]))

    # ---------------------------------------------------------------- geometry
    def patches(self, L=None, Sy=1.0, Sz=1.0):
        """Six (n, n, 3) patches for the given design variables."""
        L = self.L0 if L is None else L
        out = []
        for s in self.sph:
            dx, dy, dz = s[..., 0], s[..., 1], s[..., 2]
            xi = (1.0 - dx) / 2.0
            r0 = self.rz0(xi)
            th = np.arctan2(dz, dy)
            out.append(np.stack([self.x_nose + L * xi,
                                 Sy * self.ratio0 * r0 * np.cos(th),
                                 Sz * r0 * np.sin(th)], axis=-1))
        if self.flips is not None:
            out = [P[:, ::-1, :] if f else P for P, f in zip(out, self.flips)]
        return out

    def _decide_flips(self):
        n, cx = self.n, self.x_nose + 0.5 * self.L0
        i = j = n // 2
        flips = []
        for P in self.patches(self.L0, 1.0, 1.0):
            nrm = np.cross(P[i + 1, j] - P[i - 1, j], P[i, j + 1] - P[i, j - 1])
            flips.append(bool(np.dot(nrm, P[i, j] - np.array([cx, 0.0, 0.0])) < 0))
        return flips

    def nodes(self, L=None, Sy=1.0, Sz=1.0):
        """Flat (6*n*n, 3) node array. Row order is fixed for a given n."""
        return np.concatenate([P.reshape(-1, 3) for P in self.patches(L, Sy, Sz)])

    def baseline(self):
        return self.nodes(self.L0, 1.0, 1.0)

    def dv0(self):
        return {"L": self.L0, "Sy": 1.0, "Sz": 1.0}

    # --------------------------------------------------- geometric sensitivity
    def dnodes(self, dv, step=1e-3):
        """d(nodes)/d(dv) by CENTRAL difference on the generator.

        Pure geometry: two generator calls, milliseconds, and ZERO CFD solves. That
        asymmetry against a full finite difference is the point of the adjoint chain.
        The node positions are LINEAR in each of L, Sy and Sz, so this difference is
        exact rather than merely second order. scripts/check_surface.py asserts it.
        """
        if dv not in DVS:
            raise KeyError("unknown design variable %r; expected one of %s" % (dv, DVS))
        lo, hi = self.dv0(), self.dv0()
        lo[dv] -= step
        hi[dv] += step
        return (self.nodes(**hi) - self.nodes(**lo)) / (2.0 * step)

    # ------------------------------------------------------------------ checks
    def normals(self):
        """Outward unit normals at every node, from the in-surface tangents."""
        gen = self.baseline()
        cx = 0.5 * (gen[:, 0].min() + gen[:, 0].max())
        out = []
        for P in self.patches():
            nv = np.cross(np.gradient(P, axis=0), np.gradient(P, axis=1))
            ln = np.linalg.norm(nv, axis=-1, keepdims=True)
            nv = np.divide(nv, ln, out=np.zeros_like(nv), where=ln > 1e-30)
            s = np.sign(np.einsum("ijk,ijk->ij", nv, P - np.array([cx, 0.0, 0.0])))[..., None]
            s[s == 0] = 1.0
            out.append((nv * s).reshape(-1, 3))
        return np.concatenate(out)

    def health(self):
        """Numbers that must all pass before the surface is extruded."""
        from scipy.spatial import cKDTree
        pat = self.patches()
        gen = np.concatenate([P.reshape(-1, 3) for P in pat])
        d, _ = cKDTree(gen).query(gen, k=2)
        min_edge = min(min(np.linalg.norm(np.diff(P, axis=0), axis=-1).min(),
                           np.linalg.norm(np.diff(P, axis=1), axis=-1).min()) for P in pat)
        return {
            "nodes": int(len(gen)),
            "patches": len(pat),
            "shared_edge_gap": float(d[:, 1].min()),   # must be ~0: edges must coincide
            "min_edge_length": float(min_edge),        # must be > 0: no collapsed cell
            "flips": list(self.flips),
            "bounds": [float(v) for v in (gen[:, 0].min(), gen[:, 0].max(),
                                          gen[:, 1].min(), gen[:, 1].max(),
                                          gen[:, 2].min(), gen[:, 2].max())],
        }


def from_profile_npz(path, n=None):
    """Build a Body from the table that profile.py writes."""
    d = np.load(path)
    return Body(d["xi"], d["rz0"], float(d["x_nose"]), float(d["length"]),
                float(d["ratio"]), int(n if n is not None else d["n"]))
