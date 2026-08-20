"""Measure a body radius profile from an exported triangulated surface.

The nTop export is a triangle mesh. The structured generator needs a smooth radius
against normalized station. This module slices the mesh and fits a smoothing spline.

TWO measurement traps, both learned by getting them wrong on the AGARD-C body:

1. Do NOT measure at the equator. On a wing-body geometry the wing also sits near z = 0,
   so a band around z = 0 reads the wing SPAN (0.61 m) instead of the body radius
   (0.18 m). Measure the bottom of a narrow centreplane band instead, which is clear of
   the wing above it and of the fin.

2. Measure ry / rz separately and check it against the nTop input. On AGARD-C the
   measured ratio was 1.234 against 1.230 from the notebook's FUSE-Y/ZScale inputs. That
   agreement is what confirms the measurement is reading the body and nothing else.

Output is a normalized table, so a change in body length stretches the profile instead of
truncating it.
"""
import numpy as np


def measure(mesh, x_from, x_to, x_nose, x_tail,
            n_station=200, band=0.05, side="bottom", smooth=2e-6):
    """Radius against x, from slices of a triangulated surface.

    Parameters
    ----------
    mesh : pyvista surface
    x_from, x_to : float
        Axial range to SAMPLE. Keep it strictly inside the body ends, because a slice
        exactly at a closed tip returns too few points to read.
    x_nose, x_tail : float
        The body ends. The spline is anchored to zero radius at these, which is where the
        body actually closes. Anchoring at the sampling range instead duplicates a knot
        when the first sample sits on the range start, and UnivariateSpline then returns
        NaN with no error.
    band : float
        Half width of the centreplane band, in metres.
    side : {"bottom", "top"}
        Which side of the centreplane band to read. Use the side that is clear of the
        wing and the fin.
    smooth : float
        Smoothing factor for UnivariateSpline. Larger removes more measurement noise, and
        more real shape with it.
    """
    from scipy.interpolate import UnivariateSpline
    if not (x_nose < x_from < x_to < x_tail):
        raise ValueError("need x_nose < x_from < x_to < x_tail; got %r"
                         % ((x_nose, x_from, x_to, x_tail),))
    xs, rs = [], []
    for x in np.linspace(x_from, x_to, n_station):
        sl = mesh.slice(normal=[1, 0, 0], origin=[x, 0, 0])
        if sl.n_points < 4:
            continue
        cp = sl.points[np.abs(sl.points[:, 1]) < band]
        if len(cp) < 3:
            continue
        v = cp[:, 2].min() if side == "bottom" else cp[:, 2].max()
        if (side == "bottom" and v >= -1e-4) or (side == "top" and v <= 1e-4):
            continue
        xs.append(x)
        rs.append(abs(v))
    if len(xs) < 10:
        raise RuntimeError("only %d usable stations; check the x range, band and side"
                           % len(xs))
    kx = np.r_[x_nose, np.array(xs), x_tail]
    ky = np.r_[0.0, np.array(rs), 0.0]
    if np.any(np.diff(kx) <= 0):
        raise ValueError("spline stations are not strictly increasing; a duplicate knot "
                         "makes UnivariateSpline return NaN")
    sp = UnivariateSpline(kx, ky, s=smooth, k=3)
    if not np.all(np.isfinite(sp(np.linspace(x_nose, x_tail, 101)))):
        raise RuntimeError("profile spline is not finite; try a larger smooth value")
    return np.array(xs), np.array(rs), sp


def width_ratio(mesh, x, band=0.05):
    """Measured ry / rz at ONE station.

    Pick the station carefully. On a wing-body geometry a slice through the wing reads
    the wing, not the body: on AGARD-C a slice at mid-body returned ry/rz = 2.16 because
    the wing is there, against the true body value of 1.234. Use a station forward of the
    wing root leading edge, or aft of everything. There is no way to detect this
    automatically, so the caller must choose.
    """
    sl = mesh.slice(normal=[1, 0, 0], origin=[x, 0, 0])
    if sl.n_points < 4:
        raise RuntimeError("slice at x=%.4f has too few points" % x)
    p = sl.points
    ry = np.abs(p[np.abs(p[:, 2]) < band][:, 1]).max()
    rz = np.abs(p[np.abs(p[:, 1]) < band][:, 2]).max()
    return float(ry / rz), float(ry), float(rz)


def save_table(path, spline, x_nose, x_tail, ratio, n, n_table=8001, raw=None):
    """Write the NORMALIZED profile table that surface.py consumes.

    The table is sampled here, once, and never re-fitted downstream. That keeps the
    surface bit-identical on the workstation and on the solver host, and removes any
    dependence on the scipy version present on either.
    """
    length = x_tail - x_nose
    xi = np.linspace(0.0, 1.0, n_table)
    rz0 = np.clip(spline(x_nose + length * xi), 0.0, None)
    kw = {}
    if raw is not None:
        kw["xs_raw"], kw["rs_raw"] = raw
    np.savez(path, xi=xi, rz0=rz0, x_nose=x_nose, x_tail=x_tail,
             length=length, ratio=ratio, n=n, **kw)
    return dict(length=float(length), max_rz=float(rz0.max()),
                max_ry=float(rz0.max() * ratio),
                xi_at_max=float(xi[rz0.argmax()]),
                end_radii=[float(rz0[0]), float(rz0[-1])])
