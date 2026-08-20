"""Offline checks on the surface generator. No solver, no host, seconds to run.

Run these after ANY change to surface.py or profile.py, before pushing anything:

    uv run --with numpy --with scipy --with pytest python -m pytest tests -q

The linearity test is the one that matters most for the adjoint: the geometry term is a
central difference on the generator, and it is only exact because the node positions are
linear in every design variable. If that stops being true, dX_surf/dp silently picks up a
truncation error and every gradient inherits it.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ntop_adflow import plot3d, surface       # noqa: E402


def make_body(n=16):
    """A smooth analytic test body. No file or geometry export needed."""
    xi = np.linspace(0.0, 1.0, 2001)
    rz0 = 0.15 * np.sin(np.pi * xi) ** 0.8          # closes at both ends
    return surface.Body(xi, rz0, x_nose=0.01, length=3.0, ratio=1.25, n=n)


def test_odd_n_rejected():
    xi = np.linspace(0, 1, 101)
    with pytest.raises(ValueError, match="EVEN"):
        surface.Body(xi, 0.1 * np.sin(np.pi * xi), 0.0, 1.0, 1.0, n=15)


def test_node_count_and_bounds():
    b = make_body(16)
    nodes = b.baseline()
    assert nodes.shape == (6 * 16 * 16, 3)
    assert nodes[:, 0].min() > 0.0
    assert nodes[:, 0].max() < 3.02


def test_shared_edges_coincide_exactly():
    """Patches are rotations of one base face, so shared edges must be identical."""
    h = make_body(16).health()
    assert h["shared_edge_gap"] == 0.0
    assert h["min_edge_length"] > 0.0


def test_normals_point_outward():
    b = make_body(16)
    nodes, nrm = b.baseline(), b.normals()
    cx = 0.5 * (nodes[:, 0].min() + nodes[:, 0].max())
    radial = nodes - np.array([cx, 0.0, 0.0])
    # Skip nodes near the tips, where the radial direction is nearly axial and the
    # sign test is ill-conditioned rather than wrong.
    keep = np.abs(radial[:, 1]) + np.abs(radial[:, 2]) > 1e-3
    assert np.all(np.einsum("ij,ij->i", nrm[keep], radial[keep]) > 0)


@pytest.mark.parametrize("dv", surface.DVS)
def test_nodes_are_linear_in_each_dv(dv):
    """The generator must be exactly linear in L, Sy and Sz.

    That is what makes the central difference in dnodes() exact instead of second order,
    and it is asserted here rather than assumed.
    """
    b = make_body(16)
    d_big = b.dnodes(dv, 1e-1)
    d_small = b.dnodes(dv, 1e-4)
    scale = np.linalg.norm(d_small)
    assert np.linalg.norm(d_big - d_small) / scale < 1e-9


def test_dv_names_validated():
    with pytest.raises(KeyError):
        make_body(16).dnodes("nonsense")


def test_flips_are_frozen_not_recomputed():
    """A perturbed shape must reuse the baseline flip decision.

    If the flip were recomputed per perturbation it could invert at +h but not at -h,
    scrambling node correspondence with no error. Detect it by checking that a large
    perturbation keeps the node ordering continuous.
    """
    b = make_body(16)
    a = b.nodes(Sz=1.0)
    c = b.nodes(Sz=1.4)
    # Same row must stay the same point on the body: x is untouched by Sz.
    assert np.allclose(a[:, 0], c[:, 0])
    # and z must scale monotonically, row by row, with no reordering
    nz = np.abs(a[:, 2]) > 1e-6
    assert np.allclose(c[nz, 2] / a[nz, 2], 1.4, atol=1e-9)


def test_plot3d_roundtrip(tmp_path):
    b = make_body(16)
    p = str(tmp_path / "s.fmt")
    plot3d.write(p, b.patches())
    back = plot3d.read_nodes(p)
    # 10 significant digits in the file, so expect file rounding and nothing more
    assert np.abs(back - b.baseline()).max() < 1e-9


def test_agardc_regression_if_available():
    """If the AGARD-C case has been built, its surface must not have moved.

    This is the end-to-end guard on the generator: the reference surface was extruded and
    solved, and its numbers are in docs/VALIDATION.md.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    npz = os.path.join(root, "work", "agardc_body", "profile.npz")
    fmt = os.path.join(root, "work", "agardc_body", "surface.fmt")
    if not (os.path.exists(npz) and os.path.exists(fmt)):
        pytest.skip("agardc_body not built; run scripts/build_case.py first")
    body = surface.Body.from_npz(npz)
    ref = plot3d.read_nodes(fmt)
    assert np.abs(body.baseline() - ref).max() < 1e-9
    h = body.health()
    assert h["nodes"] == 6144
    assert h["shared_edge_gap"] == 0.0
    assert abs(h["min_edge_length"] - 1.524e-03) < 1e-5
