"""Local step: geometry -> measured profile -> structured surface -> upload bundle.

Everything this writes goes into work/<case>/ :

    profile.npz     normalized radius table
    surface.fmt     PLOT3D cubed-sphere surface for pyHyp
    case.json       the case snapshot the remote scripts read
    health.json     the surface checks, all of which must pass

Run:
    uv run --with pyyaml --with numpy --with scipy --with pyvista \
        python scripts/build_case.py cases/agardc_body.yaml
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from ntop_adflow import config, plot3d, profile, surface


def main(case_path, export=False):
    cfg = config.load(case_path)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    work = os.path.join(root, "work", cfg["name"])
    os.makedirs(work, exist_ok=True)
    g = cfg["geometry"]

    # ---------------------------------------------------------- geometry source
    surf_file = g.get("surface_file")
    if export or not surf_file:
        from ntop_adflow import ntop
        nb = g["notebook"]
        print("running ntopcl on %s" % nb)
        hits = ntop.run(nb, g.get("inputs", {}), os.path.join(work, "ntop"),
                        expect=g.get("expect", "*.obj"))
        surf_file = hits[0]
        print("  exported %s" % surf_file)
    if not os.path.exists(surf_file):
        raise SystemExit("geometry surface not found: %s" % surf_file)

    # ------------------------------------------------------------ measure profile
    import pyvista as pv
    print("reading %s" % surf_file)
    mesh = pv.read(surf_file).extract_surface(algorithm="dataset_surface")
    xs, rs, spline = profile.measure(
        mesh, g["measure_from"], g["measure_to"], g["x_nose"], g["x_tail"],
        band=g.get("band", 0.05), side=g.get("side", "bottom"),
        smooth=g.get("smooth", 2e-6))
    print("  %d usable stations, max measured radius %.6f m" % (len(xs), rs.max()))

    # The ratio cross-check needs a station the caller picked, clear of the wing. There is
    # no reliable way to find one automatically, so it is skipped unless ratio_at_x is set.
    ratio = g.get("ratio")
    x_chk = g.get("ratio_at_x")
    if x_chk is not None:
        meas_ratio, ry, rz = profile.width_ratio(mesh, x_chk, band=g.get("band", 0.05))
        print("  measured ry/rz at x=%.3f: %.4f  (ry=%.4f rz=%.4f)"
              % (x_chk, meas_ratio, ry, rz))
        if ratio is None:
            ratio = meas_ratio
        elif abs(meas_ratio - ratio) / ratio > 0.05:
            print("  WARNING: configured ratio %.4f differs from measured %.4f by more "
                  "than 5 percent. Either ratio_at_x cuts through a wing or fin, or the "
                  "configured value is wrong. See docs/GOTCHAS.md." % (ratio, meas_ratio))
    elif ratio is None:
        raise SystemExit("set geometry.ratio, or geometry.ratio_at_x at a station that is "
                         "clear of the wing and the fin")
    else:
        print("  ratio %.4f taken from the case (no ratio_at_x given, so no cross-check)"
              % ratio)

    npz = os.path.join(work, "profile.npz")
    info = profile.save_table(npz, spline, g["x_nose"], g["x_tail"], ratio,
                              cfg["surface"]["n"], raw=(xs, rs))
    print("  profile: length %.6f m, max rz %.6f, max ry %.6f, ends %.2e / %.2e"
          % (info["length"], info["max_rz"], info["max_ry"],
             info["end_radii"][0], info["end_radii"][1]))

    # ------------------------------------------------------------- build surface
    body = surface.from_profile_npz(npz)
    h = body.health()
    print("surface: %d nodes in %d patches" % (h["nodes"], h["patches"]))
    print("  shared edge gap  %.2e   (must be ~0)" % h["shared_edge_gap"])
    print("  min edge length  %.3e   (must be > 0)" % h["min_edge_length"])
    print("  flips %s" % h["flips"])
    if h["shared_edge_gap"] > 1e-12:
        raise SystemExit("shared patch edges do not coincide; pyHyp will report a free "
                         "corner or produce bad cells")
    if h["min_edge_length"] <= 0.0:
        raise SystemExit("a collapsed cell exists; check that surface.n is EVEN")

    fmt = os.path.join(work, "surface.fmt")
    plot3d.write(fmt, body.patches())
    back = plot3d.read_nodes(fmt)
    err = float(np.abs(back - body.baseline()).max())
    print("  wrote %s, round-trip max error %.2e" % (fmt, err))

    with open(os.path.join(work, "health.json"), "w") as f:
        json.dump({"surface": h, "profile": info, "roundtrip": err}, f, indent=1)
    cfg = dict(cfg)
    cfg["geometry"] = dict(g, surface_file=surf_file, ratio=ratio)
    config.snapshot(cfg, os.path.join(work, "case.json"))

    r = cfg["remote"]
    print()
    print("Next, push and extrude on the solver host:")
    print("  scripts/push.sh %s" % cfg["name"])
    print("  ssh %s 'cd %s && ./extrude.sh'" % (r["host"], r["dir"]))
    return work


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1], export="--export" in sys.argv)
