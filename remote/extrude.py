"""Extrude the structured surface to a volume mesh with pyHyp. Runs on the solver host.

Reads case.json and surface.fmt from the working directory and writes volume.cgns.

CHECK THE LOG. The bad-cell column must be 0 and the minimum quality about 0.9. Two
failures mean the SURFACE is wrong, not the solver:

    a segfault, or metrics printed as 0.180+309   -> zero-area cells, the topology has a
                                                     pole on a grid point
    "free corner or other topology" error         -> collapsed patch edges

`fileType: PLOT3D` is REQUIRED for .fmt and .xyz input. Without it pyHyp tries to read the
file as CGNS and stops with "Bad integer for item 1 in list input".

Run: python extrude.py
"""
import json
import os
import sys

DEFAULTS = {
    "N": 65, "s0": 1e-5, "marchDist": 50.0,
    "ps0": -1.0, "pGridRatio": -1.0, "cMax": 3.0,
    "epsE": 1.0, "epsI": 2.0, "theta": 3.0,
    "volCoef": 0.25, "volBlend": 1e-4, "volSmoothIter": 100,
}


def main(work="."):
    with open(os.path.join(work, "case.json")) as f:
        cfg = json.load(f)
    surf = os.path.join(work, "surface.fmt")
    out = os.path.join(work, "volume.cgns")
    if not os.path.exists(surf):
        raise SystemExit("no surface.fmt in %s; run scripts/build_case.py and push first"
                         % os.path.abspath(work))

    opts = dict(DEFAULTS)
    opts.update(cfg.get("extrude", {}))
    opts.update({
        "inputFile": surf,
        "fileType": "PLOT3D",          # required for .fmt input
        "unattachedEdgesAreSymmetry": False,
        "outerFaceBC": "farfield",
        "autoConnect": True,
        "BC": {},
        "families": "wall",
    })
    print("pyHyp options:")
    for k in sorted(opts):
        print("   %-26s %s" % (k, opts[k]))

    from pyhyp import pyHyp
    h = pyHyp(options=opts)
    h.run()
    h.writeCGNS(out)
    print("wrote %s" % out)
    print("Now check above: bad cells must be 0 and minimum quality about 0.9.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
