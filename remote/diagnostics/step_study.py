"""Is the finite difference converged, or is it the noisy side?

When the adjoint and the finite difference disagree, this separates the two possible
causes. A central difference carries two errors that move in OPPOSITE directions with the
step size:

    round-off   falls as 1/h    -> a LARGER step agrees better
    truncation  grows as h^2    -> a SMALLER step agrees better

So plot |FD(h) - adjoint| against h:

    slope about -1   the finite difference is round-off limited; use a larger step
    slope about +2   the finite difference is truncation limited; use a smaller step
    FLAT             neither. The finite difference is converged and the disagreement
                     belongs to the adjoint side. Go look at K.

On the reference case every line came out flat over a factor of 30 in h, which is what
ruled out a finite-difference noise floor and pointed at cancellation instead.

Needs gradients.json from a completed gradients.py run, for the adjoint values.

Run from the case directory:
    RB_DV=L RB_STEPS=1e-2,3e-2 mpirun -np 8 --mca pml ob1 --mca btl self,vader,tcp \
        python diagnostics/step_study.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from config import adflow_options            # noqa: E402
from surface import Body                     # noqa: E402


def main(work="."):
    from adflow import ADFLOW
    from idwarp import USMesh
    from baseclasses import AeroProblem
    from mpi4py import MPI
    from scipy.spatial import cKDTree

    comm = MPI.COMM_WORLD
    root = comm.rank == 0

    def log(*a):
        if root:
            print(*a, flush=True)

    with open(os.path.join(work, "case.json")) as f:
        cfg = json.load(f)
    gpath = os.path.join(work, "gradients.json")
    if not os.path.exists(gpath):
        raise SystemExit("run gradients.py first: this study compares against its adjoint "
                         "values")
    with open(gpath) as f:
        ref = json.load(f)["chained"]

    FUNCS = cfg["gradients"]["functions"]
    DV = os.environ.get("RB_DV", cfg["gradients"]["design_vars"][0])
    STEPS = [float(x) for x in os.environ.get("RB_STEPS", "1e-2,3e-2").split(",")]

    grid = os.path.join(work, "volume.cgns")
    outdir = os.path.join(work, "out_step")
    if root and not os.path.exists(outdir):
        os.makedirs(outdir)
    comm.barrier()

    CFDSolver = ADFLOW(options=adflow_options(cfg, grid, outdir))
    CFDSolver.setMesh(USMesh(options={"gridFile": grid}))
    f_, r_ = cfg["flow"], cfg["reference"]
    ap = AeroProblem(name=cfg["name"], mach=f_["mach"], alpha=f_["alpha"],
                     altitude=f_["altitude"], areaRef=r_["areaRef"],
                     chordRef=r_["chordRef"], xRef=r_["xRef"], evalFuncs=FUNCS)

    body = Body.from_npz(os.path.join(work, "profile.npz"))
    gen0 = body.baseline()
    xs_local = CFDSolver.getSurfaceCoordinates()
    dist, idx = cKDTree(gen0).query(xs_local, k=1)
    assert comm.allreduce(dist.max() if len(dist) else 0.0, op=MPI.MAX) < 1e-8

    def solve_at(**dv):
        CFDSolver.setSurfaceCoordinates(body.nodes(**dv)[idx])
        CFDSolver.updateGeometryInfo()
        CFDSolver.resetFlow(ap)
        t0 = time.time()
        CFDSolver(ap)
        d = {}
        CFDSolver.evalFunctions(ap, d)
        out = {k: d["%s_%s" % (cfg["name"], k)] for k in FUNCS}
        log("      %s  (%.0f s)"
            % ("  ".join("%s=% .10e" % (k, out[k]) for k in FUNCS), time.time() - t0))
        return out

    log("step study on %s. adjoint: %s"
        % (DV, "  ".join("d%s=% .8e" % (f, ref[f][DV]) for f in FUNCS)))
    rows = []
    for h in STEPS:
        log("   h=%.1e" % h)
        p = body.dv0()
        hi = dict(p); hi[DV] += h
        fp = solve_at(**hi)
        lo = dict(p); lo[DV] -= h
        fm = solve_at(**lo)
        row = {"h": h}
        for fn in FUNCS:
            row[fn] = (fp[fn] - fm[fn]) / (2.0 * h)
            row["rel_" + fn] = abs(ref[fn][DV] - row[fn]) / max(abs(row[fn]), 1e-30)
        rows.append(row)

    log("")
    log("=== summary, dv=%s ===" % DV)
    hdr = "   %-10s" % "h"
    for fn in FUNCS:
        hdr += " %16s %10s" % ("d" + fn, "rel")
    log(hdr)
    for r in rows:
        line = "   %-10.0e" % r["h"]
        for fn in FUNCS:
            line += " % 16.8e %10.2e" % (r[fn], r["rel_" + fn])
        log(line)
    log("")
    log("Read the rel columns DOWN each function. Falling with larger h means round-off. "
        "Rising means truncation. Flat means the FD is converged and the gap is the "
        "adjoint's; check K in gradients.json.")

    if root:
        with open(os.path.join(work, "step_study_%s.json" % DV), "w") as fh:
            json.dump(dict(dv=DV, adjoint={f: ref[f][DV] for f in FUNCS}, rows=rows),
                      fh, indent=1)
        print("wrote step_study_%s.json" % DV, flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
