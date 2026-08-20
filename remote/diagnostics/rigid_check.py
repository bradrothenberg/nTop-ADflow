"""Validate the gradient chain against a design variable whose answer is KNOWN.

Adjoint against finite difference compares two measurements. When they disagree, neither
one tells you which is wrong. This test removes that ambiguity.

Translate the whole body rigidly. A closed body moving through a uniform stream cannot
change its force coefficients, so the true derivative of CL and CD is ZERO. The moment
does change, because the moment reference stays fixed in space while the body moves, and
that non-zero value is a second, non-trivial check.

Run this first whenever the chain is modified. It exercises the node mapping, the assembly
of partial contributions across MPI ranks, and the resBar seeding, all at once. On the
reference case it matched the finite difference to 3e-7 on CMY and returned ~1e-15 for CL
and CD.

Caveat worth knowing: translation invariance is an identity the discrete adjoint satisfies
by construction, so passing this test proves the plumbing is right but does NOT prove the
surface derivative is locally accurate. Use zone_check.py for that.

Run from the case directory:
    mpirun -np 8 --mca pml ob1 --mca btl self,vader,tcp python diagnostics/rigid_check.py
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
    H = float(os.environ.get("RB_H", "1e-3"))

    def log(*a):
        if root:
            print(*a, flush=True)

    with open(os.path.join(work, "case.json")) as f:
        cfg = json.load(f)
    FUNCS = cfg.get("gradients", {}).get("functions", ["cl", "cd", "cmy"])
    grid = os.path.join(work, "volume.cgns")
    outdir = os.path.join(work, "out_rigid")
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
    NG = len(gen0)
    xs_local = CFDSolver.getSurfaceCoordinates()
    dist, idx = cKDTree(gen0).query(xs_local, k=1)
    assert comm.allreduce(dist.max() if len(dist) else 0.0, op=MPI.MAX) < 1e-8

    def assemble(local):
        g = np.zeros((NG, 3))
        if local is not None and len(local):
            np.add.at(g, idx, local)
        return comm.allreduce(g, op=MPI.SUM)

    def solve_at(nodes, tag):
        CFDSolver.setSurfaceCoordinates(nodes[idx])
        CFDSolver.updateGeometryInfo()
        CFDSolver.resetFlow(ap)
        t0 = time.time()
        CFDSolver(ap)
        d = {}
        CFDSolver.evalFunctions(ap, d)
        out = {k: d["%s_%s" % (cfg["name"], k)] for k in FUNCS}
        log("   %-12s %s  (%.0f s)"
            % (tag, "  ".join("%s=% .10e" % (k, out[k]) for k in FUNCS), time.time() - t0))
        return out

    log("")
    log("=== baseline and adjoint ===")
    base = solve_at(gen0, "baseline")
    gx = {}
    for fn in FUNCS:
        CFDSolver.solveAdjoint(ap, fn)
        psi = -CFDSolver.getAdjoint(fn)
        gx[fn] = assemble(CFDSolver.computeJacobianVectorProductBwd(
            resBar=psi, funcsBar=CFDSolver._getFuncsBar(fn), xSDeriv=True))

    fields = {
        # rigid x is almost purely TANGENTIAL over a long mid-body
        "rigid_x": np.tile([1.0, 0.0, 0.0], (NG, 1)),
        # rigid z is normal to the surface on the top and the bottom
        "rigid_z": np.tile([0.0, 0.0, 1.0], (NG, 1)),
    }
    out = {}
    for name, D in fields.items():
        log("")
        log("=== %s, h=%.1e (true dCL and dCD are ZERO) ===" % (name, H))
        fp = solve_at(gen0 + H * D, name + " +h")
        fm = solve_at(gen0 - H * D, name + " -h")
        out[name] = {}
        for fn in FUNCS:
            a = float((gx[fn] * D).sum())
            b = (fp[fn] - fm[fn]) / (2.0 * H)
            rel = abs(a - b) / max(abs(b), 1e-30)
            out[name][fn] = dict(adjoint=a, fd=b, rel=rel)
            log("   d%-4s adjoint=% .8e  FD=% .8e  rel=%.2e" % (fn, a, b, rel))

    log("")
    log("=== verdict ===")
    worst_zero = max(abs(out[n][f]["adjoint"]) for n in fields for f in ("cl", "cd")
                     if f in FUNCS)
    log("   largest |adjoint| where the answer must be zero: %.2e" % worst_zero)
    if "cmy" in FUNCS:
        worst_cmy = max(out[n]["cmy"]["rel"] for n in fields)
        log("   worst relative difference on CMY: %.2e" % worst_cmy)
        log("   %s" % ("PASS: the chain assembly is correct" if worst_cmy < 1e-4
                       else "FAIL: the chain assembly is wrong, fix it before anything else"))
    if root:
        with open(os.path.join(work, "rigid_check.json"), "w") as fh:
            json.dump(dict(h=H, baseline=base, fields=out), fh, indent=1)
        print("wrote rigid_check.json", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
