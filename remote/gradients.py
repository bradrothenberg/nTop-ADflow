"""Adjoint gradients with respect to geometry parameters. Runs on the solver host.

    dJ/dp = (dJ/dX_surf) . (dX_surf/dp)

The flow term is ADflow's discrete adjoint: ONE adjoint solve per function, and the cost
does not grow with the number of design variables. The geometry term is pure geometry: two
generator calls per design variable and ZERO CFD solves. That asymmetry is the whole point.
A full finite difference needs two flow solves per variable, so the two costs cross at two
design variables.

THREE THINGS THAT WILL BITE YOU. All three were found the hard way; see docs/GOTCHAS.md.

1. The flow term needs BOTH seeds. computeJacobianVectorProductBwd with funcsBar alone
   returns the FROZEN-FLOW PARTIAL derivative, which ignores how the pressure field
   responds to the shape change. The total needs resBar = -getAdjoint(f) passed together
   with funcsBar, after solveAdjoint. That is what evalFunctionsSens does internally. The
   partial was wrong by up to 17 times and wrong in SIGN for 2 of 9 pairs, yet under a
   width scaling it reproduced CL and CD to nine digits, so it looked right. This script
   computes both and reports the difference, because that trap is easy to fall back into.

2. Do NOT deduplicate the surface gradient. ADflow exposes 6144 surface nodes on a 32x32
   cubed sphere while IDWarp's internal surface has 5768: the difference is patch-edge
   nodes stored once per patch. ADflow returns a PARTIAL contribution on each copy, so
   summing the copies is correct. Dividing by multiplicity made the median error worse.

3. Verify by WARPING, never by re-extruding. pyHyp rebuilds the volume mesh instead of
   warping it, so it changes the discretization as well as the shape. The finite-difference
   check here uses setSurfaceCoordinates then updateGeometryInfo, which is the chain the
   adjoint actually linearizes.

K, THE CANCELLATION FACTOR. For each gradient this script reports

    K = sum_i |c_i| / |sum_i c_i|,   c_i = (dJ/dX_i) . (dX_i/dp)

K measures how much the projection relies on cancellation, and it costs nothing: both
terms are already in hand. On the 32x32 AGARD-C body, every pair with K below 63 agreed
with a finite difference to better than 2 percent and every pair with K above 376
disagreed by 5 percent or more, with nothing in between. Rank correlation between K and
the error was 0.983 over four decades. Read K before you trust a gradient. Do not expect a
quantitative law from it: a linear K times 1 percent model over-predicts badly at large K.

Environment overrides, so the smoke test runs this exact code path:
    RB_L2      solver and adjoint tolerance (default from the case)
    RB_DVS     comma-separated subset of design variables
    RB_NOFD    set to 1 to skip the finite-difference verification

Run: mpirun -np 8 --mca pml ob1 --mca btl self,vader,tcp python gradients.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# config.py and surface.py are pushed alongside these scripts, so the solver host imports
# the SAME code that ran locally. No second copy to drift out of sync.
from config import adflow_options            # noqa: E402
from surface import Body                      # noqa: E402


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
    gcfg = cfg.get("gradients", {})
    FUNCS = gcfg.get("functions", ["cl", "cd"])
    DVS = tuple(os.environ.get("RB_DVS", ",".join(gcfg.get("design_vars",
                                                           ["L", "Sy", "Sz"]))).split(","))
    geom_step = float(gcfg.get("geom_step", 1e-3))
    fd_step = gcfg.get("fd_step", {})
    k_warn = float(gcfg.get("k_warn", 100.0))
    do_fd = os.environ.get("RB_NOFD", "0") != "1"

    if "RB_L2" in os.environ:
        cfg.setdefault("solver", {})
        cfg["solver"]["L2Convergence"] = float(os.environ["RB_L2"])
        cfg["solver"]["adjointL2Convergence"] = float(os.environ["RB_L2"])

    grid = os.path.join(work, "volume.cgns")
    outdir = os.path.join(work, "out_grad")
    if root and not os.path.exists(outdir):
        os.makedirs(outdir)
    comm.barrier()

    CFDSolver = ADFLOW(options=adflow_options(cfg, grid, outdir))
    CFDSolver.setMesh(USMesh(options={"gridFile": grid}))
    f_, r_ = cfg["flow"], cfg["reference"]
    ap = AeroProblem(name=cfg["name"], mach=f_["mach"], alpha=f_["alpha"],
                     altitude=f_["altitude"], areaRef=r_["areaRef"],
                     chordRef=r_["chordRef"], xRef=r_["xRef"], evalFuncs=FUNCS)

    # ------------------------------------------------- surface node correspondence
    body = Body.from_npz(os.path.join(work, "profile.npz"))
    gen0 = body.baseline()
    NG = len(gen0)
    xs_local = CFDSolver.getSurfaceCoordinates()
    dist, idx = cKDTree(gen0).query(xs_local, k=1)
    maxd = comm.allreduce(dist.max() if len(dist) else 0.0, op=MPI.MAX)
    log("surface nodes: generator %d, adflow %d over %d ranks"
        % (NG, comm.allreduce(len(xs_local), op=MPI.SUM), comm.size))
    log("surface match: max |x_adflow - x_generator| = %.3e" % maxd)
    if maxd > 1e-8:
        raise SystemExit("surface node match failed: the generator and the mesh disagree. "
                         "The volume mesh was probably extruded from a different surface.")

    def assemble(local):
        """Scatter local per-node values onto generator rows and sum across ranks.

        Patch-edge nodes are duplicated, and each rank holds only its own block's partial
        contribution, so a plain sum assembles the true nodal value. See note 2 above.
        """
        g = np.zeros((NG, 3))
        if local is not None and len(local):
            np.add.at(g, idx, local)
        return comm.allreduce(g, op=MPI.SUM)

    def solve(tag):
        CFDSolver.resetFlow(ap)
        t0 = time.time()
        CFDSolver(ap)
        d = {}
        CFDSolver.evalFunctions(ap, d)
        out = {k: d["%s_%s" % (cfg["name"], k)] for k in FUNCS}
        log("   %-16s %s  (%.0f s)"
            % (tag, "  ".join("%s=% .10e" % (k, out[k]) for k in FUNCS), time.time() - t0))
        return out

    def set_surface(nodes):
        CFDSolver.setSurfaceCoordinates(nodes[idx])
        CFDSolver.updateGeometryInfo()

    log("")
    log("=== 1. baseline ===")
    base = solve("baseline")

    log("")
    log("=== 2. adjoint: dJ/dX_surf, total and frozen-flow ===")
    gx, gx_part = {}, {}
    t0 = time.time()
    for fn in FUNCS:
        gx_part[fn] = assemble(CFDSolver.computeJacobianVectorProductBwd(
            funcsBar={fn: 1.0}, xSDeriv=True))
        CFDSolver.solveAdjoint(ap, fn)
        psi = -CFDSolver.getAdjoint(fn)
        gx[fn] = assemble(CFDSolver.computeJacobianVectorProductBwd(
            resBar=psi, funcsBar=CFDSolver._getFuncsBar(fn), xSDeriv=True))
        log("   d%-4s/dX  total |g|=%.6e   frozen-flow |g|=%.6e"
            % (fn, np.linalg.norm(gx[fn]), np.linalg.norm(gx_part[fn])))
    log("   %d adjoint solves in %.0f s" % (len(FUNCS), time.time() - t0))
    if getattr(ap, "adjointFailed", False):
        raise SystemExit("an adjoint solve failed; the gradients are not usable")

    log("")
    log("=== 3. geometry: dX_surf/dp, no CFD ===")
    dXdp = {}
    for dv in DVS:
        cols = [body.dnodes(dv, h) for h in (geom_step * 10, geom_step, geom_step / 10)]
        spread = max(abs(np.linalg.norm(c) - np.linalg.norm(cols[1])) for c in cols)
        dXdp[dv] = cols[1]
        log("   dX/d%-3s |dX/dp|=%.8e   step spread over 2 decades %.2e (want ~0)"
            % (dv, np.linalg.norm(cols[1]), spread))

    log("")
    log("=== 4. chained gradients ===")
    chained, frozen, Kfac = {}, {}, {}
    for fn in FUNCS:
        chained[fn], frozen[fn], Kfac[fn] = {}, {}, {}
        for dv in DVS:
            c = (gx[fn] * dXdp[dv]).sum(axis=1)
            net = c.sum()
            chained[fn][dv] = float(net)
            frozen[fn][dv] = float((gx_part[fn] * dXdp[dv]).sum())
            Kfac[fn][dv] = float(np.abs(c).sum() / abs(net)) if net else float("inf")
            flag = "   <-- large K, distrust" if Kfac[fn][dv] > k_warn else ""
            log("   d%-4s/d%-3s = % .8e   K=%9.1f   (frozen-flow % .8e)%s"
                % (fn, dv, chained[fn][dv], Kfac[fn][dv], frozen[fn][dv], flag))

    fd = {fn: {} for fn in FUNCS}
    if do_fd:
        log("")
        log("=== 5. verification: central FD on the total, via IDWarp ===")
        for dv in DVS:
            h = float(fd_step.get(dv, 1e-3))
            log("   %s: h=%.1e" % (dv, h))
            p = body.dv0()
            hi = dict(p); hi[dv] += h
            set_surface(body.nodes(**hi))
            fp = solve("%s + h" % dv)
            lo = dict(p); lo[dv] -= h
            set_surface(body.nodes(**lo))
            fm = solve("%s - h" % dv)
            for fn in FUNCS:
                fd[fn][dv] = (fp[fn] - fm[fn]) / (2.0 * h)
        set_surface(gen0)

        log("")
        log("=== 6. summary ===")
        log("   %-4s %-4s %18s %18s %10s %10s" %
            ("func", "dv", "adjoint chain", "central FD", "rel.diff", "K"))
        for fn in FUNCS:
            for dv in DVS:
                a, b = chained[fn][dv], fd[fn][dv]
                rel = abs(a - b) / max(abs(b), 1e-30)
                log("   %-4s %-4s % 18.8e % 18.8e %10.2e %10.1f"
                    % (fn, dv, a, b, rel, Kfac[fn][dv]))

    if root:
        np.savez(os.path.join(work, "gradients.npz"), gen0=gen0,
                 **{"gxs_" + fn: gx[fn] for fn in FUNCS},
                 **{"gxspart_" + fn: gx_part[fn] for fn in FUNCS},
                 **{"dXdp_" + dv: dXdp[dv] for dv in DVS})
        with open(os.path.join(work, "gradients.json"), "w") as fh:
            json.dump(dict(baseline=base, chained=chained, frozen=frozen, K=Kfac, fd=fd,
                           functions=FUNCS, design_vars=list(DVS),
                           geom_step=geom_step, fd_step=fd_step,
                           n_surf_nodes=NG, surf_match=float(maxd),
                           k_warn=k_warn), fh, indent=1)
        print("wrote gradients.npz and gradients.json", flush=True)
    log("DONE")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
