"""Solve the case with ADflow. Runs on the solver host under mpirun.

Reads case.json and volume.cgns, writes results.json plus the surface and volume solution
if the case asks for them.

Run: mpirun -np 8 --mca pml ob1 --mca btl self,vader,tcp python solve.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# config.py, surface.py and plot3d.py are pushed alongside these scripts, so the solver
# host imports the SAME code that ran locally. No second copy to drift out of sync.
from config import adflow_options            # noqa: E402


def main(work="."):
    from adflow import ADFLOW
    from baseclasses import AeroProblem
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    root = comm.rank == 0

    with open(os.path.join(work, "case.json")) as f:
        cfg = json.load(f)
    grid = os.path.join(work, "volume.cgns")
    outdir = os.path.join(work, "out")
    if root and not os.path.exists(outdir):
        os.makedirs(outdir)
    comm.barrier()

    funcs_wanted = cfg.get("gradients", {}).get("functions", ["cl", "cd"])
    opts = adflow_options(cfg, grid, outdir)
    if root:
        print("liftindex = %d  (2 is lift along y, 3 is lift along z)" % opts["liftindex"],
              flush=True)

    CFDSolver = ADFLOW(options=opts)
    f = cfg["flow"]
    r = cfg["reference"]
    ap = AeroProblem(name=cfg["name"], mach=f["mach"], alpha=f["alpha"],
                     altitude=f["altitude"], areaRef=r["areaRef"],
                     chordRef=r["chordRef"], xRef=r["xRef"], evalFuncs=funcs_wanted)

    t0 = time.time()
    CFDSolver(ap)
    dt = time.time() - t0
    out = {}
    CFDSolver.evalFunctions(ap, out)
    res = {k: out["%s_%s" % (cfg["name"], k)] for k in funcs_wanted}

    if root:
        print("solve time %.1f s on %d ranks" % (dt, comm.size), flush=True)
        for k in funcs_wanted:
            print("   %-4s = % .10e" % (k, res[k]), flush=True)
        if "cd" in res and cfg["flow"].get("equationType", "Euler") == "Euler":
            print("   NOTE: Euler on a closed body must give zero drag. The CD above is "
                  "discretization error, not physics.", flush=True)
        with open(os.path.join(work, "results.json"), "w") as fh:
            json.dump({"functions": res, "solve_seconds": dt, "ranks": comm.size,
                       "options": {k: v for k, v in opts.items()
                                   if isinstance(v, (int, float, str, bool))}},
                      fh, indent=1)
        print("wrote results.json", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
