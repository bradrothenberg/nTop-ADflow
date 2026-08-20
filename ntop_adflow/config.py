"""Case configuration: load a YAML case, validate it, emit a JSON snapshot.

Why two formats. YAML is for people to author. The solver host only has numpy and scipy in
the MACH-Aero venv, so the remote scripts read a JSON snapshot that the local build step
writes next to the surface mesh. Nothing on the remote side needs pyyaml, and the snapshot
records exactly what was used for a run.
"""
import json
import os

REQUIRED = {
    "name": str,
    "geometry": dict,
    "surface": dict,
    "flow": dict,
    "reference": dict,
}
FLOW_REQUIRED = ("mach", "alpha", "altitude")
REF_REQUIRED = ("areaRef", "chordRef", "xRef")

# The pyHyp options this pipeline sets from a case. pyHyp rejects anything it does not
# know, so validating here turns a post-push failure into an immediate one.
PYHYP_KEYS = {
    "N", "s0", "marchDist", "ps0", "pGridRatio", "cMax",
    "epsE", "epsI", "theta", "volCoef", "volBlend", "volSmoothIter",
    "splay", "splayEdgeOrthogonality", "splayCornerOrthogonality",
    "nodeTol", "symTol", "kspRelTol", "kspMaxIts", "kspSubspaceSize",
}


def load(path):
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    validate(cfg)
    cfg["_source"] = os.path.abspath(path)
    return cfg


def validate(cfg):
    for k, t in REQUIRED.items():
        if k not in cfg:
            raise KeyError("case is missing the %r section" % k)
        if not isinstance(cfg[k], t):
            raise TypeError("case section %r must be %s" % (k, t.__name__))
    for k in FLOW_REQUIRED:
        if k not in cfg["flow"]:
            raise KeyError("flow section is missing %r" % k)
    for k in REF_REQUIRED:
        if k not in cfg["reference"]:
            raise KeyError("reference section is missing %r" % k)
    n = cfg["surface"].get("n")
    if not isinstance(n, int) or n % 2:
        raise ValueError("surface.n must be an EVEN integer; got %r" % (n,))
    # liftindex is the single most common silent error in this pipeline, so it is
    # required rather than defaulted. See docs/GOTCHAS.md.
    li = cfg["flow"].get("liftindex")
    if li not in (2, 3):
        raise ValueError(
            "flow.liftindex must be set explicitly to 2 or 3. ADflow defaults to 2 "
            "(lift along y). z-up geometry needs 3, otherwise the reported CL is "
            "actually side force.")
    # Catch a mistyped pyHyp option here, locally, instead of after a push and a
    # 90 second extrusion. pyHyp rejects unknown options outright.
    bad = sorted(set(cfg.get("extrude", {})) - PYHYP_KEYS)
    if bad:
        raise ValueError("unknown extrude option(s) %s; pyHyp accepts %s"
                         % (bad, sorted(PYHYP_KEYS)))
    return cfg


def snapshot(cfg, path):
    """Write the JSON the remote scripts read."""
    out = {k: v for k, v in cfg.items() if not k.startswith("_")}
    out["_source"] = cfg.get("_source", "")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    return path


def read_snapshot(path):
    with open(path) as f:
        return json.load(f)


def adflow_options(cfg, grid, outdir):
    """Solver options built from the case. Kept in one place so the solve and the
    gradient run cannot drift apart."""
    s = cfg.get("solver", {})
    return {
        "gridFile": grid,
        "outputDirectory": outdir,
        "monitorvariables": ["resrho", "cl", "cd"],
        "equationType": cfg["flow"].get("equationType", "Euler"),
        "MGCycle": s.get("MGCycle", "sg"),
        "useANKSolver": True,
        "useNKSolver": True,
        "NKSwitchTol": s.get("NKSwitchTol", 1e-6),
        "L2Convergence": s.get("L2Convergence", 1e-12),
        "nCycles": s.get("nCycles", 4000),
        "liftindex": cfg["flow"]["liftindex"],
        "adjointL2Convergence": s.get("adjointL2Convergence", 1e-12),
        "adjointSolver": "GMRES",
        "adjointMaxIter": s.get("adjointMaxIter", 1000),
        "ADPC": True,
        "outerPreconIts": 3,
        "applyAdjointPCSubspaceSize": 400,
        "adjointSubspaceSize": 200,
        "writeTecplotSurfaceSolution": s.get("writeTecplot", False),
        "writeVolumeSolution": s.get("writeVolume", False),
        "writeSurfaceSolution": s.get("writeSurface", False),
        "printIterations": s.get("printIterations", False),
    }
