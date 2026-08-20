# nTop to ADflow

Run ADflow (MDO Lab) on nTop-generated geometry, and get adjoint gradients of the forces
with respect to nTop geometry parameters at a cost that does not grow with the number of
design variables.

```
nTop notebook --ntopcl--> triangulated surface --measure--> radius profile
    --cubed sphere--> structured surface --pyHyp--> volume mesh --ADflow--> forces
                                                            \--adjoint--> dJ/dX_surf
                                                                            |
                             dJ/dp = (dJ/dX_surf).(dX_surf/dp) <------------/
```

## Scope

Works today for a **closed slender body**: a fuselage, a store, a nacelle centrebody.
Validated end to end against a reference case to 2e-9 on the forces.

Not implemented: wings, fins, tails, full aircraft. Those need an overset mesh, because
ADflow reads structured or overset CGNS only and there is no automatic structured mesher
for a general part. Optimization is blocked until pyOptSparse is installed.

## Getting started

Read `CLAUDE.md`. It is the working instruction set, and several failure modes here return
a plausible wrong number instead of an error.

Reproduce the validated case first:

```bash
uv run --with pyyaml --with numpy --with scipy --with pyvista python scripts/build_case.py cases/agardc_body.yaml
bash scripts/push.sh agardc_body
ssh 192.168.20.10 "cd ~/ntop_adflow_cases/agardc_body && ./extrude.sh && ./solve.sh"
```

Offline tests, no solver needed:

```bash
uv run --with numpy --with scipy --with pytest python -m pytest tests -q
```

## Documentation

| File | Contents |
|---|---|
| `CLAUDE.md` | the working instruction set: workflow, gradients, traps |
| `docs/GOTCHAS.md` | every failure mode found so far, worst first |
| `docs/VALIDATION.md` | reference numbers to check against, and what has been ruled out |
| `docs/BUILD_RHEL.md` | building ADflow and MACH-Aero without sudo |

## A number worth knowing before you use a gradient

Every gradient is reported with a cancellation factor `K = sum|c_i| / |sum c_i|`. On the
validated case, every gradient with K below 63 agreed with a finite difference to better
than 2 percent, and every one with K above 376 was off by 5 to 32 percent, with nothing in
between. K costs nothing to compute. Read it.
