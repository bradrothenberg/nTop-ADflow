# Validation and regression numbers

Every number here was measured on the reference host (8 MPI ranks). If a change moves any
of them, the change broke something. Check these before trusting a new case.

## Level 1: toolchain, MACH-Aero tutorial wing

Confirms ADflow itself, independently of anything in this repo.

```bash
ssh 192.168.20.10
cd ~/adflow_val
nohup ./run.sh > rb.out 2>&1 &
```

| Quantity | Expected |
|---|---|
| Mesh | `tutorial/wing/meshing/volume/wing_vol_L3.cgns`, 200,832 cells |
| CL | 0.3911 |
| CD | 0.01456 |
| Convergence | iteration 44 |
| Time | 25.7 s on 8 cores |

## Level 2: this pipeline, AGARD-C body, `cases/agardc_body.yaml`

Confirms the whole chain: profile measurement, cubed-sphere surface, pyHyp extrusion,
ADflow solve. Run `scripts/build_case.py`, `scripts/push.sh`, then `./extrude.sh` and
`./solve.sh`.

### Local build

| Check | Expected |
|---|---|
| Usable measurement stations | 196 |
| Measured ry/rz at x = 0.60 | 1.2307 (notebook FUSE-Y/ZScale gives 1.230) |
| Body length | 3.102000 m |
| Max rz | 0.143980 m |
| Max ry | 0.177672 m |
| Surface nodes | 6144 in 6 patches |
| Shared edge gap | 0.00e+00 |
| Min edge length | 1.524e-03 |
| Flips | all six True |

### Extrusion

| Check | Expected |
|---|---|
| Layers | 65 |
| Bad cells | **0** |
| Min quality | 0.9154 |
| volume.cgns size | 9,662,464 bytes |
| Volume cells | 369,024 |
| Time | about 4 s |

### Solve

| Quantity | Expected | Reproduced by this repo |
|---|---|---|
| CL | 8.9272430699e-03 | 8.9272430514e-03 (rel 2.1e-9) |
| CD | 4.2321577353e-02 | 4.2321577351e-02 (rel 5e-11) |
| CMY | 2.5715836959e-02 | 2.5715836990e-02 (rel 1.2e-9) |
| Solve time | 133 s to L2 1e-12 | 133.0 s |

**CD is not drag.** The model is inviscid and the body is closed, so the true drag is
exactly zero. That 0.0423 is discretization error. See `docs/GOTCHAS.md`.

## Level 3: gradients

`./smoke.sh` runs the same code path at loose tolerance on one design variable in about
2 minutes. Expected, at `RB_L2=1e-5`, `dv=Sz`:

| Quantity | Expected |
|---|---|
| baseline CL | 9.2952812e-03 (loose, so it differs from the converged value above) |
| total \|dCL/dX\| | 1.747007e+00 |
| frozen-flow \|dCL/dX\| | 1.658086e-01 |
| dCL/dSz | -1.2166981e-03, K = 1690.4, flagged |
| dCD/dSz | 4.6934544e-02, K = 10.6, not flagged |
| dCMY/dSz | -2.7288000e-03, K = 2091.4, flagged |
| step spread of dX/dSz | 3.87e-13 |

Note the total is about 10 times the frozen-flow term. If they are close, you have almost
certainly dropped the `resBar` seed.

### Full gradient run, converged

`./gradients.sh` takes about 21 minutes: 1 solve, 3 adjoints, 6 solves for the
finite-difference check. Reference values on the 32x32 mesh:

| Function | DV | Adjoint chain | Central FD | Rel. diff. | K |
|---|---|---|---|---|---|
| CL | L | -6.238442e-04 | -7.640936e-04 | 1.8e-01 | 9298 |
| CL | Sy | 1.987657e-02 | 2.014871e-02 | 1.4e-02 | 59 |
| CL | Sz | -8.135758e-04 | -6.508829e-04 | 2.5e-01 | 2476 |
| CD | L | -3.657211e-03 | -3.484140e-03 | 5.0e-02 | 376 |
| CD | Sy | 4.778222e-02 | 4.772839e-02 | 1.1e-03 | 10 |
| CD | Sz | 4.694052e-02 | 4.645595e-02 | 1.0e-02 | 11 |
| CMY | L | 7.791643e-04 | 1.148954e-03 | 3.2e-01 | 21286 |
| CMY | Sy | 4.842589e-02 | 4.768274e-02 | 1.6e-02 | 63 |
| CMY | Sz | -3.830515e-03 | -4.233772e-03 | 9.5e-02 | 1466 |

The pattern is the point: **every pair with K <= 63 verifies to better than 2 percent, and
every pair with K >= 376 fails.** Nothing lands in between. Rank correlation between K and
the error is 0.983. This is a property of the projection, not a bug: the adjoint surface
derivative is accurate to about 1 percent everywhere on the body, and cancellation
amplifies that.

## Level 4: the gradient chain itself, rigid-translation test

The only test here with a **known** answer, so run it first whenever the chain changes.
Takes about 15 minutes: 1 solve, 3 adjoints, 4 solves.

```bash
mpirun -np 8 --mca pml ob1 --mca btl self,vader,tcp python diagnostics/rigid_check.py
```

Measured in this repo:

| Quantity | Expected | Meaning |
|---|---|---|
| largest \|adjoint\| for CL and CD | 1.42e-14 | true answer is exactly zero |
| worst relative difference on CMY | 2.90e-07 | true answer is non-zero and non-trivial |
| verdict | PASS | node mapping, MPI assembly and resBar seeding all correct |

A closed body translating through a uniform stream cannot change its force coefficients, so
dCL and dCD must vanish. CMY does change, because the moment reference stays fixed in space
while the body moves, and that non-zero value is the real check.

Caveat: translation invariance is an identity the discrete adjoint satisfies by
construction. Passing proves the plumbing is right; it does not prove the surface
derivative is locally accurate. That is what item 3 in the CLAUDE.md missing-work list
would settle.

## What was ruled out, and how

If you see the same 5 to 32 percent gap on a new case, these are already eliminated. Do
not spend time re-deriving them.

| Suspected cause | Test | Result |
|---|---|---|
| Geometry term | step study on the generator | exact to 1.0e-10 |
| Chain assembly | rigid translation, true answer zero | matches FD to 3e-7 on CMY |
| Mesh warp derivative | IDWarp forward vs a difference of its own warp | consistent to 1.2e-3 |
| Finite-difference noise | step study, h over a factor of 30 | FD moves < 1.6 percent |
| Duplicated node double counting | divide by multiplicity, re-chain | made it worse |
| Bad adjoint at the closure | normal displacement confined to a zone | aft closure agrees to 0.6-1.2 percent, no worse than the nose |

The diagnostics that produced these live in `remote/diagnostics/`.
