# Gotchas

Every item here cost real time to find. They are ordered by how badly they bite. Items
marked **SILENT** produce a plausible wrong answer instead of an error, which makes them
the dangerous ones.

## Solver

### SILENT: `liftindex` must match your geometry
ADflow defaults to `liftIndex = 2`, which means lift along **y**. nTop aircraft geometry is
normally z-up with span along y. With the default, the number ADflow calls CL is actually
**side force**. On the AGARD-C body that was 0.0052 instead of the correct 0.0089.

`config.py` refuses to build a case unless `flow.liftindex` is set explicitly, because
there is no safe default.

### SILENT: an inviscid closed body must produce zero drag
At a subcritical Mach number, a closed body in Euler flow has exactly zero drag: the
fore-and-aft pressure forces cancel. Any CD you get is discretization error. On the
AGARD-C body at Mach 0.75 the mesh reported CD = 0.0423, which is 100 percent error.

Consequences:
- Never quote that CD as drag.
- Never run a drag minimization on such a mesh. You would be optimizing mesh error.
- Use CL, CMY or a target pressure distribution instead, or add a wing, go supersonic
  where wave drag is real, or switch to RANS.

### ADflow reads structured multiblock or overset CGNS only
It cannot read the unstructured tet/prism CGNS that AFLR3 and FUN3D produce. Do not try.
A structured surface seed is required, which is the hard part of this whole pipeline.

## Surface topology

### A closed body needs a cubed sphere, and N must be EVEN
A single polar patch collapses the nose and the tail to a point, giving zero-area cells.
Two topologies failed on the AGARD-C fuselage before the cubed sphere worked:

| Attempt | Result |
|---|---|
| Half body, collapsed nose/tail edges, `unattachedEdgesAreSymmetry=True` | "free corner or other topology that is not an edge" |
| Full periodic single patch | segfault, 12,384 bad cells, metrics `0.180+309` (infinite) |
| **6-patch cubed sphere, N even** | 65 layers, **0 bad cells**, min quality 0.915 |

With N odd a grid point lands exactly on a pole and that cell degenerates. `surface.Body`
raises on an odd N.

### Build the six patches as rotations of ONE base face
`normalize((tan u, tan v, 1))` for u, v in [-pi/4, pi/4], then apply the six rotations.
That is what makes the shared patch edges coincide to machine precision. Building each
face independently leaves gaps and pyHyp reports a free corner.
`Body.health()` checks the shared-edge gap is 0.

### All patch normals must point outward
Mixed normals make pyHyp mark **every** cell bad from the first layer. The generator flips
the j index on any patch whose `cross(t_i, t_j) . radial` is negative.

### SILENT: freeze the normal-flip decision
Recomputing the flip for each perturbed shape can invert one patch at `p+h` but not at
`p-h`. Node correspondence is then scrambled with no error message, and `dX_surf/dp` is
quietly wrong. `Body` decides the flips once at the baseline and reuses them.

### pyHyp needs `fileType: PLOT3D`
For `.fmt` and `.xyz` input. Without it pyHyp tries to parse the file as CGNS and dies
with `Bad integer for item 1 in list input`.

### Reading the extrusion log
Bad cells must be **0** and minimum quality about **0.9**. A segfault or a metric printed
as `0.180+309` means the SURFACE topology is wrong. It does not mean the solver is broken.

### Overset assembly, for wings and other lifting surfaces
Not implemented in this repo. This is the route, taken from
`MACH-Aero/tutorial/overset/mesh/run_pyhyp.py`:

1. One pyHyp near-body extrusion per component, each with a SHORT `marchDist`. A wing needs
   its own structured surface; the cubed sphere here cannot make one.
2. `from cgnsutilities.cgnsutilities import readGrid, combineGrids`, then
   `combineGrids([...])`.
3. `.symmZero("y")` if the case is symmetric.
4. `simpleOCart(nearfield, dhStar, 40.0, nFarfield, "y", 1, farfield)` for the Cartesian
   background grid.
5. Combine near field and far field.
6. ADflow then needs `surfaceFamilyGroups` mapping each component family to `wall`.

The adjoint works on overset meshes, but note that `getSurfaceCoordinates` grows a zipper
mesh contribution there, so the node-matching logic in `gradients.py` needs re-checking
before the gradients can be trusted on an overset case.

## Geometry measurement

### SILENT: do not measure the body radius at the equator
On a wing-body geometry the wing also sits near z = 0, so a band around the equator reads
the **wing span**, not the body radius. On AGARD-C that is 0.61 m instead of 0.18 m.
Measure the bottom of a narrow centreplane band, which is clear of the wing above it.

### SILENT: pick the ry/rz cross-check station by hand
A slice at mid-body on AGARD-C returns ry/rz = 2.16 because the wing is there. The true
body value is 1.234. At x = 0.60, forward of the wing root leading edge, the measurement
gives 1.2307 against the notebook's FUSE-Y/ZScale value of 1.230, which is the agreement
that confirms the measurement reads the body. There is no way to find such a station
automatically, so `geometry.ratio_at_x` must be set by a person.

### Anchor the profile spline at the body ends, not the sampling range
If the first sample sits exactly on the sampling range start you get a duplicate knot, and
`UnivariateSpline` then returns **NaN with no error**. Anchor at `x_nose` and `x_tail` with
zero radius, which is also where the body physically closes.

## Adjoint gradients

### SILENT and severe: the surface derivative needs BOTH seeds
`computeJacobianVectorProductBwd(funcsBar=..., xSDeriv=True)` alone returns the
**frozen-flow partial** derivative. It ignores how the pressure field responds to the shape
change. The total derivative is

```python
CFDSolver.solveAdjoint(ap, f)
psi = -CFDSolver.getAdjoint(f)
dJdX = CFDSolver.computeJacobianVectorProductBwd(
    resBar=psi, funcsBar=CFDSolver._getFuncsBar(f), xSDeriv=True)
```

which is what `evalFunctionsSens` does internally. Measured on AGARD-C, the partial was
wrong by up to **17 times** and wrong in **sign** for 2 of 9 pairs. It is a trap because
under a width scaling it reproduces CL and CD to **nine digits**, so it looks correct.
`gradients.py` reports both so the difference stays visible.

### Do not deduplicate the surface gradient
ADflow exposes 6144 surface nodes on a 32x32 cubed sphere; IDWarp's internal surface has
5768. The difference is the 744 patch-edge nodes stored once per patch (720 doubled, 24
tripled). ADflow returns a **partial contribution** on each copy, so summing the copies is
correct. Dividing by multiplicity made the median error against finite difference worse,
from 5.0e-2 to 5.8e-2, and one pair went from 25 percent to 443 percent.

### Verify by warping, never by re-extruding
pyHyp rebuilds the volume mesh instead of warping it, so a re-extruded finite difference
changes the discretization as well as the shape. Use `setSurfaceCoordinates` then
`updateGeometryInfo`, which is the chain the adjoint actually linearizes.

### A standalone `USMesh` segfaults in `warpDerivFwd`
It also builds its own internal surface, merging the duplicated nodes. Attach the mesh to
ADflow with `setMesh` first, so ADflow supplies the surface definition and the external
mesh indices.

### Read K before trusting a gradient
`K = sum|c_i| / |sum c_i|` where `c_i = (dJ/dX_i).(dX_i/dp)` measures how much the
projection relies on cancellation. On the 32x32 AGARD-C body:

| K | Outcome |
|---|---|
| <= 63 | all four pairs verified to better than 2 percent |
| >= 376 | all five pairs failed by 5 to 32 percent |
| 63 to 376 | no pair landed here |

Rank correlation between K and the observed error was 0.983 over four decades. K costs
nothing: both terms are already in hand after the adjoint solve, so no extra flow solve
and no finite difference is needed. `gradients.py` prints it and flags `K > k_warn`.

Do **not** treat K as a quantitative law. A linear `K x 1 percent` model over-predicts by
one to two orders at large K, because the nodal errors partly cancel among themselves. The
ordering is reliable; the magnitude is not.

### What the finite difference can and cannot tell you
It confirms the derivative of the **discrete** functional. It cannot confirm the physics:
the adjoint and the finite difference share one mesh and inherit the same discretization
error. A gradient can match its own discrete model exactly and still point the wrong way
for the real flow.

## Environment

### Always pass the mpirun flags
`--mca pml ob1 --mca btl self,vader,tcp` silences harmless
`mca_base_component_repository_open ... libfabric.so.1` warnings from the OpenMPI RPM.
The drivers read them from `remote.mpi_flags`.

### ntopcl: the exit code is not a success signal
It has been seen as 0 on success on nTop 5.49/5.50 and as 72 on success on older setups.
Gate on the expected **output files** appearing. A hung run still writes early artifacts,
so the presence of a STEP or a JSON alone is not success either.

### ntopcl can hang after meshing
The mesh finishes, export never completes, the process stays alive with no progress and no
error. Always use a hard timeout plus retry. Some design points hang deterministically at
any timeout; that is a notebook meshing defect and no retry will fix it.

### ntopcl needs `--trustnotebook` to write files
Without it, Run-Command and export blocks quietly no-op. You get no geometry and no error.

### .ntop files are not backward-openable
A notebook saved in a newer nTop fails to load in an older one with a misleading error.
`AGARD C.ntop` was saved newer than the nTop 5.37.3 on the RHEL box, so it must run on the
Windows workstation where `ntopcl.exe` is 5.53.2.

### Credentials
`ntopcl` logs in on every invocation; there is no cached token in headless use. Set
`NTOP_USERNAME` and `NTOP_PASSWORD`. Watch shell escaping: a password stored with a stray
backslash before `!` is rejected as a bad password.
