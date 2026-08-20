# Building ADflow and MACH-Aero on a Linux host, without sudo

This recipe is what actually worked on the reference host: RHEL 8.9, 64 cores, 125 GB RAM,
no root access. Every workaround below exists because something failed without it.

**Check first whether the stack is already built.** On the reference host it is:

```bash
ssh 192.168.20.10
source ~/mach_env.sh
python -c "import adflow, pygeo, idwarp, pyspline, baseclasses, pyhyp, cgnsutilities; print('ok')"
```

If that prints `ok`, skip this document.

## Versions that work

These are the MDO Lab "stable" combination. Do not mix others in casually.

| Component | Version | Note |
|---|---|---|
| OpenMPI | 4.1.1 | already present as an RPM: `module load mpi/openmpi-x86_64`, **no sudo** |
| PETSc | 3.21.6 | only the `real-opt` arch is needed |
| CGNS | 4.5.0 | `CGNS_ENABLE_HDF5=OFF`, `ENABLE_64BIT=OFF`, `ENABLE_FORTRAN=ON` |
| Python | 3.11 | via `uv`, because the box has no `python3.11-devel` |

## The environment file

Everything downstream depends on this. Keep it at `~/mach_env.sh`; a reference copy is in
`remote/mach_env.sh`.

```bash
source /etc/profile.d/modules.sh 2>/dev/null || true
module load mpi/openmpi-x86_64 2>/dev/null
export PATH=$HOME/.local/bin:$PATH
export PETSC_DIR=$HOME/packages/petsc-3.21.6
export PETSC_ARCH=real-opt
export CGNS_HOME=$HOME/packages/CGNS-4.5.0/opt-gfortran
export PATH=$PATH:$CGNS_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CGNS_HOME/lib:$PETSC_DIR/$PETSC_ARCH/lib
source $HOME/mach-venv/bin/activate
```

## Step 1: PETSc

There is no system BLAS or LAPACK, so PETSc must download its own. Missing
`--download-fblaslapack=1` is the first failure.

```bash
./configure --with-shared-libraries --download-superlu_dist \
  --download-parmetis=yes --download-metis=yes --with-fortran-bindings=1 \
  --with-debugging=0 --with-scalar-type=real --PETSC_ARCH=real-opt \
  --with-cxx-dialect=C++11 --download-fblaslapack=1
```

Build only `real-opt`. Skipping the debug and complex arches saves about 30 minutes.

## Step 2: Python 3.11 with headers, no sudo

The box has no Python headers (`python3.11-devel` needs root). `uv` installs a
python-build-standalone interpreter in userspace, which ships headers.

```bash
uv python install 3.11
uv venv --python 3.11 ~/mach-venv
```

Then **two extra steps the MDO Lab Makefiles require**, without which `pyspline` and
`cgnsutilities` fail with `python3-config: Command not found` and then
`Python.h: No such file or directory`:

```bash
# 1. the Makefiles call python3-config and python3.11-config
ln -sf "$(command -v python3-config)"    ~/mach-venv/bin/python3-config
ln -sf "$(command -v python3.11-config)" ~/mach-venv/bin/python3.11-config

# 2. the venv's include dir is empty; point it at the real headers
ln -sfn <uv-python-base>/include/python3.11 ~/mach-venv/include/python3.11
```

## Step 3: the MDO Lab repos, in order

Order matters, because later packages import earlier ones at build time.

```
baseclasses -> pyspline -> pygeo -> cgnsutilities -> idwarp -> pyhyp -> adflow
```

For each compiled repo:

```bash
cp config/defaults/config.LINUX_GFORTRAN.mk config/config.mk
make && pip install .
```

## Step 4: verify

```bash
source ~/mach_env.sh
python -c "import adflow, pygeo, idwarp, pyspline, baseclasses, pyhyp, cgnsutilities; print('ok')"
```

Then run the tutorial-wing validation in `docs/VALIDATION.md`. If those numbers do not
reproduce, stop and fix the build before running any real case.

## Still missing: pyOptSparse

pyOptSparse is the optimizer wrapper and it is **not on PyPI**. Without it you can compute
gradients but not drive an optimization.

```bash
pip install "pyoptsparse @ git+https://github.com/mdolab/pyoptsparse.git"
```

That build provides SLSQP, ALPSO, NSGA2 and CONMIN. SLSQP is enough for a handful of
design variables. SNOPT needs a licence; IPOPT needs a separate build.

## Runtime noise

`mca_base_component_repository_open ... libfabric.so.1` warnings are harmless. Silence
them with `--mca pml ob1 --mca btl self,vader,tcp`, which the drivers pass automatically
from `remote.mpi_flags`.
