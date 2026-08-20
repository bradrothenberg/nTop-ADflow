# Sourced by every driver. Loads MPI, PETSc, CGNS and the MACH-Aero venv, then reads the
# rank count and mpirun flags out of case.json so they live in exactly one place.
# A reference copy of the host's ~/mach_env.sh is kept at remote/mach_env.sh.
source ~/mach_env.sh
CASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$CASE_DIR"
NP="$(python -c "import json;print(json.load(open('case.json'))['remote'].get('np',8))")"
MPIF="$(python -c "import json;print(json.load(open('case.json'))['remote'].get('mpi_flags','--mca pml ob1 --mca btl self,vader,tcp'))")"
# mpi_flags silences the harmless libfabric.so.1 warnings from the OpenMPI RPM.
run_mpi () { mpirun -np "$NP" $MPIF python "$@"; }
