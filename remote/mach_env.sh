# MACH-Aero / ADflow environment
source /etc/profile.d/modules.sh 2>/dev/null || true
module load mpi/openmpi-x86_64 2>/dev/null
export PATH=$HOME/.local/bin:$PATH
export PETSC_DIR=$HOME/packages/petsc-3.21.6
export PETSC_ARCH=real-opt
export CGNS_HOME=$HOME/packages/CGNS-4.5.0/opt-gfortran
export PATH=$PATH:$CGNS_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CGNS_HOME/lib:$PETSC_DIR/$PETSC_ARCH/lib
source $HOME/mach-venv/bin/activate
