
# PsiQuaSP -- Permutation symmetry for identical Quantum Systems Package

PsiQuaSP is a library that enables easy and quick setup of quantum optical Lindblad quantum master equation simulations for PETSc and SLEPc. 
The main feature of the library is the ability to treat permutationally symmetric many multi-level system setups with polynomial instead of exponential complexity. 
The methodology of this approach is described in:

M. Richter, M. Gegg, TS. Theuerholz and A. Knorr, Phys. Rev. B 91, 035306 (2015) https://arxiv.org/abs/1412.0559

M. Gegg and M. Richter, New. J. Phys. 18, 043037 (2016) http://iopscience.iop.org/article/10.1088/1367-2630/18/4/043037/meta

M. Gegg and M. Richter, arXiv 1707.01079 (2017) https://arxiv.org/abs/1707.01079

The library only provides setup functions for the master equation and the program output, the computing stage is solely done by the PETSc and SLEPc packages. 
PETSc and SLEPc are state of the art sparse linear algebra and differential equations packages that are continuously improved. 
This makes PsiQuaSP computations very fast and efficient and automatically enables the user to employ the latest and fastest solution algorithms.
PsiQuaSP uses a vectorized form of the Lindblad equation, treating density matrices as vectors and Liouville superoperators as matrices. 
This enables the library to directly interface to the entirety of the PETSc/SLEPc linear algebra and time integration methods. 
The PETSc matrices are sparse by default, making PsiQuaSP very storage efficient. PETSc and SLEPc support MPI distributed memory parallelism, thus providing good scalabiltiy of the library.
PETSc/SLEPc also provide interfaces to a large variety of additional, external packages, e.g. serial and parallel, sparse and dense linear algebra packages such as LAPACK, ARPACK, MUMPS, etc. 
or graph paritioning libraries for optimal parallel performance such as METIS/PARMETIS, SCOTCH, etc. Thus PETSc/SLEPc provides an incredibly rich and versatile toolbox for all sorts of differential 
equation/linear algebra computations and PsiQuaSP is designed in a way that provides the user with maximal flexibility in the choice of the solution methods.

Installation instructions can be found in the INSTALL.md file.

PsiQuaSP uses Doxygen commenting by Dimitri van Heesch, http://www.stack.nl/~dimitri/doxygen/. Install doxygen and run

`doxygen Doxyfile`

in the PsiQuaSP folder. The doxumentation will be build into the `doc/` folder.
Example codes introducing a large portion of the features of PsiQuaSP are in the `example/` folder.


PsiQuaSP Team: Michael Gegg and Marten Richter

Theoretical, algorithmic, and numerical concepts and strategies: Michael Gegg and Marten Richter

Implementation: Michael Gegg
