
# Installation instructions

## Building the PETSc and SLEPc libraries

### General remarks 

Time integration and basic linear algebra functionalities are provided by PETSc http://www.mcs.anl.gov/petsc/, derived packages such as SLEPc and external packages that can be interfaced to PETSc such as PARMETIS or MUMPS. 
In the beginning the user only needs to install and learn PETSc and then optionally add other packages later as needed. In the examples in the `example/` folder most examples do not require anything other than PETSc. 
For example the SLEPc linear algebra solvers can provide a speedup by many orders of magnitude compared to PETSc time integration when only steady states are considered. 
There is a large variety of possible numerical algorithms/strategies for PsiQuaSP quantum master equations through the PETSc and related/external packages. The performance of these algorithms is problem dependent and so should be the choice of method. 
In the examples some of these algorithms are introduced and how they can be used for the vectorized PsiQuaSP quantum master equations. But please note that solving large, sparse linear algebra problems via direct integration, 
factorization, iterative methods, etc. is a science by itself and the examples only provide an introduction to this field. The PETSc and related packages allow for simple solution strategies and advanced, 
specialized solution strategies that may outperform the simple strategies by orders of magnitude in computation time. 

### Installing PETSc

Go to the PETSc website https://www.mcs.anl.gov/petsc/ or to their bitbucket account https://bitbucket.org/petsc/petsc to download the current version. Installation PsiQuaSP relies on a standard PETSc installation, with certain arguments:
PsiQuaSP quantum master equations are complex valued equations so the PETSc installation needs the flag `--with-scalar-type=complex`. Also the occuring equations can be "ill conditioned" and preconditioning is something that needs a lot 
of expertise. Thus in order to get converged results it may be necessary to use quadruple precision, which is achieved by using the `--with-precision=__float128` flag - this is only possible with C and not C++, thus use `--with-clanguage=c`. 
Also if parallel operation is needed the user needs to make sure to set the corresponding flags in the PETSc installation i.e. set `--download-mpich`. There is an exhaustive documentation for the installation of PETSc on their website 
https://www.mcs.anl.gov/petsc/documentation/installation.html and we only wish to provide basic examples: The standard PETSc installation steps for PsiQuaSP would probably look like:

`export PETSC_DIR = /path/to/our/petsc/`

`export PETSC_ARCH = yourpetscbuildname`

`./configure --with-cc=gcc --with-cxx=g++ --with-fc=gfortran --download-mpich --with-clanguage=c --with-scalar-type=complex --with-precision=__float128 --download-f2cblaslapack`

and then follow the instructions. The `PETSC_ARCH` variable allows to have multiple PETSc bulids simultaneously, which is highly recommended. For instance there is a debug and a no-debug version of PETSc. 
The debug version allows the use of debuggers such as `gdb` however once the application code is stable the user should use the optimized PETSc build in order to get good performance. This is done by setting the `--with-debugging=0` 
flag in the PETSc configuration. So the user should have at least two PETSc builds one with and one without debugging. PARMETIS and other external packages are installed by setting the corresponding flag in the PETSc installation.
PsiQuaSP is tested for PETSc version 3.7.6

### Installing SLEPc

Once PETSc is configured the SLEPc build automatically uses the same options. You need to set the `SLEPC_DIR` variable to your SLEPc directory path (additionally to the `PETSC_DIR` and `PETSC_ARCH` variables) and then go to this folder, run
`./configure`
and then follow the instructions. Please note that the version numbers of PETSc and SLEPc have to be the same, otherwise this does not work. An example using SLEPc can be found in example/ex2b.

### Installing PsiQuaSP

Go to your PsiQuaSP folder and edit the `options.mk`. You only need to specify your `PETSC_DIR` and `PETSC_ARCH` varibles. SLEPc and other derived packages only need to be included at the application code level (see e.g. example/ex2b), 
it is not needed to build the PsiQuaSP library. Run
`make`
in the PsiQuaSP folder and the library will be build into a folder named `PETSC_ARCH`, same as with PETSc.

