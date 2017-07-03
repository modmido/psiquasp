
all: ex2b

CC		= g++
CXXFLAGS	= -std=c++0x -O3 -Wall

PETSC_DIR	= /home/gegg/petsc/petsccurrent/
PETSC_ARCH	= c-float128
SLEPC_DIR	= /home/gegg/slepc/slepccurrent/

include		${SLEPC_DIR}/lib/slepc/conf/slepc_common

PETSC_INCLUDE	= -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include
SLEPC_INCLUDE	= -I$(SLEPC_DIR)/include -I$(SLEPC_DIR)/$(PETSC_ARCH)/include

PQSP_DIR	= ../../
PQSP_LIB	= -L$(PQSP_DIR)/$(PETSC_ARCH)

ex2b: ex2b.cpp ex2b.hpp
	$(CC) $(CXXFLAGS) ex2b.cpp $(SLEPC_INCLUDE) $(PETSC_INCLUDE) ${SLEPC_EPS_LIB} $(PQSP_LIB) -lpsiquasp
	
