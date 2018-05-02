#####

#
# User options
# The user should use the same compiler as in the PETSc installation
#

CC		= g++
CXXFLAGS	= -std=c++0x -O0 -g -Wall

#####
#
# Here the user needs to specify the PETSc environment variables 
#

PETSC_DIR	= /yourpetscdir/
PETSC_ARCH	= yourpetscarch

include		$(PETSC_DIR)/lib/petsc/conf/variables
include		$(PETSC_DIR)/lib/petsc/conf/rules

PETSC_INCLUDE   = -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include
