#####

#
# User options
# The user should use the same compiler as in the PETSc installation
#

#CC		= clang++
#CXXFLAGS	= -Wc++11-extensions
CXXFLAGS	= -std=c++98

#####
#
# Here the user needs to specify the PETSc environment variables 
#

PETSC_DIR	= /Users/michael/dev/petsc
PETSC_ARCH	= c-double-complex-debug

include		$(PETSC_DIR)/lib/petsc/conf/variables
include		$(PETSC_DIR)/lib/petsc/conf/rules

PETSC_INCLUDE   = -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include
