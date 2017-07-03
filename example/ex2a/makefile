all: ex2a

include ../../options.mk

PQSP_DIR	= ../../
PQSP_LIB	= -L$(PQSP_DIR)/$(PETSC_ARCH)

ex2a: ex2a.cpp ex2a.hpp
	$(CC) $(CXXFLAGS) ex2a.cpp $(PETSC_INCLUDE) $(PQSP_LIB) ${PETSC_LIB} -lpsiquasp
	