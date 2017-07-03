all: ex1a

include ../../options.mk

PQSP_DIR	= ../../
PQSP_LIB	= -L$(PQSP_DIR)/$(PETSC_ARCH)

ex1a: ex1a.cpp ex1a.hpp
	$(CC) $(CXXFLAGS) ex1a.cpp $(PETSC_INCLUDE) $(PQSP_LIB) ${PETSC_LIB} -lpsiquasp
	
