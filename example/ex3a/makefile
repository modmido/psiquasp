
all: ex3a

include ../../options.mk

PQSP_DIR	= ../../
PQSP_LIB	= -L$(PQSP_DIR)/$(PETSC_ARCH)

ex3a: ex3a.cpp ex3a.hpp
	$(CC) $(CXXFLAGS) ex3a.cpp $(PETSC_INCLUDE) $(PQSP_LIB) ${PETSC_LIB} -lpsiquasp
	