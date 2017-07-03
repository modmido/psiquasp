
all: ex3b

include ../../options.mk

PQSP_DIR	= ../../
PQSP_LIB	= -L$(PQSP_DIR)/$(PETSC_ARCH)

ex3b: ex3b.cpp ex3b.hpp
	$(CC) $(CXXFLAGS) ex3b.cpp $(PETSC_INCLUDE) $(PQSP_LIB) ${PETSC_LIB} -lpsiquasp
	