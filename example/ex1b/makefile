
all: ex1b

include ../../options.mk

PQSP_DIR	= ../../
PQSP_LIB	= -L$(PQSP_DIR)/$(PETSC_ARCH)

ex1b: ex1b.cpp ex1b.hpp
	$(CC) $(CXXFLAGS) ex1b.cpp $(PETSC_INCLUDE) $(PQSP_LIB) ${PETSC_LIB} -lpsiquasp