
all: ex4a

include ../../options.mk

PQSP_DIR	= ../../
PQSP_LIB	= -L$(PQSP_DIR)/$(PETSC_ARCH)

ex4a: ex4a.cpp ex4a.hpp
	$(CC) $(CXXFLAGS) ex4a.cpp $(PETSC_INCLUDE) $(PQSP_LIB) ${PETSC_LIB} -lpsiquasp
	