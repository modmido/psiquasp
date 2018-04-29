
all: ex6a

include ../../options.mk

PQSP_DIR	= ../../
PQSP_LIB	= -L$(PQSP_DIR)/$(PETSC_ARCH)

ex6a: ex6a.cpp ex6a.hpp
	$(CC) $(CXXFLAGS) ex6a.cpp $(PETSC_INCLUDE) $(PQSP_LIB) ${PETSC_LIB} -lpsiquasp
