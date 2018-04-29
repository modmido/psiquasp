
all: ex6b

include ../../options.mk

PQSP_DIR	= ../../
PQSP_LIB	= -L$(PQSP_DIR)/$(PETSC_ARCH)

ex6b: ex6b.cpp ex6b.hpp
	$(CC) $(CXXFLAGS) ex6b.cpp $(PETSC_INCLUDE) $(PQSP_LIB) ${PETSC_LIB} -lpsiquasp
