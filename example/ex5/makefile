
all: ex5

include ../../options.mk

PQSP_DIR	= ../../
PQSP_LIB	= -L$(PQSP_DIR)/$(PETSC_ARCH)

ex5: ex5.cpp ex5.hpp
	$(CC) $(CXXFLAGS) ex5.cpp $(PETSC_INCLUDE) $(PQSP_LIB) ${PETSC_LIB} -lpsiquasp
	