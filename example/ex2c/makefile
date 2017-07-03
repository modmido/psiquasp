
all: ex2c

include ../../options.mk

PQSP_DIR	= ../../
PQSP_LIB	= -L$(PQSP_DIR)/$(PETSC_ARCH)

ex2c: ex2c.cpp ex2c.hpp
	$(CC) $(CXXFLAGS) ex2c.cpp $(PETSC_INCLUDE) $(PQSP_LIB) ${PETSC_LIB} -lpsiquasp
