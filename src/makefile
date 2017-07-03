
#
# makefile acronym library
#

include ../options.mk

SRC += index.cpp
SRC += dim.cpp
SRC += sutil.cpp
SRC += sliouville.cpp
SRC += smodular.cpp
SRC += output.cpp
SRC += observables.cpp
SRC += gnfcts.cpp
SRC += distributions.cpp
SRC += twolevelsystems.cpp


OBJ = $(SRC:%.cpp=%.o)

all: psiquasp

psiquasp: $(OBJ)
	@mkdir ../$(PETSC_ARCH)
	@ar rs ../$(PETSC_ARCH)/libpsiquasp.a $+
	@rm *.o
	@echo
	@echo "  PsiQuaSP library build successful"
	@echo
	

%.o: %.cpp
	@echo "  Compiling src/$< with optimizations"
	$(eval COMMAND = $(CC) $(CXXFLAGS) -c $< $(PETSC_INCLUDE) ${PETSC_LIB} )
	@$(COMMAND) || (echo "Failure while executing command: $(COMMAND)" && exit 1)
