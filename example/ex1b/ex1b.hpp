
/**
 * @file	ex1b.hpp
 * 
 * 		Same as ex1a.hpp, except that all setup functions now have return type PetscErrorCode. Also the mode is coupled to a thermal bath instead of a vacuum bath. This enables Petsc error handling. Petsc error handling prints error messages into stderr if sth goes wrong.
 * 		Petsc/Slepc and also PsiQuaSP come with a lot of sanity checks based on this. So in order to understand why the code crashes and to speed up debugging it is recommended to use these petsc error handling features.
 * 
 * @author	Michael Gegg
 * 
 */
 
#include"../../include/psiquasp.hpp"

#define		hbar	0.658212196			// hbar in [eV*fs]

/*
 * Derived class for the open Tavis-Cummings model (OTC).
 * This just provides a setup function
 */

class OTC: public TLS
{
  public:
    PetscErrorCode	Setup(Vec * dm, Mat * AA);
};


/*
 * Dervived class for the program output. Provides a capsule for all output related things.
 * Also just provides a setup function.
 */

class MyOut: public Output
{
  public:
    PetscErrorCode	SetupMyOut(OTC * system);
};


/*
 * Class for the user specified observables output file, child of PropFile.
 * The only thing that needs to be specified here are the actual observables that should be printed to the file.
 */

class ObservablesFile: public PropFile
{
  public:
    PetscErrorCode	SetupMyObsFile(OTC * system, std::string name);
};


/*
 * Class for the user specified correlation function file. Child of PropFile. 
 */

class CorrelationsFile: public PropFile
{
  public:
    PetscErrorCode	SetupMyGnFile(OTC * sys, std::string name);			//set it up
};
