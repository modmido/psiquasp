
/**
 * @file	ex3a.hpp
 * 
 * 		This example solves the three-level system master equation shown in the PsiQuaSP paper. The three-level systems are in \f$ \Lambda \f$ configuration, one transition is driven by an external classical laser and the other transition couples to
 * 		a bosonic mode. In cw operation the steady state of the system is all three-level systems in the lower level of the transition that is coupled to the mode, again trivial.
 * 		In this example we introduce the usage of the more general System class functions for setting up general master equations.
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

class Lambda: public System
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
    PetscErrorCode	SetupMyOut(Lambda * system);
};


/*
 * Class for the user specified observables output file, child of PropFile.
 * The only thing that needs to be specified here are the actual observables that should be printed to the file.
 */

class ObservablesFile: public PropFile
{
  public:
    PetscErrorCode	SetupMyObsFile(Lambda * system, std::string name);
};


/*
 * Class for the user specified correlation function file. Child of PropFile. 
 */

class CorrelationsFile: public PropFile
{
  public:
    PetscErrorCode	SetupMyGnFile(Lambda * sys, std::string name);			//set it up
};


