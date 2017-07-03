
/**
 * @file	ex2b.hpp
 * 
 * 		In this example we solve a two-level laser theory using the eigensolver provided by slepc. Apart form the different solution method the example is identical to ex1a.
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

class Laser: public TLS
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
    PetscErrorCode	SetupMyOut(Laser * system);
};


/*
 * Class for the user specified observables output file, child of PropFile.
 * The only thing that needs to be specified here are the actual observables that should be printed to the file.
 */

class ObservablesFile: public PropFile
{
  public:
    PetscErrorCode	SetupMyObsFile(Laser * system, std::string name);
};


/*
 * Class for the user specified correlation function file. Child of PropFile. 
 */

class CorrelationsFile: public PropFile
{
  public:
    PetscErrorCode	SetupMyGnFile(Laser * sys, std::string name);			//set it up
};


/*
 * Class for the user specified < J_{10} b > observable, which is usually encountered when deriving a laser rate equation theory, makes these theories comparable
 */

class J10b: public PModular
{
  public:
    PetscErrorCode	Setup(Laser* sys);
};


/*
 * Class for the user specified < J_{11} b^\dagger b > observable, which is usually factorized when deriving a laser rate equation theory, this way we can check the range of validity of the rate equation theory.
 */

class J11bdb: public PModular
{
  public:
    PetscErrorCode	Setup(Laser* sys);
};
