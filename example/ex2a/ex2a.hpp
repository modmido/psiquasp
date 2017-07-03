
/**
 * @file	ex2a.hpp
 * 
 * 		This example introduces the custom observables derived from PModular. The master equation is a two-level laser, which has a nontrivial steady state. 
 * 		Here we use time integration, in ex2b we will find the steady state using SLEPc eigensolvers.
 * 
 * @author	Michael Gegg
 * 
 */
 
#include"../../include/psiquasp.hpp"

#define		hbar	0.658212196			// hbar in [eV*fs]


class Laser: public TLS
{
  public:
    PetscErrorCode	Setup(Vec * dm, Mat * AA);
};

class MyOut: public Output
{
  public:
    PetscErrorCode	SetupMyOut(Laser * system);
};

class ObservablesFile: public PropFile
{
  public:
    PetscErrorCode	SetupMyObsFile(Laser * system, std::string name);
};

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
