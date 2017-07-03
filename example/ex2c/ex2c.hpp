
/**
 * @file	ex2c.hpp
 *
 * 		This example introduces the non RWA terms in the Dicke interaction Hamiltonian
 *    in the master equation.
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
