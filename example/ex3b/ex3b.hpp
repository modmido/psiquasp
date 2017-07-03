

/**
 * @file	ex3b.hpp
 * 
 * 		This example solves the three-level laser master equation shown in the PsiQuaSP paper.
 * 
 * @author	Michael Gegg
 * 
 */
 
#include"../../include/psiquasp.hpp"

#define		hbar	0.658212196			// hbar in [eV*fs]


class Laser: public System
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


