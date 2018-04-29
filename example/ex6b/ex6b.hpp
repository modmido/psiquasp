
/**
 * @file	ex6b.hpp
 * 
 *  Two types of TLS which each interact with the same two radiation modes.
 * 
 * @author	Michael Gegg
 * 
 */
 
#include"../../include/psiquasp.hpp"

#define		hbar	0.658212196			// hbar in [eV*fs]

/*
 * This just provides a setup function
 */

class TwoTLS: public System
{
  public:
    PetscErrorCode	Setup(Vec * dm, Mat * AA);
    PetscErrorCode  MatJ10Left(Mat * out);
};


/*
 * Dervived class for the program output. Provides a capsule for all output related things.
 * Also just provides a setup function.
 */

class MyOut: public Output
{
  public:
    PetscErrorCode	SetupMyOut(TwoTLS * system);
};


/*
 * Class for the user specified observables output file, child of PropFile.
 * The only thing that needs to be specified here are the actual observables that should be printed to the file.
 */

class ObservablesFile: public PropFile
{
  public:
    PetscErrorCode	SetupMyObsFile(TwoTLS * system, std::string name);
};


/*
 * Class for the user specified correlation function file. Child of PropFile. 
 */

class CorrelationsFile: public PropFile
{
  public:
    PetscErrorCode	SetupMyGnFile(TwoTLS * sys, std::string name);			//set it up
};


/*
 * Class for the user specified, inter-tls gtwo function numerator < Jd Jd J J >, with J = J_{01,0}+J_{01,1} observable
 */

class InterGTwo: public PModular
{
public:
    PetscErrorCode    Setup(TwoTLS* sys);
};


/*
 * Class for the user specified, inter-tls gtwo function denominator < Jd J >, with J = J_{01,0}+J_{01,1} observable
 */

class InterGOne: public PModular
{
public:
    PetscErrorCode    Setup(TwoTLS* sys);
};
