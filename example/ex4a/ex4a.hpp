
/**
 * @file	ex4a.hpp
 * 
 * 		Application header for the phononlaser,lasercooling with many two-level systems. This example heavily relies on the modular AddMLSSingleArrowConnecting() and AddMLSSingleArrowNonconnecting() functions.
 * 		The functions for the basic matrices like TLS_J10left() are the same as the ones defined in the TLS class, but here we explicitly construct them again in order to illustrate the usage of the modular setup of arbitrary master equations.
 * 
 * @author	Michael Gegg
 * 
 */

#include"../../include/psiquasp.hpp"

#define		hbar	0.658212196			// hbar in [eV*fs]

class Phononlaser: public System
{
  public:
    PetscErrorCode	Setup(Vec * dm, Mat * AA);
    
    PetscErrorCode	H0Part(Mat AA);
    PetscErrorCode	CoherentDrivePart(Mat AA);
    PetscErrorCode	ElectronPhononPart(Mat AA);
    PetscErrorCode	DissipationPart(Mat AA);
    PetscErrorCode	NoRWA1(Mat AA);
    PetscErrorCode	NoRWA2(Mat AA);
    
    PetscErrorCode	TLS_J10left(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscScalar matrixelem);
    PetscErrorCode	TLS_J01left(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscScalar matrixelem);
    PetscErrorCode	TLS_J10right(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscScalar matrixelem);
    PetscErrorCode	TLS_J01right(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscScalar matrixelem);
    
    PetscErrorCode	TLS_J11left(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscScalar matrixelem);
    PetscErrorCode	TLS_J11right(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscScalar matrixelem);
};

class MyObsFile: public PropFile
{
  public:
    PetscErrorCode	SetupMyObsFile(Phononlaser * system, std::string name, PetscReal rotfreq, std::string param);		//set up everything function
};

class MyGnFile: public PropFile
{
  public:
    PetscErrorCode	SetupMyGnFile(Phononlaser * sys,std::string name,std::string param);			//set it up
};

class MyOut: public Output
{
  public:
    PetscErrorCode	SetupMyOut(Phononlaser * system, std::string param);							//not using default constructor since we want a PetscErrorCode return type!
};
