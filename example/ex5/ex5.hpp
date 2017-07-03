
/**
 * @file	ex5.hpp
 * 
 * 		This example solves the three-level system master equation shown in the PsiQuaSP paper, same as in ex3a. However here we apply Parmetis reordering in order to achieve a reduction in the degrees of freedom of the master equation itself.
 * 		In some master equations additional symmetries apply, other than the permutational symmetry. These symmetries may lead to a decoupling of coherence degrees of freedom, similar to what can be seen in the sketches, but less obvious. By using
 * 		Parmetis to interpret the Liouvillian matrix as a graph and finding disconnected partitionings, we can eliminate (parts) of the dead parts of the system. We always choose the partition containing the ground state and eliminate the rest.
 * 		Parmetis produces a load balanced paritioning, which is intended for optimizing parallel performance, but is actually not well suited for the current purpose. Nonetheless it can be used to reduce the complexity by about an order in the polynomial
 * 		scaling in this example, i.e. from N^8 to N^7. This again is not an approximation.
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
    PetscErrorCode	Setup(Vec * dm, Mat * AA, Mat * PP, PetscInt *dim);
    
    PetscErrorCode	SetLiouvillian(Mat AA);
    PetscErrorCode	SetLiouvillianOne(Mat AA);
    
    PetscErrorCode	ParmetisReduce(Mat ADJ, Mat* PP, PetscInt* dim);					//two different strategies, since the Parmetis tool is not ideal, since it tries to chop the matrix into chunks of equal (!) size, which is bad,
    PetscErrorCode	ParmetisReduce2(Vec dmfull, Mat LFULL, Vec* dm, Mat* LL, Mat* PP, PetscInt* dim);	//actually a graph search algorithm that just looks for the connectivity graph containing the ground state would be best.
    PetscErrorCode	MatProject(IS is, Mat *AA, PetscInt *dim);						//create a projection matrix out of the index set, the index set is the output of the parmetis reordering
    
    //the downprojection requires custom observalbe types
    PetscErrorCode	MatId(Mat *AA, Mat PP, PetscScalar matrixelem);
    PetscErrorCode	MatJ11Left(Mat *AA, Mat PP, PetscScalar matrixelem);
    PetscErrorCode	MatJ22Left(Mat *AA, Mat PP, PetscScalar matrixelem);
    PetscErrorCode	MatJ10Left(Mat *AA, Mat PP, PetscScalar matrixelem);
    PetscErrorCode	MatJ21Left(Mat *AA, Mat PP, PetscScalar matrixelem);
    PetscErrorCode	MatJ20Left(Mat *AA, Mat PP, PetscScalar matrixelem);
    PetscErrorCode	MatBdBLeft(Mat *AA, Mat PP, PetscScalar matrixelem);
    PetscErrorCode	MatBLeft(Mat *AA, Mat PP, PetscScalar matrixelem);
};


/*
 * Dervived class for the program output. Provides a capsule for all output related things.
 * Also just provides a setup function.
 */

class MyOut: public Output
{
  public:
    PetscErrorCode	SetupMyOut(Lambda * system,Mat PP,PetscInt dim);
};


/*
 * Class for the user specified observables output file, child of PropFile.
 * The only thing that needs to be specified here are the actual observables that should be printed to the file.
 */

class ObservablesFile: public PropFile
{
  public:
    PetscErrorCode	SetupMyObsFile(Lambda * system, std::string name, Mat PP,PetscInt dim);
};

class MyObs: public PModular
{
  public:
    PetscErrorCode	Setup(Lambda* system, Mat AA, Mat PP, PetscInt dim, PetscInt herm, PetscReal inshift, PetscReal inomega);
};

/*
 * Class for the user specified correlation function file. Child of PropFile. 
 */

class CorrelationsFile: public PropFile
{
  public:
    PetscErrorCode	SetupMyGnFile(Lambda * sys, std::string name);			//set it up
};


