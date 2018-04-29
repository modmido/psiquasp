
/**
 * @file	observables.hpp
 * 
 * 		Header file for the Observable class.
 * 
 * @author	Michael Gegg
 * 
 */


#ifndef _Observable
#define _Observable

#include"system.hpp"
#include"output.hpp"


/**
 * @brief	Enables computation of all "simple" observables.
 * 
 */

class Observable: public PropBase
{
  protected:
    
  //-------------------------------------------------------------------------------------
  //internal variables needed for expecation value computation
  //-------------------------------------------------------------------------------------
    PetscReal		shift;							    //!< possible shift of the final(global) result, so far only used to compute Tr[rho]-1, where the shift is -1, otherwise set to zero.
    PetscInt		freqcomponents;						//!< the number of different frequency components, arises from the rotating frame back transformation time dynamics
    PetscInt		*lengths;						    //!< the lengths of the local arrays
    PetscInt		**dmindex;						    //!< the indices of the relevant local dm entries
    PetscReal		*domega;						    //!< phase velocities of the rotating frame backtransformation, one per frequency component
    PetscReal		**prefactor;						//!< the prefactors that have to be multiplied with the respective entries
    
    
  //-------------------------------------------------------------------------------------
  //storage
  //-------------------------------------------------------------------------------------
    PetscErrorCode	AllocateLocStorage(const PetscInt Freqco, const PetscInt *Lengths, const PetscReal *Domega);
    
  public:
  //-------------------------------------------------------------------------------------
  //computation and setup of observables
  //-------------------------------------------------------------------------------------
    Observable()	{ isherm = 0; alloc = 0; }				//!< default constructor, isherm = 0
    ~Observable();
    
    PetscErrorCode	Compute(Vec dm,PetscReal time,PetscScalar * ret,PetscInt number);		    //compute the expecation value, only first processor gets the answer
    PetscErrorCode	ComputeAll(Vec dm,PetscReal time,PetscScalar * ret,PetscInt number);		//compute the expecation value, all processors get the answer
    
    //basic
    PetscErrorCode	SetupTrMinus1(System * sys);							//setup the trace
    PetscErrorCode	SetupTr(System * sys);
    
    //mls
    PetscErrorCode	SetupMlsOccupation(System * sys,MLSDim& mlsdens);
    PetscErrorCode	SetupMlsPolarization(System * sys, MLSDim& mlsop, PetscReal freq);
    PetscErrorCode	SetupMlsHigherPolarization(System * sys, MLSDim& mlsop, PetscInt order, PetscReal freq);
    PetscErrorCode	SetupMLSOccupationFull(System * sys, MLSDim& mlsop);
    PetscErrorCode	SetupMLSIntercoupling(System * sys, MLSDim& mlsop);
    PetscErrorCode	SetupMlsJzDiff(System * sys,MLSDim& Mlsdens1, MLSDim& Mlsdens2);
    PetscErrorCode	SetupMlsJzSquaredNorm(System * sys,MLSDim& Mlsdens1, MLSDim& Mlsdens2);
    PetscErrorCode	SetupMlsJzNorm(System * sys,MLSDim& Mlsdens1, MLSDim& Mlsdens2);
    PetscErrorCode	SetupTotalSpin(System * sys,MLSDim& Mlsdens1, MLSDim& Mlsdens2);
    
    //mode
    PetscErrorCode	SetupModeOccupation(System * sys,PetscInt mode);
    PetscErrorCode	SetupModePolarization(System * sys, PetscInt mode, PetscReal freq);

    //combined, not implemented yet
    PetscErrorCode	SetupCombOccupation(System * sys, MLSDim& mlsop, PetscInt mode);
    PetscErrorCode	SetupCombPolarization(System * sys, MLSDim& mlsop, PetscInt mode);
    
};

/**
 * @brief	Class for all custom made modular system properties like observables, correlation functions<br>
 * 
 * 		This class does not support possible phase factors arising from a rotating frame backtransformation.
 * 		In order to allow for such rotating frame backtransformation phase factors the user would need the separate the operators according to the different phase velocites and use the System::VecContractXXX() functionalities to transform the linear
 *		functional vectors for the different frequency components of the observable into the array format of the standard Observable class (see above).
 * 		to the normal Observables class.
 * 
 */

class PModular: public PropBase
{
  protected:
    Vec		left;							        //!< the trace vector times the O_left matrix, where O ist the quantity whose expect. value shall be computed i.e. < tr | O_left
    PetscScalar	omega;							    //!< possible rotating frame angular frequency
    PetscReal	shift;							    //!< possible shift, like in Tr(\rho)-1
    
    PetscErrorCode	GenerateLeft(System * sys, Mat AA);		//!< computes < tr | AA and stores it in left, also creates left
    PetscErrorCode	LeftOverwrite(System * sys, Mat AA);	//!< overwrites and existing left vector
    PetscErrorCode	LeftUpdate(System * sys, Mat AA);		//!< changes an existing left vector left += < tr| AA
    PetscErrorCode	MultiplyLeft(Mat AA);				    //!< multiply existing left vector with AA
    
    
  public:
    PModular()	{ isherm = 0; alloc = 0; omega = 0; shift = 0; }			//!< default constructor, isherm = 0
    ~PModular();
    
    PetscErrorCode	Compute(Vec dm,PetscReal time,PetscScalar * ret,PetscInt number);			//compute the expecation value, only first processor gets the answer
    PetscErrorCode	ComputeAll(Vec dm,PetscReal time,PetscScalar * ret,PetscInt number);		//compute the expecation value, all processors get the answer
    
//     virtual PetscErrorCode	Setup(System * sys) = 0;
};

#endif		// _Observable
