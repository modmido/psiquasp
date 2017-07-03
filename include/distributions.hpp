
/**
 * @file	distributions.hpp
 * 
 * 		Definition of the Distribution and DModular classes.
 * 
 * @author	Michael Gegg
 * 
 */


#ifndef _Distribution
#define _Distribution

#include"output.hpp"


/**
 * @brief	Enables computation of distributions. Provides functionality for basic distributions in the mls density matrix elements and for distributions in the bosonic number states.
 * 
 */

class Distribution: public PropBase
{
  protected:
    PetscInt		length;				//!< The number of different number states for the distribution
    PetscInt		*lengths;			//!< The number of relevant local density matrix elements per number state
    PetscInt		**dmindex;			//!< The indices of the relevant elements, for each number state, starting with zero at the first local element.
    
    PetscErrorCode	AllocateLocStorage(const PetscInt Lenght, const PetscInt *Lengths);	//could be extended
    
  public:    
  //-------------------------------------------------------------------------------------
  //computation and setup of observables
  //-------------------------------------------------------------------------------------
    Distribution()	{ isherm = 1; length = 0; alloc = 0; }					//!< default constructor, isherm = 1, projectors are generally hermitian
    ~Distribution();										//nonempty destructor
    
    PetscInt		PrintTotalNum() { return length; }					//!< Return the number of different number states.
    
    PetscErrorCode	Compute (Vec dm,PetscReal time,PetscScalar * ret,PetscInt number);	//compute the expecation value
    PetscErrorCode	ComputeAll (Vec dm,PetscReal time,PetscScalar * ret,PetscInt number);	//compute the expecation value
    
    PetscErrorCode	SetupMLSDensityDistribution(System * sys,MLSDim mlsdens);
    PetscErrorCode	SetupMLSOffdiagDistribution(System * sys,MLSDim mlsop,PetscInt number);
    PetscErrorCode	SetupModeDistribution(System * sys,PetscInt modenumber);
};


/**
 * @brief	DModular class, enables the definition of more advanced, specialized, user dependent distributions, such as e.g. Dicke state distributions.
 * 
 */

class DModular: public PropBase
{
  protected:
    PetscInt		totalnum;			//!< The total number of different states
    PetscInt		*lengths;			//!< The number of relevant local density matrix elements per state
    PetscInt		**dmindex;			//!< The indices of the relevant elements, for each state, starting with zero at the first local element.
    PetscReal		**prefactors;			//!< The corresponding prefactors
    
    
    PetscErrorCode	AllocateLocStorage(const PetscInt numspin);		//allocate storage
    PetscErrorCode	JBlockShift(System * sys, PetscInt step, Vec elem);	//create root of the next pseudospin subspace
    
  public:    
  //-------------------------------------------------------------------------------------
  //computation and setup of observables
  //-------------------------------------------------------------------------------------
    DModular()	{ isherm = 1; alloc = 0; totalnum = 0; }				//!< default constructor, isherm = 1, projectors are generally hermitian
    ~DModular();									//nonempty destructor
    
    PetscInt		PrintTotalNum() { return totalnum; }					//!< Return the number of different states.
    
    PetscErrorCode	Compute (Vec dm,PetscReal time,PetscScalar * ret,PetscInt number);	//compute the expecation value
    PetscErrorCode	ComputeAll (Vec dm,PetscReal time,PetscScalar * ret,PetscInt number);	//compute the expecation value
    PetscErrorCode	SetupDickeDistribution(System * sys);
};


#endif		// _Distribution