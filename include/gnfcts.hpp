
/**
 * @file	gnfcts.hpp
 * 
 * 		Gnfct and Elem classes for correlation functions of arbitrary order.
 * 
 * @author	Michael Gegg
 * 
 */


#ifndef _Gnfcts
#define _Gnfcts

#include"output.hpp"
#include"index.hpp"
#include<stack>

/**
 * @brief	Enables computation of MLS and mode correlation functions of arbitrary order.<br>
 * 
 * 		Mode correlation functions are e.g. the second order correlation function \f$ g^{(2)}(0) = \langle b^\dagger b^\dagger b b \rangle / \langle b^\dagger b\rangle^2 \f$
 * 		MLS correlation functions are analog to the mode correlation function: strating from a generic collective lowering operator \f$ J_{xy} \f$ we define e.g. \f$ g^{(2)}_{MLS} (0) = \langle J_{yx} J_{yx} J_{xy} J_{xy} \rangle/ \langle J_{yx} J_{xy}\rangle^2\f$
 * 
 */

class Gnfct: public PropBase
{
  protected:
  //-------------------------------------------------------------------------------------
  //parameters
  //-------------------------------------------------------------------------------------
    PetscInt		*lengths;			//!< The number of relevant local density matrix elements for lenghts[0] the numerator and lengths[1] the denominator
    PetscInt		**numbers;			//!< The indices of the relevant local density matrix elements for numerator and denominator. Starting with zero at the first local element.
    PetscReal		**factors;			//!< The corresponding prefactors.
    PetscInt		order;				//!< The order of the correlation function.
    
    
  //-------------------------------------------------------------------------------------
  //internal functions
  //-------------------------------------------------------------------------------------
    PetscErrorCode	AllocateLocStorage(PetscInt count,PetscInt choose);
    PetscErrorCode	SingleElementMLSNO(System * sys, PetscInt mlsdens, PetscInt leftpol, PetscInt rightpol, PetscInt lowerdens, std::list<Elem*> * result, std::stack<Elem*> * input);
    PetscErrorCode	MLSNormalorderedExpecationvalue(System * sys, PetscInt order, MLSDim destructor, PetscInt choose);
    
    PetscErrorCode	CombineListElems(std::list<Elem*> * clean, std::list<Elem*> * raw);			//combine the raw output of the recursive SingleElementMLSNE
    PetscErrorCode	ComputeIndex(System * sys, std::list<Elem*> * clean);				//compute the dmindex of each element in the list
    
    
  public:    
  //-------------------------------------------------------------------------------------
  //user functions
  //-------------------------------------------------------------------------------------
    Gnfct();												//default constructor
    virtual ~Gnfct();
    
    PetscErrorCode	Compute(Vec dm,PetscReal time,PetscScalar * ret,PetscInt number);		//compute the expecation value
    PetscErrorCode	ComputeAll(Vec dm,PetscReal time,PetscScalar * ret,PetscInt number);		//compute the expecation value
    PetscErrorCode	SetupModeGnfct(System * sys,PetscInt modenumber,PetscInt order);			//setup bosonic gnfct of mode mode and of order order 
    PetscErrorCode	SetupMLSGnfct(System * sys,MLSDim destructor,PetscInt inorder);			//setup excitonic gnfct of mlsdens dof and of order order
    
};


#endif		// _Gnfcts