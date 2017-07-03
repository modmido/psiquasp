
/**
 * @file	distributions.cpp
 * 
 * 		Member functions of the Distribution and DModular classes.
 * 
 * @author	Michael Gegg
 * 
 */


#include"../include/distributions.hpp"
#include"../include/index.hpp"
#include"../include/dim.hpp"


//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  distribution class: constructors/destructors
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

/**
 * @brief	Default destructor.
 * 
 */

Distribution::~Distribution()
{
    if(alloc)						//if it has been allocated it needs to be freed
    {
      delete[]		lengths;
      
      PetscInt		i;
      for(i=0; i < length; i++)
      {
	delete[]	dmindex[i];
      }
      delete[]		dmindex;
    }
}


#undef __FUNCT__
#define __FUNCT__ "AllocateLocStorage"

/**
 * @brief	This function initializes the Distribution object for a bosonic distribution.
 * 
 * @param	Lenght		number of different arrays, i.e. number of different number state projectors.
 * @param	Lengths		the lenghts of the respective arrays.
 * 
 */

PetscErrorCode Distribution::AllocateLocStorage(const PetscInt Lenght, const PetscInt* Lengths)
{
    PetscFunctionBeginUser;
    
    PetscInt	i;
    
    length		= Lenght;
    
    PetscInt *loclen	= new PetscInt [length];
    for(i=0; i < length; i++)
    {
      loclen[i]		= Lengths[i];
    }
    lengths		= loclen;
    
    PetscInt **locindex	= new PetscInt* [length];
    for(i=0; i < length; i++)
    {
      locindex[i]	= new PetscInt [Lengths[i]];
    }
    dmindex		= locindex;
    
    alloc++;						//tells us that it has been allocated
	
    PetscFunctionReturn(0);
}



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  distribution class: compute
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

#undef __FUNCT__
#define __FUNCT__ "Compute"

/**
 * @brief	This function computes a single entry of the distribution function. Usually called by the monitor function. Only the first processor gets the global result.
 * 
 * @param	dm		the density matrix.
 * @param	time		the current integration time. Not needed so far.
 * @param	ret		the global return value, only first processor gets it tough...
 * @param	number		the number of the number state whose occupation is to be computed.
 * 
 */

PetscErrorCode Distribution::Compute (Vec dm,PetscReal time,PetscScalar * ret,PetscInt number)
{
    PetscFunctionBeginUser;
  
    PetscErrorCode	ierr;
    PetscInt		n;
    const PetscScalar	*a;
    PetscScalar		local=0;
    
    ierr = VecGetArrayRead(dm,&a);CHKERRQ(ierr);
    
    
    for(n=0; n < lengths[number]; n++)
    {
      local+=a[dmindex[number][n]];
    }
    
    ierr = VecRestoreArrayRead(dm,&a);CHKERRQ(ierr);
    
    MPI_Reduce(&local,ret,1,MPIU_SCALAR,MPIU_SUM,0,PETSC_COMM_WORLD);	//add all the local subsums together
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ComputeAll"

/**
 * @brief	This function computes a single entry of the distribution function. All processors get the global result.
 * 
 * @param	dm		the density matrix.
 * @param	time		the current integration time. Not needed so far.
 * @param	ret		the global return value, only first processor gets it tough...
 * @param	number		the number of the number state whose occupation is to be computed.
 * 
 */

PetscErrorCode Distribution::ComputeAll (Vec dm,PetscReal time,PetscScalar * ret,PetscInt number)
{
    PetscFunctionBeginUser;
  
    PetscErrorCode	ierr;
    PetscInt		n;
    const PetscScalar	*a;
    PetscScalar		local=0;
    
    ierr = VecGetArrayRead(dm,&a);CHKERRQ(ierr);
    
    
    for(n=0; n < lengths[number]; n++)
    {
      local+=a[dmindex[number][n]];
    }
    
    ierr = VecRestoreArrayRead(dm,&a);CHKERRQ(ierr);
    
    MPI_Allreduce(&local,ret,1,MPIU_SCALAR,MPIU_SUM,PETSC_COMM_WORLD);	//add all the local subsums together
    
    PetscFunctionReturn(0);
}



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  distribution class: setup functions
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

#undef __FUNCT__
#define __FUNCT__ "SetupModeDistribution"

/**
 * @brief	This function initializes the Distribution object for a bosonic distribution
 * 
 * @param	sys		the pointer to the system specification object. Needed for things like length of the dimension etc.
 * @param	modenumber	the name of the mode (i.e. m1, m2, etc) of the mode under consideration.
 * 
 */

PetscErrorCode Distribution::SetupModeDistribution(System * sys,PetscInt modenumber)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;

    
    //finding the dimensions  
    PetscInt	dim = 0;
    ModeDim	modedim (0,modenumber);
    
    ierr = sys->FindMatch(&modedim,&dim); CHKERRQ(ierr);
    
    
    //basic properites
    real_value_tolerance	= sys->RealValueTolerance();				//hermitian observables need a tolerance for their realvaluedness
    name			= "mode"+ std::to_string(modenumber);
    
    
    //how many local dm entries?
    PetscInt		locindex;
    PetscInt		templengths[sys->index->MaxQN(dim)+1] = {};			//number of local entries per number state    
    locindex		= sys->index->InitializeLocal();
      
    while ( sys->index->ContinueLocal() )						//loop over all local rows
    {
      if( !sys->index->IsPol() )							//if everyting is density like we take it
      {
	templengths[sys->index->ModeQN(dim)]++;
      }
      locindex	= sys->index->Increment();
    }

      
    //allocate local storage for entries
    ierr = AllocateLocStorage(sys->index->MaxQN(dim)+1,templengths); CHKERRQ(ierr);
    
    
    //fill it with values
    PetscInt		loccount[sys->index->MaxQN(dim)+1] = {};			//index for the individual number states
    locindex		= sys->index->InitializeLocal();
      
    while ( sys->index->ContinueLocal() )						//loop over all local rows
    {
      if( !sys->index->IsPol() )							//if everything is density like we take it
      {
	dmindex[sys->index->ModeQN(dim)][loccount[sys->index->ModeQN(dim)]]	= locindex - sys->index->LocStart();	//local array starts with index zero, i.e. is shifted with respect to the global index
	loccount[sys->index->ModeQN(dim)]++;
      }
      locindex	= sys->index->Increment();
    }

    
    //ouput part
    if(sys->LongOut() || sys->PropOut())
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nDistribution for mode %s initialized.\n",modedim.ToString().c_str()); CHKERRQ(ierr);
    }
 
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SetupMLSDensityDistribution"

/**
 * @brief	This function initializes the Distribution object for a mls density degree of freedom.
 * 
 * @param	sys		the pointer to the system specification object. Needed for things like length of the dimension etc.
 * @param	mlsdens		the name of the mls denstiy (i.e. n00, n11, etc. )
 * 
 */

PetscErrorCode Distribution::SetupMLSDensityDistribution(System * sys,MLSDim mlsdens)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;

    
    //finding the dimensions  
    PetscInt	dim=0;
    
    ierr = sys->FindMatch(&mlsdens,&dim); CHKERRQ(ierr);
    
    
    //basic properites
    real_value_tolerance	= sys->RealValueTolerance();				//hermitian observables need a tolerance for their realvaluedness
    name			= "MLS"+ mlsdens.ToString();
    
    
    //how many local dm entries?
    PetscInt		locindex;
    PetscInt		templengths[sys->index->MaxQN(dim)+1] = {};			//number of local entries per number state, no truncation so far!!!!!!!
    locindex		= sys->index->InitializeLocal();
    
    while ( sys->index->ContinueLocal() )						//loop over all local rows
    {
      if( !sys->index->IsPol() )							//if everyting is density like we take it
      {
	templengths[sys->index->MLSQN(dim)]++;
      }
      locindex	= sys->index->Increment();
    }

      
    //allocate local storage for entries
    ierr = AllocateLocStorage(sys->index->MaxQN(dim)+1,templengths); CHKERRQ(ierr);
    
    
    //set it up
    PetscInt		loccount[sys->index->MaxQN(dim)+1] = {};				//index for the individual number states, no truncation so far!!!
    locindex		= sys->index->InitializeLocal();
     
    while ( sys->index->ContinueLocal() )							//loop over all local rows
    {
      if( !sys->index->IsPol() )								//if everyting is density like we take it
      {
	dmindex[sys->index->MLSQN(dim)][loccount[sys->index->MLSQN(dim)]]	= locindex - sys->index->LocStart();	//local array starts with index zero, i.e. is shifted with respect to the global index
	loccount[sys->index->MLSQN(dim)]++;
      }
      locindex	= sys->index->Increment();
    }

    
    //ouput part
    if(sys->LongOut() || sys->PropOut())
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nDistribution for mls density %s initialized.\n",mlsdens.ToString().c_str()); CHKERRQ(ierr); 
    }
    
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SetupMLSOffdiagDistribution"

/**
 * @brief	This function initializes the Distribution for the offdiagonal mls contributions, e.g. for two-level systems the P[n,x,x] distribution with x \neq 0
 * 
 * @param	sys		the pointer to the system specification object. Needed for things like length of the dimension etc.
 * @param	mlsop		the name of the operator belonging to the transition
 * @param	number		the offdiagonal index -- the x in the above example.
 * 
 */

PetscErrorCode Distribution::SetupMLSOffdiagDistribution(System * sys,MLSDim mlspol1_name,PetscInt number)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;

    
    //finding the dimension
    PetscInt		mlsdens=0, mlspol1=0, mlspol2=0;
    MLSDim		mlspol2_name = mlspol1_name.Swap(mlspol1_name);	//swap constructor
    MLSDim		mlsdens_name (1,mlspol1_name);			//density constructor
    
    ierr = sys->FindMatch(&mlspol1_name,&mlspol1); CHKERRQ(ierr);
    ierr = sys->FindMatch(&mlspol2_name,&mlspol2); CHKERRQ(ierr);
    ierr = sys->FindMatch(&mlsdens_name,&mlsdens); CHKERRQ(ierr);
      
    
    //basic properites
    real_value_tolerance	= sys->RealValueTolerance();				//hermitian observables need a tolerance for their realvaluedness
    name			= "MLS"+ mlsdens_name.ToString() + "_offdiag" + std::to_string(number);
    
    
    //how many local dm entries?
    PetscInt		locindex,max = MIN(sys->index->MaxQN(mlsdens)+1,sys->index->NMls()+1-2*number);
      
    if( sys->index->MaxQN(mlspol1) < number )	max = 0;
      
    if( max <= 0 )								//dimensions are too small for this distribution, so we fill in dummy values so that the program does not crash
    {
      PetscInt	templengths[1] = {};
      ierr = AllocateLocStorage(1,templengths); CHKERRQ(ierr);
    }
    else
    {
      PetscInt		templengths[max] = {};					//number of local entries per number state, smells like segfault :-(
      locindex		= sys->index->InitializeLocal();
    
      while ( sys->index->ContinueLocal() )										//loop over all local rows
      {
	if( sys->index->IsMLSTwoDimNumberOffdiag(mlspol1,mlspol2,number) && sys->index->IsModeDensity() )		//if everyting is density like we take it
	{
	  templengths[sys->index->MLSQN(mlsdens)]++;
	}
	locindex	= sys->index->Increment();
      }
      
      
      //allocate local storage for entries
      ierr = AllocateLocStorage(max,templengths); CHKERRQ(ierr);
      
      
      //set it up
      PetscInt		loccount[max] = {};					//index for the individual number states, no truncation so far!!!  
      locindex		= sys->index->InitializeLocal();
    
      while ( sys->index->ContinueLocal() )								//loop over all local rows
      {
	if( sys->index->IsMLSTwoDimNumberOffdiag(mlspol1,mlspol2,number) && sys->index->IsModeDensity() )	//if everyting is density like we take it
	{
	  dmindex[sys->index->MLSQN(mlsdens)][loccount[sys->index->MLSQN(mlsdens)]]	= locindex - sys->index->LocStart();	//local array starts with index zero, i.e. is shifted with respect to the global index
	  loccount[sys->index->MLSQN(mlsdens)]++;
	}
	locindex	= sys->index->Increment();
      }
    }
      
      
    //ouput part
    if(sys->LongOut() || sys->PropOut())
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nDistribution for offdiagonal mls entries %s initialized.\n",mlspol1_name.ToString().c_str()); CHKERRQ(ierr); 
    }

    
    PetscFunctionReturn(0);
}






//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  dmodular class: constructors/destructors
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

/**
 * @brief	Default destructor.
 * 
 */

DModular::~DModular()
{
    if(alloc)						//if it has been allocated it needs to be freed
    {
      //delete[]		numinv;
      delete[]		lengths;
     
      PetscInt	i;
      for(i=0; i < totalnum; i++)
      {
	delete[]	dmindex[i];
	delete[]	prefactors[i];
      }
      delete[]		dmindex;
      delete[]		prefactors;
    }
}


#undef __FUNCT__
#define __FUNCT__ "AllocateLocStorage"

/**
 * @brief	This function initializes the DModular object for a bosonic distribution. This function does not allocate the whole storage, just the first level storage for the number of different states.
 * 		The storage of the individual states must be allocated elsewhere, e.g. using the System::VecContractYYY() functions.
 * 
 * @param	length		the number of different states in the user defined distribution.
 * 
 */

PetscErrorCode DModular::AllocateLocStorage(PetscInt length)
{
    PetscFunctionBeginUser;
    
    PetscInt  *loclenghts	= new PetscInt   [length];		//the array containing the number of relevant matrix elements for each spin and each inversion quantum number
    PetscInt  **locindex	= new PetscInt*  [length];		//the array containing indices of the relevant dm elements per diagonal Dicke element
    PetscReal **locpref		= new PetscReal* [length];		//the array containing the corresponding prefactors
    
    lengths			= loclenghts;				//set it to the actual members
    dmindex			= locindex;				//...
    prefactors			= locpref;				//...
    totalnum			= length;
    
    alloc = 1;								//alloc = TRUE
	
    PetscFunctionReturn(0);
}



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  dmodular class: compute
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

#undef __FUNCT__
#define __FUNCT__ "Compute"

/**
 * @brief	This function computes a single entry of the dicke distribution. Usually called by the monitor function. Only the first processor gets the global result.
 * 
 * @param	dm		the density matrix.
 * @param	time		the current integration time. Not needed so far.
 * @param	ret		the global return value, only first processor gets it tough...
 * @param	number		the number/index of the state whose occupation has to be computed.
 * 
 */

PetscErrorCode DModular::Compute (Vec dm,PetscReal time,PetscScalar * ret,PetscInt number)
{
    PetscFunctionBeginUser;
  
    PetscErrorCode	ierr;
    PetscInt		n;
    const PetscScalar	*a;
    PetscScalar		local=0;
    
    ierr = VecGetArrayRead(dm,&a);CHKERRQ(ierr);
    
    for(n=0; n < lengths[number]; n++)
    {
      local += prefactors[number][n]*a[dmindex[number][n]];
    }
    
    ierr = VecRestoreArrayRead(dm,&a);CHKERRQ(ierr);
    
    MPI_Reduce(&local,ret,1,MPIU_SCALAR,MPIU_SUM,0,PETSC_COMM_WORLD);	//add all the local subsums together
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ComputeAll"

/**
 * @brief	This function computes a single entry of the dicke distribution. All processors get the global result.
 * 
 * @param	dm		the density matrix.
 * @param	time		the current integration time. Not needed so far.
 * @param	ret		the global return value, only first processor gets it tough...
 * @param	number		the number/index of the state whose occupation has to be computed.
 * 
 */

PetscErrorCode DModular::ComputeAll (Vec dm,PetscReal time,PetscScalar * ret,PetscInt number)
{
    PetscFunctionBeginUser;
  
    PetscErrorCode	ierr;
    PetscInt		n;
    const PetscScalar	*a;
    PetscScalar		local=0;
    
    ierr = VecGetArrayRead(dm,&a);CHKERRQ(ierr);
    
    for(n=0; n < lengths[number]; n++)
    {
      local += prefactors[number][n]*a[dmindex[number][n]];
    }
    
    ierr = VecRestoreArrayRead(dm,&a);CHKERRQ(ierr);
    
    MPI_Allreduce(&local,ret,1,MPIU_SCALAR,MPIU_SUM,PETSC_COMM_WORLD);	//add all the local subsums together
    
    PetscFunctionReturn(0);
}