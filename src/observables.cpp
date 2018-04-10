
/**
 * @file	observables.cpp
 * 
 * 		Contains all function definitions of the Observable class methods.
 * 
 * @author	Michael Gegg
 * 
 * 
 */

#include"../include/observables.hpp"
#include"../include/dim.hpp"
#include"../include/index.hpp"

//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  Observable class: constructors/destructors
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

/**
 * @brief	default constructor
 */

Observable::~Observable()
{
    if(alloc)
    {
      delete[]		lengths;
      delete[]		domega;
      
      PetscInt		i;
      for(i = 0; i < freqcomponents; i++)
      {
	delete[]	dmindex[i]; 
      }
      delete[]		dmindex;
      
      for(i = 0; i < freqcomponents; i++)
      {
	delete[]	prefactor[i]; 
      }
      delete[]		prefactor;
    }
}


#undef __FUNCT__
#define __FUNCT__ "AllocateLocStorage"

/**
 * @brief	Allocates the storage for the observable class.
 * 
 * @param	Freqco	the number of different frequency components.
 * @param	Lengths	the lengths of the arrays corrsponding to the different frequency components.
 * @param	Domega	the actual frequencys of the different frequency components.
 * 
 */

PetscErrorCode	Observable::AllocateLocStorage(const PetscInt Freqco, const PetscInt *Lengths, const PetscReal *Domega)
{
    PetscFunctionBeginUser;
    
    PetscInt	i;
    
    freqcomponents		= Freqco;
    
    PetscInt *loclengths	= new PetscInt [Freqco];
    for(i=0; i < Freqco; i++)
    {
      loclengths[i]		= Lengths[i];
    }
    lengths			= loclengths;
    
    PetscReal *locdomega	= new PetscReal [Freqco];
    for(i=0; i < Freqco; i++)
    {
      locdomega[i]		= Domega[i];
    }
    domega			= locdomega;
    
    PetscInt **locdmindex	= new PetscInt* [Freqco];
    for(i=0; i < Freqco; i++)
    {
      locdmindex[i]		= new PetscInt [lengths[i]];
    }
    dmindex			= locdmindex;
    
    PetscReal **locprefactor	= new PetscReal* [Freqco];
    for(i=0; i < Freqco; i++)
    {
      locprefactor[i]		= new PetscReal [lengths[i]];
    }
    prefactor			= locprefactor;
    
    alloc++;
    
    PetscFunctionReturn(0);
}



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  Observable class: compute
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

#undef __FUNCT__
#define __FUNCT__ "Compute"

/**
 * @brief	This function computes the value of an observable. Usually called by the monitor function. Only the first processor gets the global result.
 * 
 * @param	u		the density matrix.
 * @param	time		the time of the time integration algorithm
 * @param	global		the global return value, only first processor gets it tough...
 * @param	number		not needed here.
 * 
 */

PetscErrorCode Observable::Compute(Vec u, PetscReal time, PetscScalar * global,PetscInt number)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    PetscInt		n,m;
    const PetscScalar	*a;
    PetscScalar		factor,local=0;
  //-----------------------------------------------------------------
  //take relevant dm entries, multiply with prefactor and add together
  //-----------------------------------------------------------------
    ierr = VecGetArrayRead(u,&a);CHKERRQ(ierr);
    
    for(n=0; n < freqcomponents; n++)
    {
      factor=PetscExpScalar(PETSC_i*domega[n]*time);
      
      for(m=0; m < lengths[n]; m++)
      {
	local+=prefactor[n][m]*a[dmindex[n][m]]*factor;
      }
    }
    
    ierr = VecRestoreArrayRead(u,&a);CHKERRQ(ierr);
    
    MPI_Reduce(&local,global,1,MPIU_SCALAR,MPIU_SUM,0,PETSC_COMM_WORLD);	//add all the local subtraces together
    
    *global-=shift;								//actually only needed for Tr[dm] - 1
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ComputeAll"

/**
 * @brief	This function computes the value of an observable. All processors get the global result.
 * 
 * @param	u		the density matrix.
 * @param	time		the time of the time integration algorithm
 * @param	global		the global return value, only first processor gets it tough...
 * @param	number		not needed here.
 * 
 */

PetscErrorCode Observable::ComputeAll(Vec u, PetscReal time, PetscScalar * global,PetscInt number)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    PetscInt		n,m;
    const PetscScalar	*a;
    PetscScalar		factor,local=0;
    
    
  //-----------------------------------------------------------------
  //take relevant dm entries, multiply with prefactor and add together
  //-----------------------------------------------------------------
    ierr = VecGetArrayRead(u,&a);CHKERRQ(ierr);
    
    for(n=0; n < freqcomponents; n++)
    {
      factor=PetscExpScalar(PETSC_i*domega[n]*time);
      
      for(m=0; m < lengths[n]; m++)
      {
	local+=prefactor[n][m]*a[dmindex[n][m]]*factor;
      }
    }
    
    ierr = VecRestoreArrayRead(u,&a);CHKERRQ(ierr);
    
    MPI_Allreduce(&local,global,1,MPIU_SCALAR,MPIU_SUM,PETSC_COMM_WORLD);	//add all the local subtraces together
    
    *global-=shift;								//actually only needed for Tr[dm] - 1
    
    PetscFunctionReturn(0);
}




//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  Observable class: setup functions: general
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

#undef __FUNCT__
#define __FUNCT__ "SetupTrMinus1"

/**
 * @brief	This function initializes the Observable for tr -1 computation.
 * 
 * @param	sys		also works with derived classes . Needed for local dm boundaries and things like that.
 * 
 */

PetscErrorCode Observable::SetupTrMinus1(System * sys)
{
    PetscFunctionBeginUser;
  
    PetscErrorCode	ierr;
    PetscInt		locindex,length = 0;
    
    
    //basic properties
    isherm			= 1;							//it is an observable that should be real valued.
    shift			= 1.0;							//trace gets shifted by one afterwards
    real_value_tolerance	= sys->RealValueTolerance();				//hermitian observables need a tolerance for their realvaluedness
    name			= "tr(rho)-1";
    
    
    //how many local dm entries?
    locindex	= sys->index->InitializeLocal();
    
    while ( sys->index->ContinueLocal() )						//loop over all local rows
    {
      if( !sys->index->IsPol() )
      {
	length++;
      }
      
      locindex	= sys->index->Increment();
    }
    
    
    //allocate storage
    PetscReal	Domega = 0;
    
    ierr = AllocateLocStorage(1,&length,&Domega); CHKERRQ(ierr);

    
    //fill with values
    locindex	= sys->index->InitializeLocal();
    PetscInt	count	= 0;
    
    while ( sys->index->ContinueLocal() )						//loop over all local rows
    {
      if( !sys->index->IsPol() )
      {
	dmindex[0][count]	= locindex - sys->index->LocStart();			//local array starts with index zero, i.e. is shifted with respect to the global index
	prefactor[0][count]	= 1;
	count++;
      }
      
      locindex	= sys->index->Increment();
    }
    
    
    //ouput part
    if(sys->LongOut() || sys->PropOut())
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTr(rho)-1 Observable initialized.\n"); CHKERRQ(ierr);
    }
    
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SetupTr"

/**
 * @brief	This function initializes the Observable for trace computation.
 * 
 * @param	sys		also works with derived classes . Needed for local dm boundaries and things like that.
 * 
 */

PetscErrorCode Observable::SetupTr(System * sys)
{
    PetscFunctionBeginUser;
  
    PetscErrorCode	ierr;
    PetscInt		locindex,length = 0;
    
    
    //basic properties
    isherm			= 1;							//it is an observable that should be real valued.
    shift			= 0.0;							//trace gets shifted by one afterwards
    real_value_tolerance	= sys->RealValueTolerance();				//hermitian observables need a tolerance for their realvaluedness
    name			= "tr(rho)";
    
    
    //how many local dm entries?
    locindex	= sys->index->InitializeLocal();
    
    while ( sys->index->ContinueLocal() )						//loop over all local rows
    {
      if( !sys->index->IsPol() )
      {
	length++;
      }
      
      locindex	= sys->index->Increment();
    }
    
    
    //allocate storage
    PetscReal	Domega = 0;
    
    ierr = AllocateLocStorage(1,&length,&Domega); CHKERRQ(ierr);

    
    //fill with values
    locindex	= sys->index->InitializeLocal();
    PetscInt	count	= 0;
    
    while ( sys->index->ContinueLocal() )						//loop over all local rows
    {
      if( !sys->index->IsPol() )
      {
	dmindex[0][count]	= locindex - sys->index->LocStart();			//local array starts with index zero, i.e. is shifted with respect to the global index
	prefactor[0][count]	= 1;
	count++;
      }
      
      locindex	= sys->index->Increment();
    }
    
    
    //ouput part
    if(sys->LongOut() || sys->PropOut())
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTrace Observable initialized.\n"); CHKERRQ(ierr);
    }
    
    
    PetscFunctionReturn(0);
}



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  setup functions: mls expectation values
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

#undef __FUNCT__
#define __FUNCT__ "SetupMlsOccupation"

/**
 * @brief	This function initializes the Observable for computation of the <J_{xx}> expectation value.
 * 
 * @param	sys		    also works with derived classes . Needed for local dm boundaries and things like that.
 * @param	Mlsdens		the name of the density whose occupation is to be computed, i.e. n00, n11, n22, or the corresponding operator, i.e. j00, j11, etc.
 * 
 */

PetscErrorCode Observable::SetupMlsOccupation(System * sys, MLSDim * Mlsdens)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    
    //failsafe 
    if( !Mlsdens->IsDensity() )
    {
      (*PetscErrorPrintf)("Invalid input for SetupMlsOccupation():\n");
      (*PetscErrorPrintf)("Invalid input for mls density: current MLSDim is %s\n",Mlsdens->ToString().c_str());
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
    }
    else
    {
      //finding the dimensions  
      PetscInt	dim=0;
      
      ierr = sys->FindMatch(Mlsdens,&dim); CHKERRQ(ierr);
    
      //basic properties
      isherm			    = 1;							            //it is an observable that should be real valued.
      shift			        = 0;
      real_value_tolerance	= sys->RealValueTolerance();				//hermitian observables need a tolerance for their realvaluedness
      name			        = "<J"+std::to_string(dim) + std::to_string(dim)+">";
    
    
      //how many local dm entries?
      PetscInt		locindex, length = 0;
      locindex		= sys->index->InitializeLocal();
    
      while ( sys->index->ContinueLocal() )						//loop over all local rows
      {
          if( !sys->index->IsPol() )
          {
              length++;
          }
          locindex	= sys->index->Increment();
      }
      
      
      //allocate storage
      PetscReal	Domega = 0;
      
      ierr = AllocateLocStorage(1,&length,&Domega); CHKERRQ(ierr);
      

      //fill with values
      locindex	= sys->index->InitializeLocal();
      PetscInt	count	= 0;
	
      while ( sys->index->ContinueLocal() )						        //loop over all local rows
      {
          if( !sys->index->IsPol() )
          {
              dmindex[0][count]	= locindex - sys->index->LocStart();			//local array starts with index zero, i.e. is shifted with respect to the global index
              prefactor[0][count]	= sys->index->MLSQN(dim);
              count++;
          }
          locindex	= sys->index->Increment();
      }
      
      
      //ouput part
      if(sys->LongOut() || sys->PropOut())
      {
          ierr = PetscPrintf(PETSC_COMM_WORLD,"\nObservable for occupation of %s dimension initialized.\n",(Mlsdens->ToString()).c_str()); CHKERRQ(ierr);
      }
    }
    
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SetupMlsPolarization"

/**
 * @brief	This function initializes the Observable struct for computation of the polarization observable. This polarization observable coincides with the <J_{xy}> observable for two-level systems, for multi-level systems it is only the contribution 
 * 		corresponding to the MLSDim specified.
 * 
 * @param	sys		also works with derived classes . Needed for local dm boundaries and things like that.
 * @param	mlsop		the name of the pol operator whose expectation value is to be computed, i.e. j01, j34, j20 etc.
 * @param	freq		the rotating frame frequency for this observable
 * 
 */

PetscErrorCode Observable::SetupMlsPolarization(System* sys, MLSDim * mlsop, PetscReal freq)
{
    PetscFunctionBeginUser;
  
    PetscErrorCode	ierr;
    
    
    //finding the dimensions  
    PetscInt		pol=0;
    
    ierr = sys->FindMatch(mlsop,&pol); CHKERRQ(ierr);
    
  
    //basic properties
    isherm			= 0;
    shift			= 0;
    name			= "<"+mlsop->ToString().replace(0,1,"J")+">";
    
    
    //how many local dm entries?
    PetscInt		locindex, length = 0;
    locindex	= sys->index->InitializeLocal();					//start value for the combined index
    
    while ( sys->index->ContinueLocal() )						//loop over all local rows
    {
      if( sys->index->IsMLSOneDimFirstOffdiag(pol) && sys->index->IsModeDensity() )	//all mls dofs are density like except the pol dimension which has value one && the modes are also all density like
      {
          length++;
      }
     
      locindex	= sys->index->Increment();
    }
    
    
    //Allocate the internal storage
    ierr = AllocateLocStorage(1,&length,&freq); CHKERRQ(ierr);  
      

    //fill with values
    locindex	= sys->index->InitializeLocal();
    PetscInt	count	= 0;
      
    while ( sys->index->ContinueLocal() )						//loop over all local rows
    {
        if( sys->index->IsMLSOneDimFirstOffdiag(pol) && sys->index->IsModeDensity() )	//all mls dofs are density like except the pol dimension which has value one && the modes are also all density like
        {
            dmindex[0][count]	= locindex - sys->index->LocStart();			//local array starts with index zero, i.e. is shifted with respect to the global index
            prefactor[0][count]	= 1;
            count++;
        }
        
        locindex	= sys->index->Increment();
    }

    
    //ouput part
    if(sys->LongOut() || sys->PropOut())
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nObservable for MLS polarization %s initialized.\n", (mlsop->ToString()).c_str()); CHKERRQ(ierr);
    }
    

    PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "SetupMlsHigherPolarization"

/**
 * @brief	This function initializes the Observable for computation of the higher polarization expectation value. This polarization observable coincides with the <J_{xy}^n> observable for two-level systems. For multi-level systems this should not be used.
 * 		Instead the user should construct the J_{xy} matrix and then use the Petsc MatMatMult() tool to generate the J_{xy}^n matrix and then use the PModular functionality to set up the opservable.
 * 
 * @param	sys		also works with derived classes . Needed for local dm boundaries and things like that.
 * @param	mlsop		the name of the pol operator whose expectation value is to be computed, i.e. j01, j34, j20 etc.
 * @param	order		the order of the observable, i.e. the power n of the polarization operator
 * @param	freq		the rotating frame frequency
 * 
 */

PetscErrorCode Observable::SetupMlsHigherPolarization(System* sys, MLSDim * mlsop, PetscInt order, PetscReal freq)
{
    PetscFunctionBeginUser;
  
    PetscErrorCode	ierr;
    
    
    //finding the dimensions  
    PetscInt		pol=0;
    
    ierr = sys->FindMatch(mlsop,&pol); CHKERRQ(ierr);
    

    //basic properties
    isherm		= 0;
    shift		= 0;
    name		= "<("+mlsop->ToString().replace(0,1,"J")+")^"+std::to_string(order)+">";
    
    
    //how many local dm entries?
    PetscInt		locindex, length = 0;
    locindex	= sys->index->InitializeLocal();					//start value for the combined index
    
    while ( sys->index->ContinueLocal() )						//loop over all local rows
    {
      if( sys->index->IsMLSOneDimNumberOffdiag(pol,order) && sys->index->IsModeDensity() )	//all mls dofs are density like except the pol dimension which has value one && the modes are also all density like
      {
          length++;
      }
      
      locindex	= sys->index->Increment();
    }
    
    
    //allocate storage
    PetscReal	Domega = freq*order;
    
    ierr = AllocateLocStorage(1,&length,&Domega); CHKERRQ(ierr);  
      
    
    //fill with values
    PetscInt	count	= 0;
    PetscReal	fac	= sys->Factorial(order);
      
    locindex		= sys->index->InitializeLocal();
      
    while ( sys->index->ContinueLocal() )						//loop over all local rows
    {
        if( sys->index->IsMLSOneDimNumberOffdiag(pol,order) && sys->index->IsModeDensity() )	//all mls dofs are density like except the pol dimension which has value one && the modes are also all density like
        {
            dmindex[0][count]	= locindex - sys->index->LocStart();			//local array starts with index zero, i.e. is shifted with respect to the global index
            prefactor[0][count]	= fac;
            count++;
        }
        
        locindex	= sys->index->Increment();
    }
    
    
    //ouput part
    if(sys->LongOut() || sys->PropOut())
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nObservable for order %d MLS polarization %s initialized.\n", order, (mlsop->ToString()).c_str() ); CHKERRQ(ierr);
    }
    

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SetupMLSOccupationFull"

/**
 * @brief	This function initializes the Observable for computation of the < J_{xy} J_{yx} > expectation value. Only takes into account the two levels specified by the MLSDim. For multi-level systems this should not be used.
 * 
 * @param	sys		        also works with derived classes . Needed for local dm boundaries and things like that.
 * @param	mlspol1_name	the name of the second (rightmost) polarization operator in the expression, i.e. J_{yx} in this case.
 * 
 */

PetscErrorCode	Observable::SetupMLSOccupationFull(System * sys, MLSDim * mlspol1_name)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    
    //finding the dimensions  
    PetscInt		mlsdens=0, mlspol1=0, mlspol2=0;
    MLSDim		    mlspol2_name = mlspol1_name->Swap(*mlspol1_name);		//swap constructor
    MLSDim		    mlsdens_name (1,*mlspol1_name);				            //density constructor
    
    ierr = sys->FindMatch(mlspol1_name,&mlspol1); CHKERRQ(ierr);
    ierr = sys->FindMatch(&mlspol2_name,&mlspol2); CHKERRQ(ierr);
    ierr = sys->FindMatch(&mlsdens_name,&mlsdens); CHKERRQ(ierr);
    
    
    //basic properties
    isherm			= 1;							//it is an observable that should be real valued.
    shift			= 0;
    real_value_tolerance	= sys->RealValueTolerance();				//hermitian observables need a tolerance for their realvaluedness
    name			= "<"+mlspol2_name.ToString().replace(0,1,"J") + mlspol1_name->ToString().replace(0,1,"J")+">";
      
    
    //how many local dm entries?
    PetscInt		locindex, length = 0;								//multiindex for the different dimensions, combined index, column
    
    locindex = sys->index->InitializeLocal();								//start value for the combined index
    
    while ( sys->index->ContinueLocal() )								//loop over all local rows
    {
      //		 P[.. n_yy  .. n_yx = 0 .. n_xy = 0 .. n_xx ..; .. m,m .. ] density
      if( !sys->index->IsPol() )									//is it a density?
      {
          length++;
      }

      //		 P[.. n_yy-1  .. n_yx = 1 .. n_xy = 1 .. n_xx-1 ..; .. m,m .. ] polarization
      if( sys->index->IsMLSTwoDimFirstOffdiag(mlspol1,mlspol2) && sys->index->IsModeDensity() )	//are the two pol dims equal to one and the rest density like && are all modes density like?
      {
          length++;
      }
      
      locindex	= sys->index->Increment();
    }
    
    
    //allocate storage
    PetscReal		Domega = 0.0;
      
    ierr = AllocateLocStorage(1,&length,&Domega); CHKERRQ(ierr);

    
    //fill with values
    locindex = sys->index->InitializeLocal();									//start value for the combined index

    PetscInt	count		= 0;
    while ( sys->index->ContinueLocal() )									//loop over all local rows
    {
      //		 P[.. n_yy  .. n_yx = 0 .. n_xy = 0 .. n_xx ..; .. m,m .. ] density
      if( !sys->index->IsPol() )
      {
          dmindex[0][count]	= locindex - sys->index->LocStart();						//local array starts with index zero, i.e. is shifted with respect to the global index
          prefactor[0][count]	= sys->index->MLSQN(mlsdens);							//n_xx
          count++;
      }
      
      //		 P[.. n_yy-1  .. n_yx = 1 .. n_xy = 1 .. n_xx-1 ..; .. m,m .. ] polarization
      if( sys->index->IsMLSTwoDimFirstOffdiag(mlspol1,mlspol2) && sys->index->IsModeDensity() )			//are the two pol dims equal to one and the rest density like && are all modes density like?
      {
          dmindex[0][count]	= locindex - sys->index->LocStart();						//local array starts with index zero, i.e. is shifted with respect to the global index
          prefactor[0][count]	= 1; 										//
          count++;
      }
	
      locindex	= sys->index->Increment();
    }
    
    
    //ouput part
    if(sys->LongOut() || sys->PropOut())
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSetupMLSOccupationFull initialized: mlspol1 = %s mlspol2 = %s mlsdens = %s\n", mlspol1_name->ToString().c_str(), mlspol2_name.ToString().c_str(), mlsdens_name.ToString().c_str()); CHKERRQ(ierr);
    }
    
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SetupMLSIntercoupling"

/**
 * @brief	This function initializes the Observable for computation of the < J_{xy} J_{yx} - J_{xx}> expectation value. Only meaningful for two-level systems. For multi-level systems this should be constructed from the elementary matrices and the PModular functionality.
 * 
 * @param	sys		            also works with derived classes . Needed for local dm boundaries and things like that.
 * @param	mlspol1_name		the name of the second (lowering) polarization operator in the expression, i.e. J_{yx} in this case.
 * 
 */

PetscErrorCode	Observable::SetupMLSIntercoupling(System * sys, MLSDim * mlspol1_name)
{
    PetscFunctionBeginUser;
  
    PetscErrorCode	ierr;
    
    
    //finding the dimensions  
    PetscInt		mlspol1=0, mlspol2=0;
    MLSDim		    mlspol2_name  = mlspol1_name->Swap(*mlspol1_name);			//swap constructor
    
    ierr = sys->FindMatch(mlspol1_name,&mlspol1); CHKERRQ(ierr);
    ierr = sys->FindMatch(&mlspol2_name,&mlspol2); CHKERRQ(ierr);
    
    
    //basic properties
    isherm			= 1;							//it is an observable that should be real valued.
    shift			= 0;
    real_value_tolerance	= sys->RealValueTolerance();				//hermitian observables need a tolerance for their realvaluedness
    name			= "<"+mlspol2_name.ToString().replace(0,1,"s")+ "_i " + mlspol1_name->ToString().replace(0,1,"s")+"_j"+">";
    
    
    //how many local dm entries?
    PetscInt		locindex, length = 0;								//multiindex for the different dimensions, combined index, column
    
    locindex = sys->index->InitializeLocal();								//start value for the combined index
    
    while ( sys->index->ContinueLocal() )								//loop over all local rows
    {      
      //		 P[.. n_yy-1  .. n_yx = 1 .. n_xy = 1 .. n_xx-1 ..; .. m,m .. ] polarization
      if( sys->index->IsMLSTwoDimFirstOffdiag(mlspol1,mlspol2) && sys->index->IsModeDensity() )	//are the two pol dims equal to one and the rest density like && are all modes density like?
      {
          length++;
      }
      
      locindex	= sys->index->Increment();
    }
    
    
    //allocate storage
    PetscReal		Domega = 0.0;
      
    ierr = AllocateLocStorage(1,&length,&Domega); CHKERRQ(ierr);

    
    //fill with values
    locindex = sys->index->InitializeLocal();									//start value for the combined index

    PetscInt	count		= 0;
    while ( sys->index->ContinueLocal() )									//loop over all local rows
    {
      //		 P[.. n_yy-1  .. n_yx = 1 .. n_xy = 1 .. n_xx-1 ..; .. m,m .. ] polarization
      if( sys->index->IsMLSTwoDimFirstOffdiag(mlspol1,mlspol2) && sys->index->IsModeDensity() )			//are the two pol dims equal to one and the rest density like && are all modes density like?
      {
          dmindex[0][count]	= locindex - sys->index->LocStart();						//local array starts with index zero, i.e. is shifted with respect to the global index
          prefactor[0][count]	= 1; 										//
          count++;
      }
	
      locindex	= sys->index->Increment();
    }
    
    
    //ouput part
    if(sys->LongOut() || sys->PropOut())
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSetupMLSIntercoupling initialized: mlspol1 = %s \n", mlspol1_name->ToString().c_str() ); CHKERRQ(ierr);
    }
    
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SetupMlsJzDiff"

/**
 * @brief	This function initializes the Observable for computation of the < (J_{xx} - J_{yy})^2 - 1 > expectation value.
 * 
 * @param	sys		also works with derived classes . Needed for local dm boundaries and things like that.
 * @param	mlsdens1_name	the name of the first density n_{xx}
 * @param	mlsdens2_name	the name of the second density n_{yy}
 * 
 */

PetscErrorCode Observable::SetupMlsJzDiff(System * sys,MLSDim * mlsdens1_name, MLSDim * mlsdens2_name)
{
    PetscFunctionBeginUser;
  
    PetscErrorCode	ierr;
    
    
    //finding the dimensions  
    PetscInt		dens1=0, dens2=0;
    
    ierr = sys->FindMatch(mlsdens1_name,&dens1); CHKERRQ(ierr);
    ierr = sys->FindMatch(mlsdens2_name,&dens2); CHKERRQ(ierr);

    
    //basic properties
    isherm			= 1;							//it is an observable that should be real valued.
    shift			= 1.0;
    real_value_tolerance	= sys->RealValueTolerance();				//hermitian observables need a tolerance for their realvaluedness
    name			= "<("+mlsdens1_name->ToString().replace(0,1,"J") +"-"+ mlsdens2_name->ToString().replace(0,1,"J")+")^2-1>";
    
    
    //how many local dm entries?
    PetscInt	locindex, length = 0;
    locindex	= sys->index->InitializeLocal();
      
    while ( sys->index->ContinueLocal() )						//loop over all local rows
    {
      if( !sys->index->IsPol() )
      {
	length++;
      }
      locindex	= sys->index->Increment();
    }
    
    
    //allocate storage
    PetscReal	Domega = 0;
    
    ierr = AllocateLocStorage(1,&length,&Domega); CHKERRQ(ierr);
    

    //fill with values
    locindex	= sys->index->InitializeLocal();
    PetscInt	count	= 0;
      
    while ( sys->index->ContinueLocal() )						//loop over all local rows
    {
      if( !sys->index->IsPol() )
      {
	dmindex[0][count]	= locindex - sys->index->LocStart();			//local array starts with index zero, i.e. is shifted with respect to the global index
	prefactor[0][count]	= (sys->index->MLSQN(dens1)-sys->index->MLSQN(dens2))*(sys->index->MLSQN(dens1)-sys->index->MLSQN(dens2));
	count++;
      }
      locindex	= sys->index->Increment();
    }
    
    
    //ouput part
    if(sys->LongOut() || sys->PropOut())
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSetupMlsJzDiff initialized: input: %s %s \n",mlsdens1_name->ToString().c_str(),mlsdens2_name->ToString().c_str()); CHKERRQ(ierr);
    }
    
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SetupMlsJzSquaredNorm"

/**
 * @brief	This function initializes the Observable for computation of the < (J_{xy}^z)^2 > expectation value, using the 1/2 norm convention, i.e. J_{xy}^z = 1/2 (J_{xx}-J_{yy}). 
 * 		    Needed e.g. for the spin squeezing inequality things...
 * 
 * @param	sys		        also works with derived classes . Needed for local dm boundaries and things like that.
 * @param	mlsdens1_name	the name of the "upper" density, i.e. n_xx
 * @param	mlsdens2_name	the name of the "lower" density, i.e. n_yy
 * 
 */

PetscErrorCode Observable::SetupMlsJzSquaredNorm(System * sys, MLSDim * mlsdens1_name, MLSDim * mlsdens2_name)
{
    PetscFunctionBeginUser;
  
    PetscErrorCode	ierr;
    

    //finding the dimensions  
    PetscInt		dens1=0, dens2=0;
    
    ierr = sys->FindMatch(mlsdens1_name,&dens1); CHKERRQ(ierr);
    ierr = sys->FindMatch(mlsdens2_name,&dens2); CHKERRQ(ierr);

       
    //basic properties
    isherm			= 1;							//it is an observable that should be real valued.
    shift			= 0.0;
    real_value_tolerance	= sys->RealValueTolerance();				//hermitian observables need a tolerance for their realvaluedness
    name			= "<("+mlsdens1_name->ToString().replace(0,1,"J") +"/2-"+ mlsdens2_name->ToString().replace(0,1,"J")+"/2)^2>";
    
    
    //how many local dm entries?
    PetscInt		locindex, length = 0;
    locindex		= sys->index->InitializeLocal();
      
    while ( sys->index->ContinueLocal() )						//loop over all local rows
    {
      if( !sys->index->IsPol() )
      {
	length++;
      }
      locindex	= sys->index->Increment();
    }
    
    
    //allocate storage
    PetscReal	Domega = 0;
    
    ierr = AllocateLocStorage(1,&length,&Domega); CHKERRQ(ierr);
    

    //fill with values
    locindex	= sys->index->InitializeLocal();
    PetscInt	count	= 0;
      
    while ( sys->index->ContinueLocal() )						//loop over all local rows
    {
      if( !sys->index->IsPol() )
      {
	dmindex[0][count]	= locindex - sys->index->LocStart();			//local array starts with index zero, i.e. is shifted with respect to the global index
	prefactor[0][count]	= ((sys->index->MLSQN(dens1)-sys->index->MLSQN(dens2))*(sys->index->MLSQN(dens1)-sys->index->MLSQN(dens2)))/4.0;
	count++;
      }
      locindex	= sys->index->Increment();
    }
    
    
    //ouput part
    if(sys->LongOut() || sys->PropOut())
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSetupMlsJzSquaredNorm initialized: input: %s %s \n",mlsdens1_name->ToString().c_str(),mlsdens2_name->ToString().c_str()); CHKERRQ(ierr);
    }
    
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SetupMlsJzNorm"

/**
 * @brief	This function initializes the Observable for computation of the < J_{xy}^z > expectation value, using the 1/2 norm convention, i.e. J_{xy}^z = 1/2 (J_{xx}-J_{yy}). 
 * 		Needed e.g. for the spin squeezing inequality things...
 * 
 * @param	sys		also works with derived classes . Needed for local dm boundaries and things like that.
 * @param	mlsdens1_name	the name of the upper density, i.e. n_xx
 * @param	mlsdens2_name	the name of the lower density, i.e. n_yy
 * 
 */

PetscErrorCode Observable::SetupMlsJzNorm(System * sys, MLSDim * mlsdens1_name, MLSDim * mlsdens2_name)
{
    PetscFunctionBeginUser;
  
    PetscErrorCode	ierr;
    
    
    //finding the dimensions  
    PetscInt		dens1=0, dens2=0;
    
    ierr = sys->FindMatch(mlsdens1_name,&dens1); CHKERRQ(ierr);
    ierr = sys->FindMatch(mlsdens2_name,&dens2); CHKERRQ(ierr);
       

    //basic properties
    isherm			= 1;							//it is an observable that should be real valued.
    shift			= 0.0;
    real_value_tolerance	= sys->RealValueTolerance();				//hermitian observables need a tolerance for their realvaluedness
    name			= "<("+mlsdens1_name->ToString().replace(0,1,"J") +"/2-"+ mlsdens2_name->ToString().replace(0,1,"J")+"/2)>";
    
    
    //how many local dm entries?
    PetscInt		locindex, length = 0;
    locindex		= sys->index->InitializeLocal();
    
    while ( sys->index->ContinueLocal() )						//loop over all local rows
    {
      if( !sys->index->IsPol() )
      {
	length++;
      }
      locindex	= sys->index->Increment();
    }
    
    
    //allocate storage
    PetscReal	Domega = 0;
    
    ierr = AllocateLocStorage(1,&length,&Domega); CHKERRQ(ierr);
    

    //fill with values
    locindex	= sys->index->InitializeLocal();
    PetscInt	count	= 0;
      
    while ( sys->index->ContinueLocal() )						//loop over all local rows
    {
      if( !sys->index->IsPol() )
      {
	dmindex[0][count]	= locindex - sys->index->LocStart();			//local array starts with index zero, i.e. is shifted with respect to the global index
	prefactor[0][count]	= (sys->index->MLSQN(dens1)-sys->index->MLSQN(dens2))/2.0;
	count++;
      }
      locindex	= sys->index->Increment();
    }
    
    
    //ouput part
    if(sys->LongOut() || sys->PropOut())
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSetupMlsJzNorm initialized: input: %s %s \n",mlsdens1_name->ToString().c_str(),mlsdens2_name->ToString().c_str()); CHKERRQ(ierr);
    }
    
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SetupTotalSpin"

/**
 * @brief	This function initializes the Observable for computation of the < J^2 > total spin expectation value. Is a two-level system thing, and is defined with respect to two levels, a upper and a lower level. Two-level system routine.
 * 
 * @param	sys		also works with derived classes . Needed for local dm boundaries and things like that.
 * @param	mlsdens1_name	the name of the upper density
 * @param	mlsdens2_name	the name of the lower density
 * 
 */

PetscErrorCode Observable::SetupTotalSpin(System * sys, MLSDim * mlsdens1_name, MLSDim * mlsdens2_name)
{
    PetscFunctionBeginUser;
  
    PetscErrorCode	ierr;
    
    
    //finding the dimensions  
    PetscInt		dens1=0, dens2=0, pol1 = 0, pol2 = 0;
    MLSDim		    mlspol1_name (*mlsdens2_name,*mlsdens1_name);
    MLSDim		    mlspol2_name (*mlsdens1_name,*mlsdens2_name);
    
    ierr = sys->FindMatch(mlsdens1_name,&dens1); CHKERRQ(ierr);
    ierr = sys->FindMatch(mlsdens2_name,&dens2); CHKERRQ(ierr);
    ierr = sys->FindMatch(&mlspol1_name,&pol1); CHKERRQ(ierr);
    ierr = sys->FindMatch(&mlspol2_name,&pol2); CHKERRQ(ierr);
    

    //basic properties
    isherm			= 1;							//it is an observable that should be real valued.
    shift			= 0.0;
    real_value_tolerance	= sys->RealValueTolerance();				//hermitian observables need a tolerance for their realvaluedness
    name			= "<"+mlspol1_name.ToString().replace(0,1,"J^2")+">";
    
    
    //how many local dm entries?
    PetscInt		locindex, length = 0;    
    locindex		= sys->index->InitializeLocal();
    
    while ( sys->index->ContinueLocal() )						//loop over all local rows
    {
      //		 P[.. n_yy  .. n_yx = 0 .. n_xy = 0 .. n_xx ..; .. m,m .. ] density
      if( !sys->index->IsPol() )									//is it a density?
      {
          length++;
      }
  
      //		 P[.. n_yy-1  .. n_yx = 1 .. n_xy = 1 .. n_xx-1 ..; .. m,m .. ] polarization
      if( sys->index->IsMLSTwoDimFirstOffdiag(pol1,pol2) && sys->index->IsModeDensity() )			//are the two pol dims equal to one and the rest density like && are all modes density like?
      {
          length++;
      }
      locindex	= sys->index->Increment();
    }
    
    
    //allocate storage
    PetscReal	Domega = 0;
    
    ierr = AllocateLocStorage(1,&length,&Domega); CHKERRQ(ierr);
    

    //fill with values
    locindex	= sys->index->InitializeLocal();
    PetscInt	count	= 0;
      
    while ( sys->index->ContinueLocal() )						//loop over all local rows
    {
      //		 P[.. n_yy  .. n_yx = 0 .. n_xy = 0 .. n_xx ..; .. m,m .. ] density
      if( !sys->index->IsPol() )
      {
          dmindex[0][count]	= locindex - sys->index->LocStart();			//local array starts with index zero, i.e. is shifted with respect to the global index
          prefactor[0][count]	= (sys->index->MLSQN(dens1)-sys->index->MLSQN(dens2))*(sys->index->MLSQN(dens1)-sys->index->MLSQN(dens2))/4.0 + (sys->index->MLSQN(dens1)+sys->index->MLSQN(dens2))/2.0;
          count++;
      }
      
      //		 P[.. n_yy-1  .. n_yx = 1 .. n_xy = 1 .. n_xx-1 ..; .. m,m .. ] polarization
      if( sys->index->IsMLSTwoDimFirstOffdiag(pol1,pol2) && sys->index->IsModeDensity() )			//are the two pol dims equal to one and the rest density like && are all modes density like?
      {
          dmindex[0][count]	= locindex - sys->index->LocStart();						//local array starts with index zero, i.e. is shifted with respect to the global index
          prefactor[0][count]	= 1; 										//
          count++;
      }
      locindex	= sys->index->Increment();
    }
    
    
    //ouput part
    if(sys->LongOut() || sys->PropOut())
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSetupTotalSpin initialized: input: %s %s \n",mlsdens1_name->ToString().c_str(),mlsdens2_name->ToString().c_str()); CHKERRQ(ierr);
    }
    
    
    PetscFunctionReturn(0);
}



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  Observable class: setup functions: mode expectation values
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

#undef __FUNCT__
#define __FUNCT__ "SetupModeOccupation"

/**
 * @brief	This function initializes the Observable struct for computation of the <bdb> expectation value for a specific mode.
 * 
 * @param	sys		    also works with derived classes . Needed for local dm boundaries and things like that.
 * @param	modenumber	the number of the mode
 * 
 */

PetscErrorCode Observable::SetupModeOccupation(System * sys,PetscInt modenumber)
{
    PetscFunctionBeginUser;
  
    PetscErrorCode	ierr;
    
    
    //finding the dimensions  
    PetscInt	dim = 0;
    ModeDim	modedim (0,modenumber);
    
    ierr = sys->FindMatch(&modedim,&dim); CHKERRQ(ierr);
       

    //basic properties
    isherm			= 1;							//it is an observable that should be real valued.
    shift			= 0;
    real_value_tolerance	= sys->RealValueTolerance();				//hermitian observables need a tolerance for their realvaluedness
    name			= "<bdb"+ std::to_string(modenumber) + ">";
    
    
    //how many local dm entries?
    PetscInt		locindex, length = 0;
    locindex		= sys->index->InitializeLocal();								//initialize local index
    
    while( sys->index->ContinueLocal() )
    {
      if( !sys->index->IsPol() )									//if everyting is density like we take it
      {
          length++;
      }
      locindex = sys->index->Increment();
    }
    
    
    //allocate storage
    PetscReal	Domega = 0;
    
    ierr = AllocateLocStorage(1,&length,&Domega); CHKERRQ(ierr);
      
    
    //fill with values
    locindex	= sys->index->InitializeLocal();							//initialize local index      
    PetscInt	count	= 0;
      
    while ( sys->index->ContinueLocal() )								//loop over all local rows
    {
      if( !sys->index->IsPol() )									//if everything is density like we take it
      {
          dmindex[0][count]	= locindex - sys->index->LocStart();					//local array starts with index zero, i.e. is shifted with respect to the global index
          prefactor[0][count]	= sys->index->ModeQN(dim);
          count++;
      }
      locindex	= sys->index->Increment();
    }
    
    
    //ouput part
    if(sys->LongOut() || sys->PropOut())
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nObservable for occupation of mode %d initialized.\n",modenumber ); CHKERRQ(ierr);
    }
    

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SetupModePolarization"

/**
 * @brief	This function initializes the Observable struct for computation of the < b > expectation value.
 * 
 * @param	sys		    Needed for local dm boundaries and things like that.
 * @param	modenumber	the number of the mode
 * @param	freq		the rotating frame frequency
 * 
 */

PetscErrorCode Observable::SetupModePolarization(System* sys, PetscInt modenumber, PetscReal freq)
{
    PetscFunctionBeginUser;
  
    PetscErrorCode	ierr;
    
  
    //finding the dimensions  
    PetscInt	dim = 0;
    ModeDim	modedim (0,modenumber);
    
    ierr = sys->FindMatch(&modedim,&dim); CHKERRQ(ierr);
  

    //basic properties
    isherm	= 0;
    shift	= 0;
    name			= "<b"+ std::to_string(modenumber) + ">";

    
    //how many local dm entries?
    PetscInt		locindex, length = 0;    
    locindex		= sys->index->InitializeLocal();
    
    /**
     * this uses tr(b \rho) = tr(\rho b) = sum_m tr(|m>< m|\rho b) = sum_m sqrt(m) tr(|m-1>< m| \rho ) 
     * therefore IsModeOneDimFirstLeftOffdiag, which corresponds to tr(|m-1>< m| \rho ) is correct
     */
    
    while ( sys->index->ContinueLocal() )							//loop over all local rows
    {
      if( sys->index->IsMLSDensity() && sys->index->IsModeOneDimFirstLeftOffdiag(dim) )		//all mls dofs are density like && the dim mode is in the first left offdiag
      {
	length++;
      }
      locindex	= sys->index->Increment();
    }
    
    
    //allocate storage
    ierr = AllocateLocStorage(1,&length,&freq); CHKERRQ(ierr);

    
    //fill with values
    locindex	= sys->index->InitializeLocal();
    PetscInt	count	= 0;
      
    while ( sys->index->ContinueLocal() )							//loop over all local rows
    {
      if( sys->index->IsMLSDensity() && sys->index->IsModeOneDimFirstLeftOffdiag(dim) )		//all mls dofs are density like && the dim mode is in the first left offdiag
      {
	dmindex[0][count]	= locindex - sys->index->LocStart();				//local array starts with index zero, i.e. is shifted with respect to the global index
	prefactor[0][count]	= PetscSqrtReal((PetscReal) (sys->index->ModeQN(dim)+1));	//because we use the shift m -> m-1 then b|m> = sqrt(m)|m-1> becomes b |m+1> = sqrt(m+1) |m>
	count++;
      }
      locindex	= sys->index->Increment();
    }
    
    
    //ouput part
    if(sys->LongOut() || sys->PropOut())
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nObservable for polarization of mode %d initialized.\n",modenumber ); CHKERRQ(ierr);
    }
    
    
    PetscFunctionReturn(0);
}



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  PModular class: constructors/destructors
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

PModular::~PModular()
{
    if(alloc)
    {
      VecDestroy(&left);
    }
}



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  PModular class: compute
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

#undef __FUNCT__
#define __FUNCT__ "Compute"

/**
 * @brief	This function computes the value of a current observable. Usually called by the monitor function. I think all processors get the global result...
 * 
 * @param	right		the density matrix.
 * @param	time		the time of the time integration algorithm
 * @param	ret		    the return value, only first processor gets it tough...
 * @param	number		not needed here.
 * 
 */

PetscErrorCode PModular::Compute(Vec right,PetscReal time,PetscScalar * ret,PetscInt number)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    ierr	= VecDot(left,right,ret); CHKERRQ(ierr);	// (| left >)^d | rho > = < left|rho >
    
    *ret -= shift;
    *ret *= PetscExpScalar(PETSC_i*omega*time);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ComputeAll"

/**
 * @brief	This function computes the value of a current observable. Usually called by the monitor function. I think all processors get the global result...
 * 
 * @param	right		the density matrix.
 * @param	time		the time of the time integration algorithm
 * @param	ret		    the global return value, only first processor gets it tough...
 * @param	number		not needed here.
 * 
 */

PetscErrorCode PModular::ComputeAll(Vec right,PetscReal time,PetscScalar * ret,PetscInt number)
{
    PetscFunctionBeginUser;

    PetscErrorCode	ierr;
    
    ierr	= VecDot(left,right,ret); CHKERRQ(ierr);
    
    *ret -= shift;
    *ret *= PetscExpScalar(PETSC_i*omega*time);
    
    PetscFunctionReturn(0);
}



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  PModular class: setup utilities
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

#undef __FUNCT__
#define __FUNCT__ "GenerateLeft"

/**
 * @brief	This function creates the left vector and initializes it as \f$ AA^\dagger |tr \rangle \f$
 * 
 * @param	sys		also works with derived classes .
 * @param	AA		the operator corresponding to the expecation value
 * 
 */

PetscErrorCode PModular::GenerateLeft(System * sys, Mat AA)
{
    PetscFunctionBeginUser;
  
    PetscErrorCode	ierr;
    Vec			tr;
    
    ierr = sys->PQSPCreateVec(&tr,NULL,NULL); CHKERRQ(ierr);
    ierr = sys->VecTrace(tr); CHKERRQ(ierr);
    
    ierr = sys->PQSPCreateVec(&left,NULL,NULL); CHKERRQ(ierr);
    ierr = MatMultHermitianTranspose(AA,tr,left); CHKERRQ(ierr);
    
    ierr = VecDestroy(&tr); CHKERRQ(ierr);
    
    alloc = 1;
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "LeftOverwrite"

/**
 * @brief	This function overwrites the left vector with \f$ AA^\dagger |tr \rangle \f$
 * 
 * @param	sys		also works with derived classes .
 * @param	AA		the operator corresponding to the expecation value
 * 
 */

PetscErrorCode PModular::LeftOverwrite(System * sys, Mat AA)
{
    PetscFunctionBeginUser;
  
    PetscErrorCode	ierr;
    Vec			tr;
    
    ierr = sys->PQSPCreateVec(&tr,NULL,NULL); CHKERRQ(ierr);
    ierr = sys->VecTrace(tr); CHKERRQ(ierr);
    
    ierr = VecScale(left,0.0); CHKERRQ(ierr);
    ierr = MatMultHermitianTranspose(AA,tr,left); CHKERRQ(ierr);
    
    ierr = VecDestroy(&tr); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "LeftUpdate"

/**
 * @brief	This function updates the left vector with \f$ |left \rangle += AA^\dagger | tr \rangle \f$
 * 
 * @param	sys		also works with derived classes .
 * @param	AA		the operator corresponding to the expecation value
 * 
 */

PetscErrorCode PModular::LeftUpdate(System * sys, Mat AA)
{
    PetscFunctionBeginUser;
  
    PetscErrorCode	ierr;
    Vec			tr,diff;
    
    ierr = sys->PQSPCreateVec(&tr,NULL,NULL); CHKERRQ(ierr);
    ierr = sys->PQSPCreateVec(&diff,NULL,NULL); CHKERRQ(ierr);
    ierr = sys->VecTrace(tr); CHKERRQ(ierr);
    
    ierr = MatMultHermitianTranspose(AA,tr,diff); CHKERRQ(ierr);
    ierr = VecAXPY(left,1.0,diff); CHKERRQ(ierr);
    
    ierr = VecDestroy(&tr); CHKERRQ(ierr);
    ierr = VecDestroy(&diff); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MultiplyLeft"

/**
 * @brief	This function multiplies an existing left vector with a Matrix AA
 * 
 * @param	AA		the matrix
 * 
 */

PetscErrorCode PModular::MultiplyLeft(Mat AA)
{
    PetscFunctionBeginUser;
  
    PetscErrorCode	ierr;
    Vec			dummy;
    
    ierr = VecDuplicate(left,&dummy); CHKERRQ(ierr);
    ierr = VecCopy(left,dummy); CHKERRQ(ierr);
    ierr = MatMultHermitianTranspose(AA,dummy,left); CHKERRQ(ierr);
    ierr = VecDestroy(&dummy); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}
