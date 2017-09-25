
/**
 * @file	smodular.cpp
 * 
 * 		Definition of the System class methods for the modular Liouvillian setup routines
 * 		This allows to setup a vast number of user defined master equations.
 * 
 * @author	Michael Gegg
 * 
 */

#include"../include/system.hpp"
#include"../include/index.hpp"
#include"../include/dim.hpp"


//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  mls: basic single arrow routines
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

#undef __FUNCT__
#define __FUNCT__ "AddMLSSingleArrowNonconnecting"

/**
 * @brief	Adds a single nonconnecting arrow to the Liouville space operator matrix. Has two modes one for preallocation and one for actual matrix setup. Corresponds to \f$ \dots \f$
 * 
 * @param	AA		the matrix.
 * @param	d_nnz		the array counting the number of local elements belonging to the diagonal block per row.
 * @param	o_nnz		the array counting the number of local elements belonging to the offdiagonal block per row.
 * @param	choose		0 for preallocation and 1 for actual matrix setup.
 * @param	mlselem		the name or identifier of the dimension corresponding to the bubble.
 * @param	couplingconst	the coupling constant of the as arising in the master equation.
 * 
 */

PetscErrorCode	System::AddMLSSingleArrowNonconnecting(Mat AA, PetscInt *d_nnz, PetscInt *o_nnz, PetscInt choose, MLSDim mlselem, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    
    //finding the dimensions
    PetscInt elem=0;
    
    ierr = FindMatch(&mlselem,&elem); CHKERRQ(ierr);
    
    
    //loop part  
    PetscInt	locindex;
    PetscScalar	value;
    PetscInt	d_count = 0, o_count = 0;
    
    locindex	= index->InitializeLocal();
    
    while ( index->ContinueLocal() )					//loop over all local rows
    {
      if(!choose)							//preallocation mode
      {
	d_count++;							//the element itself is always local
	d_nnz[locindex - index->LocStart()]++;
      }
      else								//set values mode
      {
	if( index->MLSQN(elem) )					//trying to avoid overhead
	{
	  value	= matrixelem*((PetscScalar) index->MLSQN(elem));
	  ierr	= MatSetValue(AA,locindex,locindex,value,ADD_VALUES); CHKERRQ(ierr);
	}
      }
      
      locindex = index->Increment();
    }
    
    
    //output part
    if(longout || liouout)
    {
      if(!choose)
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddMLSSingleArrowNonconnecting preassembly completed:\n  input %s\n",mlselem.ToString().c_str()); CHKERRQ(ierr);
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  local elements: %d, \t nonlocal elements: %d \t loc_start: %d, \t loc_end: %d\n",d_count,o_count,index->LocStart(),index->LocEnd()); CHKERRQ(ierr);
	PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
      }
      else
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddMLSSingleArrowNonconnecting assembly completed.\n  input %s\n",mlselem.ToString().c_str()); CHKERRQ(ierr);
      }
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddMLSSingleArrowConnecting"

/**
 * @brief	Adds a single connecting arrow to the Liouville space operator matrix. Has two modes one for preallocation and one for actual matrix setup.
 * 
 * @param	AA		the matrix.
 * @param	d_nnz		the array counting the number of local elements belonging to the diagonal block per row.
 * @param	o_nnz		the array counting the number of local elements belonging to the offdiagonal block per row.
 * @param	choose		0 for preallocation and 1 for actual matrix setup.
 * @param	mlsstart	the name or identifier of the dimension corresponding to the start bubble.
 * @param	mlsgoal		the name or identifier of the dimension corresponding to the goal bubble.
 * @param	couplingconst	the coupling constant of the as arising in the master equation.
 * 
 */

PetscErrorCode	System::AddMLSSingleArrowConnecting(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, MLSDim mlsstart, MLSDim mlsgoal, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    
    
    //finding the dimensions
    PetscInt start=0, goal=0;
    
    ierr = FindMatch(&mlsstart,&start); CHKERRQ(ierr);
    ierr = FindMatch(&mlsgoal,&goal); CHKERRQ(ierr);
    
    
    //loop part  
    PetscInt	locindex,column,n00;
    PetscScalar	value;
    PetscInt	d_count = 0, o_count = 0;
    
    locindex	= index->InitializeLocal();
    
    while ( index->ContinueLocal() )					//loop over all local rows
    {
      if ( start != -1 && goal != -1 )					//start is not n00 and goal is not n00
      {
	if ( index->CanDecrement(goal) && index->CanIncrement(start) )
	{
	  column	= locindex + index->MLSCPitch(goal,start);
	  if(!choose)							//preallocation mode
	  {
	    if( index->IsLocal(column)  )				//local element
	    {
	      d_count++;
	      d_nnz[locindex - index->LocStart()]++;
	    }
	    else							//nonlocal element
	    {
	      o_count++;
	      o_nnz[locindex - index->LocStart()]++;
	    }
	  }
	  else								//MatSetValue mode
	  {
	      if( column >= index->TotalDOF() )
	      {
		index->PrintIndex();
		index->PrintIndices();
	      }
	      value	= matrixelem*((PetscScalar) (index->MLSQN(start)+1));
	      ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
      }
      else if( start != -1 && goal == -1 )				//start is not n00 and goal is n00
      {
	n00 = index->MLSQN(-1);
	if ( n00 > 0 && index->CanIncrement(start) )
	{
	  column	= locindex + index->MLSIPitch(start);
	  if(!choose)							//preallocation mode
	  {
	    if( index->IsLocal(column)  )				//local element
	    {
	      d_count++;
	      d_nnz[locindex - index->LocStart()]++;
	    }
	    else							//nonlocal element
	    {
	      o_count++;
	      o_nnz[locindex - index->LocStart()]++;
	    }
	  }
	  else								//MatSetValue mode
	  {
	    value	= matrixelem*((PetscScalar) (index->MLSQN(start)+1));
	    ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
      }
      else if( start == -1 && goal != -1 )				//start is n00
      {
	n00	= index->MLSQN(-1);
	if ( index->CanDecrement(goal) )
	{
	  column	= locindex + index->MLSDPitch(goal);
	  if(!choose)							//preallocation mode
	  {
	    if( index->IsLocal(column)  )				//local element
	    {
	      d_count++;
	      d_nnz[locindex - index->LocStart()]++;
	    }
	    else							//nonlocal element
	    {
	      o_count++;
	      o_nnz[locindex - index->LocStart()]++;
	    }
	  }
	  else								//MatSetValue mode
	  {
	    value	= matrixelem*((PetscScalar) (n00+1));
	    ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
      }
      else								//down is n00 and up is n00: ERROR
      {
	(*PetscErrorPrintf)("Error: AddMLSSingleArrowConnecting assembly messed up!\n");
	(*PetscErrorPrintf)("Seems like both mls input strings are n00!\n");
	SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
      }
      
      locindex = index->Increment();
    }
    
    
    //output part
    if(longout || liouout)
    {
      if(!choose)
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddMLSSingleArrowConnecting preassembly completed:\n  input %s %s",mlsstart.ToString().c_str(),mlsgoal.ToString().c_str()); CHKERRQ(ierr);
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  local elements: %d, \t nonlocal elements: %d \t loc_start: %d, \t loc_end: %d\n",d_count,o_count,index->LocStart(),index->LocEnd()); CHKERRQ(ierr);
	PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
      }
      else
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddMLSSingleArrowConnecting assembly completed.\n  input %s %s",mlsstart.ToString().c_str(),mlsgoal.ToString().c_str()); CHKERRQ(ierr);
      }
    }
    
    
    PetscFunctionReturn(0);
}



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  mode: Lindbladian routines
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

#undef __FUNCT__
#define __FUNCT__ "AddModeLeftB"

/**
 * @brief	Adds the matrix entries corresponding to \f$ b \rho \equiv (I\otimes b) | \rho \rangle \f$ to the Lindbladian matrix AA. Has two modes one for preallocation and one for actual matrix setup.
 * 
 * @param	AA		the matrix.
 * @param	d_nnz		the array counting the number of local elements belonging to the diagonal block per row.
 * @param	o_nnz		the array counting the number of local elements belonging to the offdiagonal block per row.
 * @param	choose		0 for preallocation and 1 for actual matrix setup.
 * @param	modenumber	the number of the mode in the order it was set by the user, starting with zero
 * @param	matrixelem	Type PetscScalar; possible parameter arising form the master equation, if only \f$ (I\otimes b) \f$ is desired then set this to 1.0
 * 
 */

PetscErrorCode	System::AddModeLeftB(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt modenumber, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    
    //finding the dimensions
    PetscInt	mode;
    ModeDim	modedim (0,modenumber);
    
    ierr = FindMatch(&modedim,&mode); CHKERRQ(ierr);
    
    
    //loop part  
    PetscInt	locindex,column;
    PetscScalar	value;
    PetscInt	d_count = 0, o_count = 0;
    
    locindex	= index->InitializeLocal();
    
    while ( index->ContinueLocal() )					//loop over all local rows
    {
	if ( index->CanIncrement(mode+1) )				//is that right?
	{
	  column = locindex + index->ModeIPitch(mode+1);
	  if(!choose)							//preallocation mode
	  {
	    if( index->IsLocal(column)  )				//local element
	    {
	      d_count++;
	      d_nnz[locindex - index->LocStart()]++;
	    }
	    else							//nonlocal element
	    {
	      o_count++;
	      o_nnz[locindex - index->LocStart()]++;
	    }
	  }
	  else								//MatSetValue mode
	  {
	    value	= matrixelem*PetscSqrtReal((PetscReal) (index->ModeQN(mode+1)+1));
	    ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
      
      locindex = index->Increment();
    }
    
    
    //ouput part
    if(longout || liouout)
    {
      if(!choose)
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddModeLeftAnn preassembly completed:\n  input m%d\n",modenumber); CHKERRQ(ierr);
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  local elements: %d, \t nonlocal elements: %d \t loc_start: %d, \t loc_end: %d\n",d_count,o_count,index->LocStart(),index->LocEnd()); CHKERRQ(ierr);
	PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
      }
      else
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddModeLeftAnn assembly completed.\n  input m%d\n",modenumber); CHKERRQ(ierr);
      }
    }
    
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddModeRightB"

/**
 * @brief	Adds the matrix entries corresponding to \f$ \rho b \equiv (b^T \otimes I) | \rho \rangle \f$ to the Lindbladian matrix AA. Has two modes one for preallocation and one for actual matrix setup.
 * 
 * @param	AA		the matrix.
 * @param	d_nnz		the array counting the number of local elements belonging to the diagonal block per row.
 * @param	o_nnz		the array counting the number of local elements belonging to the offdiagonal block per row.
 * @param	choose		0 for preallocation and 1 for actual matrix setup.
 * @param	modenumber	the number of the mode in the order it was set by the user, starting with zero
 * @param	matrixelem	possible parameter arising form the master equation, if only the operator is desired then set this to 1.0
 * 
 */

PetscErrorCode	System::AddModeRightB(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt modenumber, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    
    //finding the dimensions
    PetscInt	mode;
    ModeDim	modedim (0,modenumber);
    
    ierr = FindMatch(&modedim,&mode); CHKERRQ(ierr);
    
    
    //loop part  
    PetscInt	locindex,column;
    PetscScalar	value;
    PetscInt	d_count = 0, o_count = 0;
    
    locindex	= index->InitializeLocal();
    
    while ( index->ContinueLocal() )					//loop over all local rows
    {
	if ( index->CanDecrement(mode) )				//is that right?
	{
	  column = locindex + index->ModeDPitch(mode);
	  if(!choose)							//preallocation mode
	  {
	    if( index->IsLocal(column)  )				//local element
	    {
	      d_count++;
	      d_nnz[locindex - index->LocStart()]++;
	    }
	    else							//nonlocal element
	    {
	      o_count++;
	      o_nnz[locindex - index->LocStart()]++;
	    }
	  }
	  else								//MatSetValue mode
	  {
	    value	= matrixelem*PetscSqrtReal((PetscReal) index->ModeQN(mode));
	    ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
      
      locindex = index->Increment();
    }
    
    
    //ouput part
    if(longout || liouout)
    {
      if(!choose)
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddModeRightAnn preassembly completed:\n  input m%d\n", modenumber); CHKERRQ(ierr);
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  local elements: %d, \t nonlocal elements: %d \t loc_start: %d, \t loc_end: %d\n",d_count,o_count,index->LocStart(),index->LocEnd()); CHKERRQ(ierr);
	PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
      }
      else
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddModeRightAnn assembly completed.\n  input m%d\n", modenumber); CHKERRQ(ierr);
      }
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddModeLeftBd"

/**
 * @brief	Adds the matrix entries corresponding to \f$ b^\dagger \rho \equiv (I\otimes b^\dagger) | \rho \rangle \f$ to the Lindbladian matrix AA. Has two modes one for preallocation and one for actual matrix setup.
 * 
 * @param	AA		the matrix.
 * @param	d_nnz		the array counting the number of local elements belonging to the diagonal block per row.
 * @param	o_nnz		the array counting the number of local elements belonging to the offdiagonal block per row.
 * @param	choose		0 for preallocation and 1 for actual matrix setup.
 * @param	modenumber	the number of the mode in the order it was set by the user, starting with zero
 * @param	matrixelem	possible parameter arising form the master equation, if only the operator is desired then set this to 1.0
 * 
 */

PetscErrorCode	System::AddModeLeftBd(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt modenumber, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    
    //finding the dimensions
    PetscInt	mode;
    ModeDim	modedim (0,modenumber);
    
    ierr = FindMatch(&modedim,&mode); CHKERRQ(ierr);
    
    
    //loop part  
    PetscInt	locindex,column;
    PetscScalar	value;
    PetscInt	d_count = 0, o_count = 0;
    
    locindex	= index->InitializeLocal();
    
    while ( index->ContinueLocal() )					//loop over all local rows
    {
	if ( index->CanDecrement(mode+1) )				//is that right?
	{
	  column = locindex + index->ModeDPitch(mode+1);
	  if(!choose)							//preallocation mode
	  {
	    if( index->IsLocal(column)  )				//local element
	    {
	      d_count++;
	      d_nnz[locindex - index->LocStart()]++;
	    }
	    else							//nonlocal element
	    {
	      o_count++;
	      o_nnz[locindex - index->LocStart()]++;
	    }
	  }
	  else								//MatSetValue mode
	  {
	    value	= matrixelem*PetscSqrtReal((PetscReal) index->ModeQN(mode+1));
	    ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
      
      locindex = index->Increment();
    }
    
    
    //ouput part
    if(longout || liouout)
    {
      if(!choose)
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddModeLeftCre preassembly completed:\n  input m%d\n", modenumber); CHKERRQ(ierr);
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  local elements: %d, \t nonlocal elements: %d \t loc_start: %d, \t loc_end: %d\n",d_count,o_count,index->LocStart(),index->LocEnd()); CHKERRQ(ierr);
	PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
      }
      else
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddModeLeftCre assembly completed.\n  input m%d\n", modenumber); CHKERRQ(ierr);
      }
    }
    
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddModeRightBd"

/**
 * @brief	Adds the matrix entries corresponding to \f$ \rho b^\dagger \equiv (b^* \otimes I) | \rho \rangle \f$ to the Lindbladian matrix AA. Has two modes one for preallocation and one for actual matrix setup.
 * 
 * @param	AA		the matrix.
 * @param	d_nnz		the array counting the number of local elements belonging to the diagonal block per row.
 * @param	o_nnz		the array counting the number of local elements belonging to the offdiagonal block per row.
 * @param	choose		0 for preallocation and 1 for actual matrix setup.
 * @param	modenumber	the number of the mode in the order it was set by the user, starting with zero
 * @param	matrixelem	possible parameter arising form the master equation, if only the operator is desired then set this to 1.0
 * 
 */

PetscErrorCode	System::AddModeRightBd(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt modenumber, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    
    //finding the dimensions
    PetscInt	mode;
    ModeDim	modedim (0,modenumber);
    
    ierr = FindMatch(&modedim,&mode); CHKERRQ(ierr);
    
    
    //loop part  
    PetscInt	locindex,column;
    PetscScalar	value;
    PetscInt	d_count = 0, o_count = 0;
    
    locindex	= index->InitializeLocal();
    
    while ( index->ContinueLocal() )					//loop over all local rows
    {
	if ( index->CanIncrement(mode) )				//is that right?
	{
	  column = locindex + index->ModeIPitch(mode);
	  if(!choose)							//preallocation mode
	  {
	    if( index->IsLocal(column)  )				//local element
	    {
	      d_count++;
	      d_nnz[locindex - index->LocStart()]++;
	    }
	    else							//nonlocal element
	    {
	      o_count++;
	      o_nnz[locindex - index->LocStart()]++;
	    }
	  }
	  else								//MatSetValue mode
	  {
	    value	= matrixelem*PetscSqrtReal((PetscReal) (index->ModeQN(mode)+1));
	    ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
      
      locindex = index->Increment();
    }
    
    
    //ouput part
    if(longout || liouout)
    {
      if(!choose)
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddModeRightCre preassembly completed:\n  input m%d\n", modenumber); CHKERRQ(ierr);
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  local elements: %d, \t nonlocal elements: %d \t loc_start: %d, \t loc_end: %d\n",d_count,o_count,index->LocStart(),index->LocEnd()); CHKERRQ(ierr);
	PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
      }
      else
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddModeRightCre assembly completed.\n  input m%d\n", modenumber); CHKERRQ(ierr);
      }
    }
    
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddModeLeftBdB"

/**
 * @brief	Adds the matrix entries corresponding to \f$ b^\dagger b \rho \equiv (I \otimes b^\dagger b) | \rho \rangle \f$ to the Lindbladian matrix AA. Has two modes one for preallocation and one for actual matrix setup.
 * 
 * @param	AA		the matrix.
 * @param	d_nnz		the array counting the number of local elements belonging to the diagonal block per row.
 * @param	o_nnz		the array counting the number of local elements belonging to the offdiagonal block per row.
 * @param	choose		0 for preallocation and 1 for actual matrix setup.
 * @param	modenumber	the number of the mode in the order it was set by the user, starting with zero
 * @param	matrixelem	possible parameter arising form the master equation, if only the operator is desired then set this to 1.0
 * 
 */

PetscErrorCode	System::AddModeLeftBdB(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt modenumber, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    
    //finding the dimensions
    PetscInt	mode;
    ModeDim	modedim (0,modenumber);
    
    ierr = FindMatch(&modedim,&mode); CHKERRQ(ierr);
    
    
    //loop part  
    PetscInt	locindex;
    PetscScalar	value;
    PetscInt	d_count = 0, o_count = 0;
    
    locindex	= index->InitializeLocal();
    
    while ( index->ContinueLocal() )					//loop over all local rows
    {
      if( index->ModeQN(mode+1) )					//don't set explicicit zeros
      {
	if(!choose)							//preallocation mode
	{
	  d_count++;
	  d_nnz[locindex - index->LocStart()]++;
	}
	else								//MatSetValue mode
	{
	  value	= matrixelem*((PetscScalar) index->ModeQN(mode+1));
	  ierr	= MatSetValue(AA,locindex,locindex,value,ADD_VALUES); CHKERRQ(ierr);
	}
      }
      
      locindex = index->Increment();
    }
    
    
    //ouput part
    if(longout || liouout)
    {
      if(!choose)
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddModeLeftBdB preassembly completed:\n  input m%d\n", modenumber); CHKERRQ(ierr);
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  local elements: %d, \t nonlocal elements: %d \t loc_start: %d, \t loc_end: %d\n",d_count,o_count,index->LocStart(),index->LocEnd()); CHKERRQ(ierr);
	PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
      }
      else
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddModeLeftBdB assembly completed.\n  input m%d\n", modenumber); CHKERRQ(ierr);
      }
    }
    
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddModeRightBdB"

/**
 * @brief	Adds the matrix entries corresponding to \f$ \rho  b^\dagger b \equiv ((b^\dagger b)^T \otimes I) | \rho \rangle \f$ to the Lindbladian matrix AA. Has two modes one for preallocation and one for actual matrix setup.
 * 
 * @param	AA		the matrix.
 * @param	d_nnz		the array counting the number of local elements belonging to the diagonal block per row.
 * @param	o_nnz		the array counting the number of local elements belonging to the offdiagonal block per row.
 * @param	choose		0 for preallocation and 1 for actual matrix setup.
 * @param	modenumber	the number of the mode in the order it was set by the user, starting with zero
 * @param	matrixelem	possible parameter arising form the master equation, if only the operator is desired then set this to 1.0
 * 
 */

PetscErrorCode	System::AddModeRightBdB(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt modenumber, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    
    //finding the dimensions
    PetscInt	mode;
    ModeDim	modedim (0,modenumber);
    
    ierr = FindMatch(&modedim,&mode); CHKERRQ(ierr);
    
    
    //loop part  
    PetscInt	locindex;
    PetscScalar	value;
    PetscInt	d_count = 0, o_count = 0;
    
    locindex	= index->InitializeLocal();
    
    while ( index->ContinueLocal() )					//loop over all local rows
    {
      if( index->ModeQN(mode) )						//don't set explicicit zeros
      {
	if(!choose)							//preallocation mode
	{
	  d_count++;
	  d_nnz[locindex - index->LocStart()]++;
	}
	else								//MatSetValue mode
	{
	  value	= matrixelem*((PetscScalar) index->ModeQN(mode));
	  ierr	= MatSetValue(AA,locindex,locindex,value,ADD_VALUES); CHKERRQ(ierr);
	}
      }
      
      locindex = index->Increment();
    }
    
    
    //ouput part
    if(longout || liouout)
    {
      if(!choose)
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddModeRightBdB preassembly completed:\n  input m%d\n", modenumber); CHKERRQ(ierr);
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  local elements: %d, \t nonlocal elements: %d \t loc_start: %d, \t loc_end: %d\n",d_count,o_count,index->LocStart(),index->LocEnd()); CHKERRQ(ierr);
	PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
      }
      else
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddModeRightBdB assembly completed.\n  input m%d\n", modenumber); CHKERRQ(ierr);
      }
    }
    
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddModeLeftBBb"

/**
 * @brief	Adds the matrix entries corresponding to \f$ b b^\dagger \rho \equiv (I \otimes b b^\dagger) | \rho \rangle \f$ to the Lindbladian matrix AA. Has two modes one for preallocation and one for actual matrix setup.
 * 
 * @param	AA		the matrix.
 * @param	d_nnz		the array counting the number of local elements belonging to the diagonal block per row.
 * @param	o_nnz		the array counting the number of local elements belonging to the offdiagonal block per row.
 * @param	choose		0 for preallocation and 1 for actual matrix setup.
 * @param	modenumber	the number of the mode in the order it was set by the user, starting with zero
 * @param	matrixelem	possible parameter arising form the master equation, if only the operator is desired then set this to 1.0
 * 
 */

PetscErrorCode	System::AddModeLeftBBd(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt modenumber, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    
    //finding the dimensions
    PetscInt	mode;
    ModeDim	modedim (0,modenumber);
    
    ierr = FindMatch(&modedim,&mode); CHKERRQ(ierr);
    
    
    //loop part  
    PetscInt	locindex;
    PetscScalar	value;
    PetscInt	d_count = 0, o_count = 0;
    
    locindex	= index->InitializeLocal();
    
    while ( index->ContinueLocal() )					//loop over all local rows
    {
      if(!choose)							//preallocation mode
      {
	d_count++;
	d_nnz[locindex - index->LocStart()]++;
      }
      else								//MatSetValue mode
      {
	value	= matrixelem*((PetscScalar) (index->ModeQN(mode+1)+1));			//also does this at upper truncation boundary, is that compatible with the hard truncation? Maybe its not that important
	ierr	= MatSetValue(AA,locindex,locindex,value,ADD_VALUES); CHKERRQ(ierr);
      }
      
      locindex = index->Increment();
    }
    
    
    //ouput part
    if(longout || liouout)
    {
      if(!choose)
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddModeLeftBBb preassembly completed:\n  input m%d\n", modenumber); CHKERRQ(ierr);
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  local elements: %d, \t nonlocal elements: %d \t loc_start: %d, \t loc_end: %d\n",d_count,o_count,index->LocStart(),index->LocEnd()); CHKERRQ(ierr);
	PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
      }
      else
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddModeLeftBBb assembly completed.\n  input m%d\n", modenumber); CHKERRQ(ierr);
      }
    }
    
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddModeRightBBb"

/**
 * @brief	Adds the matrix entries corresponding to \f$ \rho b b^\dagger \equiv ((b b^\dagger )^T \otimes I) | \rho \rangle \f$ to the Lindbladian matrix AA. Has two modes one for preallocation and one for actual matrix setup.
 * 
 * @param	AA		the matrix.
 * @param	d_nnz		the array counting the number of local elements belonging to the diagonal block per row.
 * @param	o_nnz		the array counting the number of local elements belonging to the offdiagonal block per row.
 * @param	choose		0 for preallocation and 1 for actual matrix setup.
 * @param	modenumber	the number of the mode in the order it was set by the user, starting with zero
 * @param	matrixelem	possible parameter arising form the master equation, if only the operator is desired then set this to 1.0
 * 
 */

PetscErrorCode	System::AddModeRightBBd(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt modenumber, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    
    //finding the dimensions
    PetscInt	mode;
    ModeDim	modedim (0,modenumber);
    
    ierr = FindMatch(&modedim,&mode); CHKERRQ(ierr);
    
    
    //loop part  
    PetscInt	locindex;
    PetscScalar	value;
    PetscInt	d_count = 0, o_count = 0;
    
    locindex	= index->InitializeLocal();
    
    while ( index->ContinueLocal() )					//loop over all local rows
    {
      if(!choose)							//preallocation mode
      {
	d_count++;
	d_nnz[locindex - index->LocStart()]++;
      }
      else								//MatSetValue mode
      {
	value	= matrixelem*((PetscScalar) (index->ModeQN(mode)+1));			//also does that at upper truncation boundary, is that compatible with hard truncation? maybe not important after all.
	ierr	= MatSetValue(AA,locindex,locindex,value,ADD_VALUES); CHKERRQ(ierr);
      }
      
      locindex = index->Increment();
    }
    
    
    //ouput part
    if(longout || liouout)
    {
      if(!choose)
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddModeRightBBb preassembly completed:\n  input m%d\n", modenumber); CHKERRQ(ierr);
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  local elements: %d, \t nonlocal elements: %d \t loc_start: %d, \t loc_end: %d\n",d_count,o_count,index->LocStart(),index->LocEnd()); CHKERRQ(ierr);
	PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
      }
      else
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddModeRightBBb assembly completed.\n  input m%d\n", modenumber); CHKERRQ(ierr);
      }
    }
    
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddModeLeftBRightBd"

/**
 * @brief	Adds the matrix entries corresponding to \f$ b \rho b^\dagger \equiv (b^* \otimes b) | \rho \rangle \f$ to the Lindbladian matrix AA. Has two modes one for preallocation and one for actual matrix setup.
 * 
 * @param	AA		the matrix.
 * @param	d_nnz		the array counting the number of local elements belonging to the diagonal block per row.
 * @param	o_nnz		the array counting the number of local elements belonging to the offdiagonal block per row.
 * @param	choose		0 for preallocation and 1 for actual matrix setup.
 * @param	modenumber	the number of the mode in the order it was set by the user, starting with zero
 * @param	matrixelem	possible parameter arising form the master equation, if only the operator is desired then set this to 1.0
 * 
 */

PetscErrorCode	System::AddModeLeftBRightBd(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt modenumber, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    
    //finding the dimensions
    PetscInt	mode;
    ModeDim	modedim (0,modenumber);
    
    ierr = FindMatch(&modedim,&mode); CHKERRQ(ierr);
    
    
    //loop part  
    PetscInt	locindex,column;
    PetscScalar	value;
    PetscInt	d_count = 0, o_count = 0;
    
    locindex	= index->InitializeLocal();
    
    while ( index->ContinueLocal() )					//loop over all local rows
    {
	if ( index->CanIncrement(mode) && index->CanIncrement(mode+1) )				//is that right?
	{
	  column = locindex + index->ModeIPitch(mode) + index->ModeIPitch(mode+1);
	  if(!choose)							//preallocation mode
	  {
	    if( index->IsLocal(column)  )				//local element
	    {
	      d_count++;
	      d_nnz[locindex - index->LocStart()]++;
	    }
	    else							//nonlocal element
	    {
	      o_count++;
	      o_nnz[locindex - index->LocStart()]++;
	    }
	  }
	  else								//MatSetValue mode
	  {
	    value	= matrixelem*PetscSqrtReal((PetscReal) ((index->ModeQN(mode)+1)*(index->ModeQN(mode+1)+1)));
	    ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
      
      locindex = index->Increment();
    }
    
    
    //ouput part
    if(longout || liouout)
    {
      if(!choose)
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddModeLeftBRightBd preassembly completed:\n  input m%d\n", modenumber); CHKERRQ(ierr);
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  local elements: %d, \t nonlocal elements: %d \t loc_start: %d, \t loc_end: %d\n",d_count,o_count,index->LocStart(),index->LocEnd()); CHKERRQ(ierr);
	PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
      }
      else
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddModeLeftBRightBd assembly completed.\n  input m%d\n", modenumber); CHKERRQ(ierr);
      }
    }
    
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddModeLeftBdRightB"

/**
 * @brief	Adds the matrix entries corresponding to \f$ b^\dagger \rho b \equiv (b^T \otimes b^\dagger) | \rho \rangle \f$ to the Lindbladian matrix AA. Has two modes one for preallocation and one for actual matrix setup.
 * 
 * @param	AA		the matrix.
 * @param	d_nnz		the array counting the number of local elements belonging to the diagonal block per row.
 * @param	o_nnz		the array counting the number of local elements belonging to the offdiagonal block per row.
 * @param	choose		0 for preallocation and 1 for actual matrix setup.
 * @param	modenumber	the number of the mode in the order it was set by the user, starting with zero
 * @param	matrixelem	possible parameter arising form the master equation, if only the operator is desired then set this to 1.0
 * 
 */

PetscErrorCode	System::AddModeLeftBdRightB(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt modenumber, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    
    //finding the dimensions
    PetscInt	mode;
    ModeDim	modedim (0,modenumber);
    
    ierr = FindMatch(&modedim,&mode); CHKERRQ(ierr);
    
    
    //loop part  
    PetscInt	locindex,column;
    PetscScalar	value;
    PetscInt	d_count = 0, o_count = 0;
    
    locindex	= index->InitializeLocal();
    
    while ( index->ContinueLocal() )					//loop over all local rows
    {
	if ( index->CanDecrement(mode) && index->CanDecrement(mode+1) )				//is that right?
	{
	  column = locindex + index->ModeDPitch(mode) + index->ModeDPitch(mode+1);
	  if(!choose)							//preallocation mode
	  {
	    if( index->IsLocal(column)  )				//local element
	    {
	      d_count++;
	      d_nnz[locindex - index->LocStart()]++;
	    }
	    else							//nonlocal element
	    {
	      o_count++;
	      o_nnz[locindex - index->LocStart()]++;
	    }
	  }
	  else								//MatSetValue mode
	  {
	    value	= matrixelem*PetscSqrtReal((PetscReal) (index->ModeQN(mode)*index->ModeQN(mode+1)));
	    ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
      
      locindex = index->Increment();
    }
    
    
    //ouput part
    if(longout || liouout)
    {
      if(!choose)
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddModeLeftBdRightB preassembly completed:\n  input m%d\n", modenumber); CHKERRQ(ierr);
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  local elements: %d, \t nonlocal elements: %d \t loc_start: %d, \t loc_end: %d\n",d_count,o_count,index->LocStart(),index->LocEnd()); CHKERRQ(ierr);
	PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
      }
      else
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddModeLeftBdRightB assembly completed.\n  input m%d\n", modenumber); CHKERRQ(ierr);
      }
    }
    
    
    PetscFunctionReturn(0);
}



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  mode: Mat routines
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

#undef __FUNCT__
#define __FUNCT__ "MatModeLeftB"

/**
 * @brief	Creates a matrix corresponding to \f$ b \rho  \equiv (I\otimes b) | \rho \rangle \f$
 * 
 * @param	AA		the matrix.
 * @param	modenumber	the number of the mode in the order it was set by the user, starting with zero
 * @param	matrixelem	possible parameter arising form the master equation, if only the operator is desired then set this to 1.0
 * 
 */

PetscErrorCode System::MatModeLeftB(Mat *AA, PetscInt modenumber, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    
    PetscInt		*d_nnz	= new PetscInt [loc_size] ();
    PetscInt		*o_nnz	= new PetscInt [loc_size] ();
    
    //create matrix
    ierr = PQSPCreateMat(AA); CHKERRQ(ierr);
    
    
    //preassembly
    ierr = AddModeLeftB(*AA,d_nnz,o_nnz,0,modenumber,1.0); CHKERRQ(ierr);
    
    ierr = MatMPIAIJSetPreallocation(*AA,0,d_nnz,0,o_nnz); CHKERRQ(ierr);			//parallel
    ierr = MatSeqAIJSetPreallocation(*AA,0,d_nnz); CHKERRQ(ierr);
    
    
    //assembly
    ierr = AddModeLeftB(*AA,d_nnz,o_nnz,1,modenumber,matrixelem); CHKERRQ(ierr);
    
    ierr = MatAssemblyBegin(*AA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);				//Assemble the matrix
    ierr = MatAssemblyEnd(*AA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);				//

    
    //clean up
    delete[] d_nnz;
    delete[] o_nnz;
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatModeLeftBd"

/**
 * @brief	Creates a matrix corresponding to \f$ b^\dagger \rho  \equiv (I\otimes b^\dagger) | \rho \rangle \f$
 * 
 * @param	AA		the matrix.
 * @param	modenumber	the number of the mode in the order it was set by the user, starting with zero
 * @param	matrixelem	possible parameter arising form the master equation, if only the operator is desired then set this to 1.0
 * 
 */

PetscErrorCode System::MatModeLeftBd(Mat *AA, PetscInt modenumber, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    
    PetscInt		*d_nnz	= new PetscInt [loc_size] ();
    PetscInt		*o_nnz	= new PetscInt [loc_size] ();
    
    //create matrix
    ierr = PQSPCreateMat(AA); CHKERRQ(ierr);
    
    
    //preassembly
    ierr = AddModeLeftBd(*AA,d_nnz,o_nnz,0,modenumber,1.0); CHKERRQ(ierr);
    
    ierr = MatMPIAIJSetPreallocation(*AA,0,d_nnz,0,o_nnz); CHKERRQ(ierr);			//parallel
    ierr = MatSeqAIJSetPreallocation(*AA,0,d_nnz); CHKERRQ(ierr);
    
    
    //assembly
    ierr = AddModeLeftBd(*AA,d_nnz,o_nnz,1,modenumber,matrixelem); CHKERRQ(ierr);
    
    ierr = MatAssemblyBegin(*AA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);				//Assemble the matrix
    ierr = MatAssemblyEnd(*AA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);				//

    
    //clean up
    delete[] d_nnz;
    delete[] o_nnz;
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatModeLeftBdB"

/**
 * @brief	Creates a matrix corresponding to \f$ b^\dagger b \rho  \equiv (I\otimes b^\dagger b) | \rho \rangle \f$
 * 
 * @param	AA		the matrix.
 * @param	modenumber	the number of the mode in the order it was set by the user, starting with zero
 * @param	matrixelem	possible parameter arising form the master equation, if only the operator is desired then set this to 1.0
 * 
 */

PetscErrorCode System::MatModeLeftBdB(Mat *AA, PetscInt modenumber, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    
    PetscInt		*d_nnz	= new PetscInt [loc_size] ();
    PetscInt		*o_nnz	= new PetscInt [loc_size] ();
    
    //create matrix
    ierr = PQSPCreateMat(AA); CHKERRQ(ierr);
    
    
    //preassembly
    ierr = AddModeLeftBdB(*AA,d_nnz,o_nnz,0,modenumber,1.0); CHKERRQ(ierr);
    
    ierr = MatMPIAIJSetPreallocation(*AA,0,d_nnz,0,o_nnz); CHKERRQ(ierr);			//parallel
    ierr = MatSeqAIJSetPreallocation(*AA,0,d_nnz); CHKERRQ(ierr);
    
    
    //assembly
    ierr = AddModeLeftBdB(*AA,d_nnz,o_nnz,1,modenumber,matrixelem); CHKERRQ(ierr);
    
    ierr = MatAssemblyBegin(*AA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);				//Assemble the matrix
    ierr = MatAssemblyEnd(*AA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);				//

    
    //clean up
    delete[] d_nnz;
    delete[] o_nnz;
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatModeLeftBBd"

/**
 * @brief	Creates a matrix corresponding to \f$ b b^\dagger \rho  \equiv (I\otimes b b^\dagger) | \rho \rangle \f$
 * 
 * @param	AA		the matrix.
 * @param	modenumber	the number of the mode in the order it was set by the user, starting with zero
 * @param	matrixelem	possible parameter arising form the master equation, if only the operator is desired then set this to 1.0
 * 
 */

PetscErrorCode System::MatModeLeftBBd(Mat *AA, PetscInt modenumber, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    
    PetscInt		*d_nnz	= new PetscInt [loc_size] ();
    PetscInt		*o_nnz	= new PetscInt [loc_size] ();
    
    //create matrix
    ierr = PQSPCreateMat(AA); CHKERRQ(ierr);
    
    
    //preassembly
    ierr = AddModeLeftBBd(*AA,d_nnz,o_nnz,0,modenumber,1.0); CHKERRQ(ierr);
    
    ierr = MatMPIAIJSetPreallocation(*AA,0,d_nnz,0,o_nnz); CHKERRQ(ierr);			//parallel
    ierr = MatSeqAIJSetPreallocation(*AA,0,d_nnz); CHKERRQ(ierr);
    
    
    //assembly
    ierr = AddModeLeftBBd(*AA,d_nnz,o_nnz,1,modenumber,matrixelem); CHKERRQ(ierr);
    
    ierr = MatAssemblyBegin(*AA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);				//Assemble the matrix
    ierr = MatAssemblyEnd(*AA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);				//

    
    //clean up
    delete[] d_nnz;
    delete[] o_nnz;
    
    PetscFunctionReturn(0);
}