
/**
 * @file	sliouville.cpp
 * 
 * 		Definition of the System class methods for setting up the Jacobian matrix of the master equation
 * 		and some other Liouville space operators.
 * 
 * @author	Michael Gegg
 * 
 */

#include"../include/system.hpp"
#include"../include/index.hpp"
#include"../include/dim.hpp"



#undef __FUNCT__
#define __FUNCT__ "AddDiagZeros"

/**
 * @brief	Write zeros into the diagonal of the matrix. Has two modes one for preallocation one for actual matrix setup.
 * 
 * @param	AA	the matrix.
 * @param	d_nnz	the array counting the number of local elements belonging to the diagonal block per row.
 * @param	choose	0 for preallocation and 1 for actual matrix setup.
 * 
 */

PetscErrorCode System::AddDiagZeros(Mat AA, PetscInt * d_nnz, PetscInt choose)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    PetscInt		locindex;
    
    
    //loop part
    locindex = index->InitializeLocal();					//initialize index to the local start position
    
    while ( index->ContinueLocal() )						//loop over all local elements
    {
      if(!choose)								//preallocation stage
      {
	d_nnz[locindex - index->LocStart()]++;					//the labeling of the local portion starts with zero...
      }
      else									//allocation stage
      {
	ierr	= MatSetValue(AA,locindex,locindex,0.0,ADD_VALUES); CHKERRQ(ierr);
      }
      
      locindex = index->Increment();						//increment the index
    }
    
    
    //ouput part
    if(longout || liouout)
    {
      if(!choose)
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddDiagZeros preassembly completed\n"); CHKERRQ(ierr);
      }
      else
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddDiagZeros assembly completed\n"); CHKERRQ(ierr);
      }
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddDiagOne"

/**
 * @brief	Write ones into the diagonal of the matrix. Has two modes one for preallocation one for actual matrix setup. Corresponds to the Liouville space identity.
 * 
 * @param	AA	the matrix.
 * @param	d_nnz	the array counting the number of local elements belonging to the diagonal block per row.
 * @param	choose	0 for preallocation and 1 for actual matrix setup.
 * 
 */

PetscErrorCode System::AddDiagOne(Mat AA, PetscInt * d_nnz, PetscInt choose)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    PetscInt		locindex;
    
    
    //loop part
    locindex = index->InitializeLocal();					//initialize index to the local start position
    
    while ( index->ContinueLocal() )						//loop over all local elements
    {
      if(!choose)								//preallocation stage
      {
	d_nnz[locindex - index->LocStart()]++;					//the labeling of the local portion starts with zero...
      }
      else									//allocation stage
      {
	ierr	= MatSetValue(AA,locindex,locindex,1.0,ADD_VALUES); CHKERRQ(ierr);
      }
      
      locindex = index->Increment();						//increment the index
    }
    
    
    //ouput part
    if(longout || liouout)
    {
      if(!choose)
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddDiagZeros preassembly completed\n"); CHKERRQ(ierr);
      }
      else
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddDiagZeros assembly completed\n"); CHKERRQ(ierr);
      }
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddLastRowTrace"

/**
 * @brief	Write the trace into the last line of the matrix. There needs to be an additional last row for this to work properly. This row must have index index->TotalDOF()+1
 * 
 * @param	AA	the matrix.
 * @param	d_nnz	the array counting the number of local elements belonging to the diagonal block per row.
 * * @param	o_nnz	the array counting the number of local elements belonging to the offdiagonal block per row.
 * @param	choose	0 for preallocation and 1 for actual matrix setup.
 * 
 */

PetscErrorCode System::AddLastRowTrace(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    PetscInt		locindex,d_count=0,o_count=0;
    PetscMPIInt		rank,size;
    
    
    //loop part
    MPI_Comm_size(PETSC_COMM_WORLD,&size);
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    
    if( rank == size-1 )							//last row should (hopefully) be on the last processor
    {
      locindex = index->InitializeGlobal();					//initialize index to the local start position
    
      while ( index->ContinueGlobal() )						//loop over all local elements
      {
	if( !index->IsPol() )
	{
	  if(!choose)								//preallocation stage
	  {
	    if( index->IsLocal(locindex) )
	    {
	      d_nnz[index->TotalDOF() - index->LocStart()]++;			//the labeling of the local portion starts with zero...
	      d_count++;
	    }
	    else
	    {
	      o_nnz[index->TotalDOF() - index->LocStart()]++;
	      o_count++;
	    }
	  }
	  else									//allocation stage
	  {
	    ierr	= MatSetValue(AA,index->TotalDOF(),locindex,1.0,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
      
	locindex = index->Increment();						//increment the index
      }
    }
    
    
    //ouput part
    if(longout || liouout)
    {
      if(!choose)
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nWriteLastRowTrace preassembly completed:\n");
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  local elements: %d, \t nonlocal elements: %d \t loc_start: %d, \t loc_end: %d\n",d_count,o_count,index->LocStart(),index->LocEnd()); CHKERRQ(ierr);
	PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
      }
      else
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nWriteLastRowTrace assembly completed.\n"); CHKERRQ(ierr);
      }
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddModeH0"

/**
 * @brief	Adds the H0 contribution of a bosonic mode in the quantum master equation to the matrix. The function sets/adds the Liouvillian matrix \f$ \mathcal{L} \f$ corresponding to \f$ \mathcal{L}\rho =  + i \omega [\rho, b^\dagger b] \f$.
 * 
 * @param	AA		the matrix.
 * @param	d_nnz		the array counting the number of local elements belonging to the diagonal block per row.
 * @param	o_nnz		the array counting the number of local elements belonging to the offdiagonal block per row.
 * @param	choose		0 for preallocation and 1 for actual matrix setup.
 * @param	Name		the name or identifier of the mode dimension, i.e. mX NOT dmX. Treats both mode degrees of freedom automatically!
 * @param	couplingconst	the coupling constant as arising in the master equation, i.e. the \f$ i \omega \f$
 * 
 */

PetscErrorCode	System::AddModeH0(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt modenumber, PetscScalar couplingconst)
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
	value	= couplingconst*((PetscScalar) (index->ModeQN(mode)-index->ModeQN(mode+1)));
	ierr	= MatSetValue(AA,locindex,locindex,value,ADD_VALUES); CHKERRQ(ierr);
      }
      
      locindex = index->Increment();
    }
    
    
    //output part
    if(longout || liouout)
    {
      if(!choose)
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nModeH0 preassembly completed:\n  input m%d\n",modenumber); CHKERRQ(ierr);
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  local elements: %d, \t nonlocal elements: %d \t loc_start: %d, \t loc_end: %d\n",d_count,o_count,index->LocStart(),index->LocEnd()); CHKERRQ(ierr);
	PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
      }
      else
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nModeH0 assembly completed.\n  input m%d\n",modenumber); CHKERRQ(ierr);
      }
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddMLSH0"

/**
 * @brief	Adds a single MLS H0 contribution to the matrix. The function sets/adds parts of the Liouvillian matrix \f$ \mathcal{L} \f$ corresponding to \f$ \mathcal{L}\rho =  + i \omega [\rho,J_{xx}] \f$.
 * 
 * 		pol refers to the polarization MLSDim which is relevant for the snippet of the H0 contribution: if e.g. H0 = hbar omega J_11, then pol may be any polarization starting with 1 i.e. n10, n12, n13 etc.
 * 		the matrix elements then are given by couplingconst*(n1i - ni1) or generally for J_{xx} -> couplingconst*(nxy - nyx).
 * 		If the user wants to set an entire MLS H0 contribution then the user has to do the following: the input MLSDim pol has the ket value x if the Hamiltonian \f$ \hbar \omega J_{xx} \f$ is desired. Then the user has to call
 * 		this functions once for all possible bra values. This depends on the number of levels and on the symmetries in the master equation, e.g. if there are symmetries in the master equation that allow to omit certain polarization degrees of freedom
 * 		this might affect the number of function calls for this function.
 * 
 * @param	AA		the matrix.
 * @param	d_nnz		the array counting the number of local elements belonging to the diagonal block per row.
 * @param	o_nnz		the array counting the number of local elements belonging to the offdiagonal block per row.
 * @param	choose		0 for preallocation and 1 for actual matrix setup.
 * @param	pol		the name or identifier of the dimension corresponding to the bubble.
 * @param	couplingconst	the coupling constant for the interaction including the i/hbar prefactor of the von-Neumann equation. Cannot be time dependent!
 * 
 */

PetscErrorCode	System::AddMLSH0(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, MLSDim polname, PetscScalar couplingconst)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    
    //finding the dimensions  
    PetscInt	pol1=0, pol2=0;
    MLSDim	pol2name = polname.Swap(polname);
    
    ierr = FindMatch(&polname,&pol1); CHKERRQ(ierr);
    ierr = FindMatch(&pol2name,&pol2); CHKERRQ(ierr);
       

    //loop part  
    PetscInt	locindex;
    PetscScalar	value;
    PetscInt	d_count = 0;
    
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
	value	= couplingconst*((PetscScalar) (index->MLSQN(pol1)-index->MLSQN(pol2)));
	ierr	= MatSetValue(AA,locindex,locindex,value,ADD_VALUES); CHKERRQ(ierr);
      }
      
      locindex = index->Increment();
    }
    
    
    //output part
    if(longout || liouout)
    {
      if(!choose)
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddMLSH0 preassembly completed:\n  input: %s\n",polname.ToString().c_str());
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  local elements: %d, \t nonlocal elements: %d \t loc_start: %d, \t loc_end: %d\n",d_count,0,index->LocStart(),index->LocEnd()); CHKERRQ(ierr);
	PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
      }
      else
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddMLSH0 assembly completed.\n  input: %s\n",polname.ToString().c_str()); CHKERRQ(ierr);
      }
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddMLSModeInt"

/**
 * @brief	Adds a single dipole-dipole like MLS-Mode interaction Hamiltonian arrow of the MLS sketches to the matrix. In the two-level system case this corresponds to the Dicke Hamiltonian.
 * 
 * 		mlsdown refers to the "lower" bubble and mlsup to the "higher" bubble in the sketches (assuming ascending energy ordering), photon refers to the mode degree of freedom, either ket or bra.
 * 		if the order of the bubbles is reversed, then the contribution corresponds to the non-rwa terms of the dipole-dipole coupling Hamiltonian
 * 
 * @param	AA		the matrix.
 * @param	d_nnz		the array counting the number of local elements belonging to the diagonal block per row.
 * @param	o_nnz		the array counting the number of local elements belonging to the offdiagonal block per row.
 * @param	choose		0 for preallocation and 1 for actual matrix setup.
 * @param	mlsdown		the name or identifier of the dimension corresponding to the lower bubble.
 * @param	mlsup		the name or identifier of the dimension corresponding to the upper bubble.
 * @param	photon		the name or identifier of the dimension corresponding to mode degree of freedom.
 * @param	couplingconst	the coupling constant for the interaction including the i/hbar prefactor of the von-Neumann equation.
 * 
 */

PetscErrorCode System::AddMLSModeInt(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, MLSDim mlsdown, MLSDim mlsup, ModeDim photon, PetscScalar couplingconst)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    
    //finding the dimensions  
    PetscInt down=0, up=0, modedim=0;
  
    ierr = FindMatch(&mlsdown,&down); CHKERRQ(ierr);
    ierr = FindMatch(&mlsup,&up); CHKERRQ(ierr);
    ierr = FindMatch(&photon,&modedim); CHKERRQ(ierr);
    
 
    //loop part  
    PetscInt	locindex,column,mvalue,n00;
    PetscScalar	value;
    PetscInt	d_count = 0, o_count = 0;
    
    locindex	= index->InitializeLocal();
    
    while ( index->ContinueLocal() )								//loop over all local rows
    {
      mvalue	= index->ModeQN(modedim);							//the actual current number state of the considered mode dimension

      if ( down != -1 && up != -1 )								//down is not n00 and up is not n00
      {
	if ( index->CanDecrement(down) && index->CanIncrement(up) && index->CanDecrement(modedim) )	//element fulfills requirements
	{
	  column	= locindex + index->MLSCPitch(down,up) + index->ModeDPitch(modedim);
	  if(!choose)										//preallocation mode
	  {
	    if( index->IsLocal(column)  )							//local element
	    {
	      d_count++;
	      d_nnz[locindex - index->LocStart()]++;
	    }
	    else										//nonlocal element
	    {
	      o_count++;
	      o_nnz[locindex - index->LocStart()]++;
	    }
	  }
	  else											//MatSetValue mode
	  {
	    if( column >= index->TotalDOF() )
	    {
	      ierr	= index->PrintIndices(); CHKERRQ(ierr);
	      ierr	= PetscPrintf(PETSC_COMM_WORLD,"column: %s = %d, %s = %d, %s = %d\t\t",(mlsdown.ToString()).c_str(),index->Indices(down)-1,(mlsup.ToString()).c_str(),index->Indices(up)+1,(photon.ToString()).c_str(),index->Indices(modedim)-1);
	      ierr	= PetscPrintf(PETSC_COMM_WORLD,"locindex = %d mls pitch = %d mode pitch = %d\n",locindex,index->MLSCPitch(down,up),index->ModeDPitch(modedim));
	    }
	    value	= couplingconst*PetscSqrtReal((PetscReal) mvalue)*((PetscScalar) (index->MLSQN(up)+1));
	    ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
	  
	if ( index->CanDecrement(up) && index->CanIncrement(down) && index->CanIncrement(modedim) )	//element fulfills requirements
	{
	  column	= locindex + index->MLSCPitch(up,down) + index->ModeIPitch(modedim);
	  if(!choose)										//preallocation mode
	  {
	    if( index->IsLocal(column)  )							//local element
	    {
	      d_count++;
	      d_nnz[locindex - index->LocStart()]++;
	    }
	    else										//nonlocal element
	    {
	      o_count++;
	      o_nnz[locindex - index->LocStart()]++;
	    }
	  }
	  else											//MatSetValue mode
	  {
	    if( column >= index->TotalDOF() )
	    {
	      ierr	= index->PrintIndices(); CHKERRQ(ierr);
	      ierr	= PetscPrintf(PETSC_COMM_WORLD,"column: %s = %d, %s = %d, %s = %d\t\t",(mlsdown.ToString()).c_str(),index->Indices(down)+1,(mlsup.ToString()).c_str(),index->Indices(up)-1,(photon.ToString()).c_str(),index->Indices(modedim)+1);
	      ierr	= PetscPrintf(PETSC_COMM_WORLD,"locindex = %d mls pitch = %d mode pitch = %d\n",locindex,index->MLSCPitch(up,down),index->ModeIPitch(modedim));
	    }
	    value	= couplingconst*PetscSqrtReal((PetscReal) (mvalue+1))*((PetscScalar) (index->MLSQN(down)+1));
	    ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
      }
      else if ( down == -1 && up != -1 )							//down is n00 and up is not n00
      {
	n00 = index->MLSQN(-1);
	 
	if ( n00 > 0 && index->CanIncrement(up) && index->CanDecrement(modedim) )			//element fulfills requirements
	{
	  column	= locindex + index->MLSIPitch(up) + index->ModeDPitch(modedim);
	  if(!choose)										//preallocation mode
	  {
	    if( index->IsLocal(column)  )							//local element
	    {
	      d_count++;
	      d_nnz[locindex - index->LocStart()]++;
	    }
	    else										//nonlocal element
	    {
	      o_count++;
	      o_nnz[locindex - index->LocStart()]++;
	    }
	  }
	  else											//MatSetValue mode
	  {	
	    if( column >= index->TotalDOF() )
	    {
	      ierr	= index->PrintIndices(); CHKERRQ(ierr);
	      ierr	= PetscPrintf(PETSC_COMM_WORLD,"column: %s = %d, %s = %d, %s = %d\t\t",(mlsdown.ToString()).c_str(),index->Indices(down)-1,(mlsup.ToString()).c_str(),index->Indices(up)+1,(photon.ToString()).c_str(),index->Indices(modedim)-1);
	      ierr	= PetscPrintf(PETSC_COMM_WORLD,"locindex = %d mls pitch = %d mode pitch = %d\n",locindex,index->MLSIPitch(up),index->ModeDPitch(modedim));
	    }
	    value	= couplingconst*PetscSqrtReal((PetscReal) mvalue)*((PetscScalar) (index->MLSQN(up)+1));
	    ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
	  
	if ( index->CanDecrement(up) && index->CanIncrement(modedim) )				//element fulfills requirements, we do not truncate the ground state density dof...
	{
	  column	= locindex + index->MLSDPitch(up) + index->ModeIPitch(modedim);
	  if(!choose)										//preallocation mode
	  {
	    if( index->IsLocal(column)  )							//local element
	    {
	      d_count++;
	      d_nnz[locindex - index->LocStart()]++;
	    }
	    else										//nonlocal element
	    {
	      o_count++;
	      o_nnz[locindex - index->LocStart()]++;
	    }
	  }
	  else											//MatSetValue mode
	  {
	    if( column >= index->TotalDOF() )
	    {
	      ierr	= index->PrintIndices(); CHKERRQ(ierr);
	      ierr	= PetscPrintf(PETSC_COMM_WORLD,"column: %s = %d, %s = %d, %s = %d\t\t",(mlsdown.ToString()).c_str(),index->Indices(down)+1,(mlsup.ToString()).c_str(),index->Indices(up)-1,(photon.ToString()).c_str(),index->Indices(modedim)+1);
	      ierr	= PetscPrintf(PETSC_COMM_WORLD,"locindex = %d mls pitch = %d mode pitch = %d\n",locindex,index->MLSDPitch(up),index->ModeIPitch(modedim));
	    }
	    value	= couplingconst*PetscSqrtReal((PetscReal) (mvalue+1))*((PetscScalar) (n00+1));
	    ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
      }
      else if ( down != -1 && up == -1 )							//down is not n00 and up is n00
      {
	n00 = index->MLSQN(-1);
	  
	if ( index->CanDecrement(down) && index->CanDecrement(modedim) )				//element fulfills requirements, we do not truncate the ground state density dof...
	{
	  column	= locindex + index->MLSDPitch(down) + index->ModeDPitch(modedim);
	  if(!choose)										//preallocation mode
	  {
	    if( index->IsLocal(column)  )							//local element
	    {
	      d_count++;
	      d_nnz[locindex - index->LocStart()]++;
	    }
	    else										//nonlocal element
	    {
	      o_count++;
	      o_nnz[locindex - index->LocStart()]++;
	    }
	  }
	  else											//MatSetValue mode
	  {
	    if( column >= index->TotalDOF() )
	    {
	      ierr	= index->PrintIndices(); CHKERRQ(ierr);
	      ierr	= PetscPrintf(PETSC_COMM_WORLD,"column: %s = %d, %s = %d, %s = %d\t",(mlsdown.ToString()).c_str(),index->Indices(down)-1,(mlsup.ToString()).c_str(),index->Indices(up)+1,(photon.ToString()).c_str(),index->Indices(modedim)-1);
	      ierr	= PetscPrintf(PETSC_COMM_WORLD,"locindex = %d mls pitch = %d mode pitch = %d\n",locindex,index->MLSDPitch(down),index->ModeDPitch(modedim));
	    }
	    value	= couplingconst*PetscSqrtReal((PetscReal) mvalue)*((PetscScalar) (n00+1));
	    ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
	  
	if ( n00 > 0 && index->CanIncrement(down) && index->CanIncrement(modedim) )		//element fulfills requirements
	{
	  column	= locindex + index->MLSIPitch(down) + index->ModeIPitch(modedim);
	  if(!choose)										//preallocation mode
	  {
	    if( index->IsLocal(column)  )							//local element
	    {
	      d_count++;
	      d_nnz[locindex - index->LocStart()]++;
	    }
	    else										//nonlocal element
	    {
	      o_count++;
	      o_nnz[locindex - index->LocStart()]++;
	    }
	  }
	  else											//MatSetValue mode
	  {
	    if( column >= index->TotalDOF() )
	    {
	      ierr	= index->PrintIndices(); CHKERRQ(ierr);
	      ierr	= PetscPrintf(PETSC_COMM_WORLD,"column: %s = %d, %s = %d, %s = %d\t",(mlsdown.ToString()).c_str(),index->Indices(down)+1,(mlsup.ToString()).c_str(),index->Indices(up)-1,(photon.ToString()).c_str(),index->Indices(modedim)+1);
	      ierr	= PetscPrintf(PETSC_COMM_WORLD,"locindex = %d mls pitch = %d mode pitch = %d\n",locindex,index->MLSIPitch(down),index->ModeIPitch(modedim));
	    }
	    value	= couplingconst*PetscSqrtReal((PetscReal) (mvalue+1))*((PetscScalar) (index->MLSQN(down)+1));
	    ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
      }
      else											//down is n00 and up is n00: ERROR
      {
	(*PetscErrorPrintf)("Error: AddHamElPh assembly messed up!\n");
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
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddHamElPh preassembly completed:\n  input: %s %s %s\n",mlsdown.ToString().c_str(),mlsup.ToString().c_str(),photon.ToString().c_str());
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  local elements: %d, \t nonlocal elements: %d \t loc_start: %d, \t loc_end: %d\n",d_count,o_count,index->LocStart(),index->LocEnd()); CHKERRQ(ierr);
	PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
      }
      else
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddHamElPh assembly completed.\n  input: %s %s %s\n",mlsdown.ToString().c_str(),mlsup.ToString().c_str(),photon.ToString().c_str()); CHKERRQ(ierr);
      }
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddMLSCohDrive"

/**
 * @brief	Adds a single MLS-external classical laser interaction Hamiltonian arrow of the MLS sketches to the matrix.
 * 
 * 		mlsdown refers to the "lower" bubble and mlsup to the "higher" bubble in the sketches (assuming ascending energy ordering). Be careful to choose the sign of the matrixelement correctly.
 * 
 * @param	AA		the matrix.
 * @param	d_nnz		the array counting the number of local elements belonging to the diagonal block per row.
 * @param	o_nnz		the array counting the number of local elements belonging to the offdiagonal block per row.
 * @param	choose		0 for preallocation and 1 for actual matrix setup.
 * @param	mlsdown		the name or identifier of the dimension corresponding to the lower bubble.
 * @param	mlsup		the name or identifier of the dimension corresponding to the upper bubble.
 * @param	couplingconst	the coupling constant for the interaction including the i/hbar prefactor of the von-Neumann equation. Cannot be time dependent!
 * 
 */

PetscErrorCode	System::AddMLSCohDrive(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, MLSDim mlsdown, MLSDim mlsup, PetscScalar couplingconst)
{
    PetscErrorCode	ierr;
    
    PetscFunctionBeginUser;
    
    
    //finding the dimensions  
    PetscInt down=0, up=0;
    
    ierr = FindMatch(&mlsdown,&down); CHKERRQ(ierr);
    ierr = FindMatch(&mlsup,&up); CHKERRQ(ierr);
       

    //loop part  
    PetscInt	locindex,column,n00;
    PetscScalar	value;
    PetscInt	d_count = 0, o_count = 0;
    
    locindex	= index->InitializeLocal();
    
    while ( index->ContinueLocal() )					//loop over all local rows
    {
      if ( down != -1 && up != -1 )					//down is not n00 and up is not n00
      {
	if ( index->CanDecrement(down) && index->CanIncrement(up) )
	{
	  column	= locindex + index->MLSCPitch(down,up);
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
	  else							//MatSetValue mode
	  {
	    if( column >= index->TotalDOF() )
	    {
	      ierr	= index->PrintIndices(); CHKERRQ(ierr);
	      ierr	= PetscPrintf(PETSC_COMM_WORLD,"column: %s = %d, %s = %d\t\t",(mlsdown.ToString()).c_str(),index->Indices(down)-1,(mlsup.ToString()).c_str(),index->Indices(up)+1);
	      ierr	= PetscPrintf(PETSC_COMM_WORLD,"locindex = %d mls pitch = %d\n",locindex,index->MLSCPitch(down,up));
	    }
	    value	= couplingconst*((PetscScalar) (index->MLSQN(up)+1));
	    ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
	  
	if ( index->CanDecrement(up) && index->CanIncrement(down) )
	{
	  column	= locindex + index->MLSCPitch(up,down);
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
	  else							//MatSetValue mode
	  {
	    if( column >= index->TotalDOF() )
	    {
	      ierr	= index->PrintIndices(); CHKERRQ(ierr);
	      ierr	= PetscPrintf(PETSC_COMM_WORLD,"column: %s = %d, %s = %d\t\t",(mlsdown.ToString()).c_str(),index->Indices(down)+1,(mlsup.ToString()).c_str(),index->Indices(up)-1);
	      ierr	= PetscPrintf(PETSC_COMM_WORLD,"locindex = %d mls pitch = %d\n",locindex,index->MLSCPitch(up,down));
	    }
	    value	= couplingconst*((PetscScalar) (index->MLSQN(down)+1));
	    ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
      }
      else if ( down == -1 && up != -1 )				//down is n00 and up is not n00
      {
	n00 = index->MLSQN(-1);
	if ( n00 > 0 && index->CanIncrement(up) )
	{
	  column	=  locindex + index->MLSIPitch(up);
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
	      ierr	= index->PrintIndices(); CHKERRQ(ierr);
	      ierr	= PetscPrintf(PETSC_COMM_WORLD,"column: %s = %d, %s = %d\t\t",(mlsdown.ToString()).c_str(),index->Indices(down)-1,(mlsup.ToString()).c_str(),index->Indices(up)+1);
	      ierr	= PetscPrintf(PETSC_COMM_WORLD,"locindex = %d mls pitch = %d\n",locindex,index->MLSIPitch(up));
	    }
	    value	= couplingconst*((PetscScalar) (index->MLSQN(up)+1));
	    ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
	  
	if ( index->CanDecrement(up) )
	{
	  column	=  locindex + index->MLSDPitch(up);
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
	      ierr	= index->PrintIndices(); CHKERRQ(ierr);
	      ierr	= PetscPrintf(PETSC_COMM_WORLD,"column: %s = %d, %s = %d\t\t",(mlsdown.ToString()).c_str(),index->Indices(down)+1,(mlsup.ToString()).c_str(),index->Indices(up)-1);
	      ierr	= PetscPrintf(PETSC_COMM_WORLD,"locindex = %d mls pitch = %d\n",locindex,index->MLSDPitch(up));
	    }
	    value	= couplingconst*((PetscScalar) (n00+1));
	    ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
      }
      else if ( down != -1 && up == -1 )				//down is not n00 and up is n00
      {
	n00 = index->MLSQN(-1);
	if ( index->CanDecrement(down) )
	{
	  column	=  locindex + index->MLSDPitch(down);
	  if(!choose)							//preallocation mode
	  {
	    if( index->IsLocal(column) )				//local element
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
	  else							//MatSetValue mode
	  {
	    if( column >= index->TotalDOF() )
	    {
	      ierr	= index->PrintIndices(); CHKERRQ(ierr);
	      ierr	= PetscPrintf(PETSC_COMM_WORLD,"column: %s = %d, %s = %d\t\t",(mlsdown.ToString()).c_str(),index->Indices(down)-1,(mlsup.ToString()).c_str(),index->Indices(up)+1);
	      ierr	= PetscPrintf(PETSC_COMM_WORLD,"locindex = %d mls pitch = %d\n",locindex,index->MLSDPitch(down));
	    }
	    value	= couplingconst*((PetscScalar) (n00+1));
	    ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
	  
	if ( n00 > 0 && index->CanIncrement(down) )
	{
	  column	=  locindex + index->MLSIPitch(down);
	  if(!choose)							//preallocation mode
	  {
	    if( index->IsLocal(column) )				//local element
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
	      ierr	= index->PrintIndices(); CHKERRQ(ierr);
	      ierr	= PetscPrintf(PETSC_COMM_WORLD,"column: %s = %d, %s = %d\t\t",(mlsdown.ToString()).c_str(),index->Indices(down)+1,(mlsup.ToString()).c_str(),index->Indices(up)-1);
	      ierr	= PetscPrintf(PETSC_COMM_WORLD,"locindex = %d mls pitch = %d\n",locindex,index->MLSIPitch(down));
	    }
	    value	= couplingconst*((PetscScalar) (index->MLSQN(down)+1));
	    ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
      }
      else								//down is n00 and up is n00: ERROR
      {
	(*PetscErrorPrintf)("Error: AddCohDrive assembly messed up!\n");
	(*PetscErrorPrintf)("Seems like both mls input strings are n00!\n");
	SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
      }
      
      locindex = index->Increment();
    }
    
    
    //ouput part
    if(longout || liouout)
    {
      if(!choose)
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddCohDrive preassembly completed:\n  input %s %s\n",mlsdown.ToString().c_str(),mlsup.ToString().c_str()); CHKERRQ(ierr);
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  local elements: %d, \t nonlocal elements: %d \t loc_start: %d, \t loc_end: %d\n",d_count,o_count,index->LocStart(),index->LocEnd()); CHKERRQ(ierr);
	PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
      }
      else
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddCohDrive assembly completed.\n  input %s %s\n",mlsdown.ToString().c_str(),mlsup.ToString().c_str()); CHKERRQ(ierr);
      }
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddModeCohDrive"

/**
 * @brief	Adds the contribution of the semiclassical, coherent driving Hamiltonian of the mode to the matrix. This function assumes a time-independent, rotating frame formulation.
 * 
 * @param	AA		the matrix.
 * @param	d_nnz		the array counting the number of local elements belonging to the diagonal block per row.
 * @param	o_nnz		the array counting the number of local elements belonging to the offdiagonal block per row.
 * @param	choose		0 for preallocation and 1 for actual matrix setup.
 * @param	modenumber	the number of the mode
 * @param	couplingconst	Type PetscScalar; the coupling constant of the as arising in the master equation.
 * 
 */

PetscErrorCode	System::AddModeCohDrive(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt modenumber, PetscScalar couplingconst)
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
	if( index->CanIncrement(mode) )
	{
	  column = locindex + index->ModeIPitch(mode);
	  if(!choose)							//preallocation mode
	  {
	    if( index->IsLocal(column) )
	    {
	      d_count++;						//the element itself is always local
	      d_nnz[locindex - index->LocStart()]++;
	    }
	    else
	    {
	      o_count++;
	      o_nnz[locindex - index->LocStart()]++;
	    }
	  }
	  else								//MatSetValue mode
	  {
	    value	= +couplingconst*PetscSqrtReal((PetscReal) (index->ModeQN(mode)+1));
	    ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
	
	if( index->CanDecrement(mode) )
	{
	  column = locindex + index->ModeDPitch(mode);
	  if(!choose)							//preallocation mode
	  {
	    if( index->IsLocal(column) )
	    {
	      d_count++;						//the element itself is always local
	      d_nnz[locindex - index->LocStart()]++;
	    }
	    else
	    {
	      o_count++;
	      o_nnz[locindex - index->LocStart()]++;
	    }
	  }
	  else								//MatSetValue mode
	  {
	    value	= +couplingconst*PetscSqrtReal((PetscReal) index->ModeQN(mode));
	    ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
	
	if ( index->CanDecrement(mode+1) )	//is that right?
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
	    value	= -couplingconst*PetscSqrtReal((PetscReal) index->ModeQN(mode+1));
	    ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
	
	if ( index->CanIncrement(mode+1) )	//is that right?
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
	    value	= -couplingconst*PetscSqrtReal((PetscReal) (index->ModeQN(mode+1)+1));
	    ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
      
      locindex = index->Increment();
    }
    
    
    //output part
    if(longout || liouout)
    {
      if(!choose)
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddModeCohDrive preassembly completed:\n  input m%d\n",modenumber); CHKERRQ(ierr);
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  local elements: %d, \t nonlocal elements: %d \t loc_start: %d, \t loc_end: %d\n",d_count,o_count,index->LocStart(),index->LocEnd()); CHKERRQ(ierr);
	PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
      }
      else
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAddModeCohDrive assembly completed.\n  input m%d\n",modenumber); CHKERRQ(ierr);
      }
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddLindbladRelaxMLS"

/**
 * @brief	Adds a single density exchanging arrow of the Lindblad dissipators to the matrix. Has two modes one for preallocation and one for actual matrix setup.
 * 		
 * 		This actually corresponds to two basic arrows, one AddMLSSingleArrowNonconnecting() arrow leading to decay of the population and one AddMLSSingleArrowConnecting() leading to the driving of the "lower" state.
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

PetscErrorCode	System::AddLindbladRelaxMLS(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, MLSDim mlsstart, MLSDim mlsgoal, PetscReal couplingconst)
{
    PetscErrorCode	ierr;
    
    PetscFunctionBeginUser;
    
    
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
	if(!choose)							//preallocation mode
	{
	  d_count++;							//the element itself is always local
	  d_nnz[locindex - index->LocStart()]++;
	}
	else								//MatSetValue mode
	{
	  value	= -2*couplingconst*((PetscScalar) index->MLSQN(start));
	  ierr	= MatSetValue(AA,locindex,locindex,value,ADD_VALUES); CHKERRQ(ierr);
	}
	
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
	      value	= 2*couplingconst*((PetscScalar) (index->MLSQN(start)+1));
	      ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
      }
      else if( start != -1 && goal == -1 )				//start is not n00 and goal is n00
      {
	if(!choose)							//preallocation mode
	{
	  d_count++;							//the element itself is always local
	  d_nnz[locindex - index->LocStart()]++;
	}
	else								//MatSetValue mode
	{
	  value	= -2*couplingconst*((PetscScalar) index->MLSQN(start));
	  ierr	= MatSetValue(AA,locindex,locindex,value,ADD_VALUES); CHKERRQ(ierr);
	}
	    
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
	    value	= 2*couplingconst*((PetscScalar) (index->MLSQN(start)+1));
	    ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
      }
      else if( start == -1 && goal != -1 )				//start is n00
      {
	n00	= index->MLSQN(-1);
	if(!choose)							//preallocation mode
	{
	  d_count++;
	  d_nnz[locindex - index->LocStart()]++;
	}
	else								//MatSetValue mode
	{
	  value	= -2*couplingconst*((PetscScalar) n00);
	  ierr	= MatSetValue(AA,locindex,locindex,value,ADD_VALUES); CHKERRQ(ierr);
	}
			
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
	    value	= 2*couplingconst*((PetscScalar) (n00+1));
	    ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
      }
      else								//down is n00 and up is n00: ERROR
      {
	(*PetscErrorPrintf)("Error: LindbladRelaxMLS assembly messed up!\n");
	(*PetscErrorPrintf)("Seems like both mls input strings are n00!\n");
	SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
      }
      
      locindex = index->Increment();
    }
    
    
    //ouput part
    if(longout || liouout)
    {
      if(!choose)
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nLindbladRelaxMLS preassembly completed:\n  input %s %s\n",mlsstart.ToString().c_str(),mlsgoal.ToString().c_str()); CHKERRQ(ierr);
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  local elements: %d, \t nonlocal elements: %d \t loc_start: %d, \t loc_end: %d\n",d_count,o_count,index->LocStart(),index->LocEnd()); CHKERRQ(ierr);
	PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
      }
      else
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nLindbladRelaxMLS assembly completed.\n  input %s %s\n",mlsstart.ToString().c_str(),mlsgoal.ToString().c_str()); CHKERRQ(ierr);
      }
    }
    
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddLindbladDephMLS"

/**
 * @brief	Adds a single dephasing arrow of the Lindblad dissipators to the matrix. 
 * 
 * 		Besides the treatment of the of the matrix element this function is equivalent to AddMLSSingleArrowNonconnecting().
 * 
 * @param	AA		the matrix.
 * @param	d_nnz		the array counting the number of local elements belonging to the diagonal block per row.
 * @param	o_nnz		the array counting the number of local elements belonging to the offdiagonal block per row.
 * @param	choose		0 for preallocation and 1 for actual matrix setup.
 * @param	mlselem		the name or identifier of the dimension corresponding to the polarization bubble.
 * @param	couplingconst	the coupling constant of the as arising in the master equation.
 * 
 */

PetscErrorCode	System::AddLindbladDephMLS(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, MLSDim mlselem, PetscReal matrixelem)
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
      else								//MatSetValue mode
      {
	value	= -matrixelem*((PetscScalar) index->MLSQN(elem));	//action on the element itself is pretty much always present, so the indices[elem] == 0 case should not produce any overhead
	ierr	= MatSetValue(AA,locindex,locindex,value,ADD_VALUES); CHKERRQ(ierr);
      }
      
      locindex = index->Increment();
    }
    
    
    //ouput part
    if(longout || liouout)
    {
      if(!choose)
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nLindbladDephMLS preassembly completed:\n  input %s\n",mlselem.ToString().c_str()); CHKERRQ(ierr);
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  local elements: %d, \t nonlocal elements: %d \t loc_start: %d, \t loc_end: %d\n",d_count,o_count,index->LocStart(),index->LocEnd()); CHKERRQ(ierr);
	PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
      }
      else
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nLindbladDephMLS assembly completed.\n  input %s\n",mlselem.ToString().c_str()); CHKERRQ(ierr);
      }
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddLindbladMode"

/**
 * @brief	Adds a simple decay dissipator for a bosonic mode to the matrix.
 * 
 * @param	AA		the matrix.
 * @param	d_nnz		the array counting the number of local elements belonging to the diagonal block per row.
 * @param	o_nnz		the array counting the number of local elements belonging to the offdiagonal block per row.
 * @param	choose		0 for preallocation and 1 for actual matrix setup.
 * @param	Name		the name or identifier of the mode dimension, i.e. mX NOT dmX. Treats both mode degrees of freedom automatically!
 * @param	couplingconst	the coupling constant of the as arising in the master equation.
 * 
 */

PetscErrorCode	System::AddLindbladMode(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt modenumber, PetscReal couplingconst)
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
	if(!choose)							//preallocation mode
	{
	  d_count++;							//the element itself is always local
	  d_nnz[locindex - index->LocStart()]++;
	}
	else								//MatSetValue mode
	{
	  value	= -couplingconst*((PetscScalar) (index->ModeQN(mode)+index->ModeQN(mode+1)));
	  ierr	= MatSetValue(AA,locindex,locindex,value,ADD_VALUES); CHKERRQ(ierr);
	}
	
	if ( index->CanIncrement(mode) && index->CanIncrement(mode+1) )	//is that right?
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
	    value	= + 2*couplingconst*PetscSqrtReal((PetscReal) ((index->ModeQN(mode)+1)*(index->ModeQN(mode+1)+1)));
	    ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}
      
      locindex = index->Increment();
    }
    
    
    //output part
    if(longout || liouout)
    {
      if(!choose)
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nLindbladMode preassembly completed:\n  input m%d\n",modenumber); CHKERRQ(ierr);
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  local elements: %d, \t nonlocal elements: %d \t loc_start: %d, \t loc_end: %d\n",d_count,o_count,index->LocStart(),index->LocEnd()); CHKERRQ(ierr);
	PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
      }
      else
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nLindbladMode assembly completed.\n  input m%d\n",modenumber); CHKERRQ(ierr);
      }
    }
    
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddLindbladModeThermal"

/**
 * @brief	Adds a Lindblad dissipator for a bosonic mode coupled to a thermal bath to the matrix.
 * 
 * @param	AA		the matrix.
 * @param	d_nnz		the array counting the number of local elements belonging to the diagonal block per row.
 * @param	o_nnz		the array counting the number of local elements belonging to the offdiagonal block per row.
 * @param	choose		0 for preallocation and 1 for actual matrix setup.
 * @param	Name		the name or identifier of the mode dimension, i.e. mX NOT dmX. Treats both mode degrees of freedom automatically!
 * @param	couplingconst	the coupling constant of the as arising in the master equation.
 * @param	beta		the beta factor 1/(k_B T)
 * 
 */

PetscErrorCode	System::AddLindbladModeThermal(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt modenumber, PetscReal couplingconst, PetscReal beta)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    
    //finding the dimensions
    PetscInt	mode;
    ModeDim	modedim (0,modenumber);
    
    ierr = FindMatch(&modedim,&mode); CHKERRQ(ierr);
    
    
    //loop part  
    PetscReal	energy = 0.0, mmean;				//the default value produces a -nan if the energy was not set properly and this function is called.
    
    ierr  = Energies(mode,&energy); CHKERRQ(ierr);
//     ierr = PetscPrintf(PETSC_COMM_WORLD,"\nenergy %e\n",(double) energy); CHKERRQ(ierr);
    mmean = 1.0/(PetscExpReal(beta*energy)-1.0);		//mmean occupation in the mode, beta and temperature are given in constants.hpp
//     mmean = 1.0;		//mmean occupation in the mode, beta and temperature are given in constants.hpp
    
    PetscInt	locindex,column;
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
	else								//MatSetValue mode
	{
	  value	= -couplingconst*(mmean+1.0)*((PetscScalar) (index->ModeQN(mode)+index->ModeQN(mode+1)));
	  
	  if( index->CanIncrement(mode) )	value -= couplingconst*mmean*(index->ModeQN(mode)+1);
	  if( index->CanIncrement(mode+1) )	value -= couplingconst*mmean*(index->ModeQN(mode+1)+1);
	  
	  ierr	= MatSetValue(AA,locindex,locindex,value,ADD_VALUES); CHKERRQ(ierr);
	}
	
	if ( index->CanIncrement(mode) && index->CanIncrement(mode+1) )	//is that right?
	{
	  column = locindex + index->ModeIPitch(mode) + index->ModeIPitch(mode+1);
	  if(!choose)							//preallocation mode
	  {
	    if( index->IsLocal(column) )				//local element
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
	    value	= +2*couplingconst*(mmean+1.0)*PetscSqrtReal((PetscReal) ((index->ModeQN(mode)+1)*(index->ModeQN(mode+1)+1)));
	    ierr	= MatSetValue(AA,locindex,column,value,ADD_VALUES); CHKERRQ(ierr);
	  }
	}	
	if ( index->CanDecrement(mode) && index->CanDecrement(mode+1) )	//is that right?
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
	    value	= + 2*couplingconst*mmean*PetscSqrtReal((PetscReal) ((index->ModeQN(mode))*(index->ModeQN(mode+1))));
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
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nLindbladModeThermal preassembly completed:\n  input m%d\n", modenumber); CHKERRQ(ierr);
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  local elements: %d, \t nonlocal elements: %d \t loc_start: %d, \t loc_end: %d\n",d_count,o_count,index->LocStart(),index->LocEnd()); CHKERRQ(ierr);
	PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
      }
      else
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\nLindbladModeThermal assembly completed.\n  input m%d\n", modenumber); CHKERRQ(ierr);
      }
    }
    
    PetscFunctionReturn(0);
}

