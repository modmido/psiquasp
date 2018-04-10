
/**
 * @file	sutil.cpp
 * 
 * 		    Definition of the general System class methods except for the Liouvillian methods.
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
//----  constructors, destructors
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

/**
 * @brief	Default constructor.
 *
 */

System::System ()
{
    num_dims		= 0;
    num_mlsdims		= 0;
    num_mlsdens		= 0;
    num_modes		= 0;
    loc_size		= 0;
    N_D_MLS		    = 0;
    useMulti        = 1;        //used for sanity check
    
    parallel_layout	= 0;
    modesetup		= 0;
    
    longout		= PETSC_FALSE;
    propout		= PETSC_FALSE;
    liouout		= PETSC_FALSE;
    
    numparams		= 0;
    
    real_value_tolerance	= 1.e-10;
    hermitian_tolerance		= 1.e-10;
}


/**
 * @brief	Virtual default destructor.
 *
 */

System::~System()
{
    delete	index;
}


//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  Add dimensions
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

#undef __FUNCT__
#define __FUNCT__ "MLSAddMulti"

/**
 * @brief    Adds a single type of mls. Only used when one type of mls is used.
 *
 * @param    nmls        the number of individual mls of this type
 *
 */

PetscErrorCode System::MLSAdd(PetscInt nmls)
{
    PetscFunctionBeginUser;
    
    if ( useMulti == 0 )             // has there been a prior call to MLSAdd?
    {
        (*PetscErrorPrintf)("Mutliple calls to MLSAdd() are not allowed. Use MLSAddMulti() for different types of MLS!\n");
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
    }
    else if( N_D_MLS > 0 )           // is there already another type of mls set?
    {
        (*PetscErrorPrintf)("Cannot mix MLSAdd() and MLSAddMulti() function calls!\n");
        (*PetscErrorPrintf)("If you want to use multiple mls types use MLSAddMulti() only!\n");
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
    }
    else if ( modesetup )       // have there been already calls to ModeAdd()? Not allowed!
    {
        (*PetscErrorPrintf)("Cannot mix MLS and Mode dimension setup functions!\n");
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
    }
    else
    {
        N_MLS[N_D_MLS]          = nmls;
        useMulti                = 0;
        N_D_MLS++;
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MLSAddMulti"

/**
 * @brief    Adds a different type of mls. Not needed when only one type of mls is considered.
 *
 * @param    nmls        the number of individual mls of this type
 *
 */

PetscErrorCode System::MLSAddMulti(PetscInt nmls)
{
    PetscFunctionBeginUser;
    
    if(N_D_MLS == MAX_D_MLS)        //is the allowed maximum of different mls reached?
    {
        (*PetscErrorPrintf)("Maximum number of different MLS types reached!\n");
        (*PetscErrorPrintf)("If you need to use more different types of mls then increase the MAX_D_MLS preprocessor constant in system.hpp.\n");
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MAX_VALUE,"");
    }
    else if( N_D_MLS > 0 )          //has there been a prior call to MLSAdd
    {
        if ( multiMLS_start[N_D_MLS-1] == num_dims )    // are there two consecutive calls to MLSAdd without adding any dimensions in between?
        {
            (*PetscErrorPrintf)("Two consecutive calls to MLSAddMulti() without adding any mls dimenstions in between!\n");
            (*PetscErrorPrintf)("Maybe you have forgotten to add the MLS dimensions. Or remove one call to MLSAddMulti().\n");
            SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MAX_VALUE,"");
        }
        if ( modesetup )    // have there been already calls to ModeAdd()? Not allowed!
        {
            (*PetscErrorPrintf)("Cannot mix MLS and Mode dimension setup functions!\n");
            SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
        }
        if( useMulti == 0 )
        {
            (*PetscErrorPrintf)("Cannot mix MLSAdd() and MLSAddMulti() function calls!\n");
            (*PetscErrorPrintf)("If you want to use multiple mls types use MLSAddMulti() only!\n");
            SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
        }
    }
    else                            //everything seems to be legit, so proceed
    {
        N_MLS[N_D_MLS]          = nmls;         //maximum number of mls for this type
        multiMLS_start[N_D_MLS] = num_dims;     //the index of the first dimension for this kind
        N_D_MLS++;                              //raise the number of different types by one
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MLSAddDens"

/**
 * @brief	Adds a mls density dimension.
 * 
 * @param	n   		the order of the density, i.e. the number of the ket and bra of the projector
 * @param	length		the length of the dim.
 * @param	energy		optional energy value for corresponding level. Only needed for thermal start values of the density matrix.
 * 
 */

PetscErrorCode System::MLSAddDens(PetscInt n, PetscInt length, PetscReal energy)
{
    PetscFunctionBeginUser;
    
    if( modesetup )
    {
      (*PetscErrorPrintf)("Cannot mix MLS and Mode dimension setup functions!\n");
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
    }
    else if ( N_D_MLS > 1 )
    {
        (*PetscErrorPrintf)("More than one type of MLS in use. Please use the System::MLSAddDens(MLSDim * indim, PetscInt length, PetscReal energy) function instead!\n");
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
    }
    else if ( N_D_MLS == 0 )
    {
        (*PetscErrorPrintf)("Please call MLSAdd() before adding any dimensions!\n");
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
    }
    else
    {
      if( length <= N_MLS[N_D_MLS-1] +1 )    //during setup the current MLS number maximum is called like this
      { 
          MLSDim	*dim = new MLSDim (n,n,0,length,energy);
	
          dimensions.push_back(dim);
          num_mlsdens++;
          num_mlsdims++;
          num_dims++;
      }
      else
      {
          (*PetscErrorPrintf)("Invalid input for mls density:\n");
          (*PetscErrorPrintf)("Current entry is: n(%d,%d) with length %d (MLS+1 = %d)\n",n,n,length,N_MLS[N_D_MLS-1]+1);
          SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
      }
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MLSAddDens"

/**
 * @brief	Adds a mls density dimension.
 * 
 * @param	indim		the dimension
 * @param	length		the length of the dim.
 * @param	energy		optional energy value for corresponding level. Only needed for thermal start values of the density matrix.
 * 
 */

PetscErrorCode System::MLSAddDens(MLSDim * indim, PetscInt length, PetscReal energy)
{
    PetscFunctionBeginUser;
    
    MLSDim         *single  = dynamic_cast<MLSDim*> (indim);
    MultiMLSDim    *multi   = dynamic_cast<MultiMLSDim*> (indim);
    
    if( modesetup )
    {
      (*PetscErrorPrintf)("Cannot mix MLS and Mode dimension setup functions!\n");
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
    }
    else if ( N_D_MLS == 0 )
    {
        (*PetscErrorPrintf)("Please call MLSAdd() or MLSAddMulti() before adding any dimensions!\n");
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
    }
    else if ( single && !multi )
    {
        if( N_D_MLS > 1 )
        {
            (*PetscErrorPrintf)("Use of MLSDim identifier together with multiple types of MLS.\n");
            (*PetscErrorPrintf)("Please use MultiMLSDim instead.\n");
            SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
        }
        else if( length <= N_MLS[N_D_MLS-1]+1 && indim->IsDensity() )   //during setup the current MLS number maximum is called like this
        {
          MLSDim	*dim = new MLSDim (*indim,0,length,energy);
	
          dimensions.push_back(dim);
          num_mlsdens++;
          num_mlsdims++;
          num_dims++;
        }
        else
        {
          (*PetscErrorPrintf)("Invalid input for mls density:\n");
          (*PetscErrorPrintf)("Current input is: %s with length %d (MLS+1 = %d)\n",indim->ToString().c_str(),length,N_MLS[N_D_MLS-1]+1);
          SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
        }
    }
    else if ( multi )
    {
        if( useMulti == 0 )
        {
            (*PetscErrorPrintf)("Use of MultiMLSDim identifier together with single MLS type operation aka. MLSAdd() call.\n");
            (*PetscErrorPrintf)("Please use MLSDim instead.\n");
            SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
        }
        else if( length <= N_MLS[N_D_MLS-1]+1 && indim->IsDensity() )   //during setup the current MLS number maximum is called like this
        {
            MultiMLSDim    *dim = new MultiMLSDim (*multi,0,length,energy);
            
            dimensions.push_back(dim);
            num_mlsdens++;
            num_mlsdims++;
            num_dims++;
        }
        else
        {
            (*PetscErrorPrintf)("Invalid input for mls density:\n");
            (*PetscErrorPrintf)("Current input is: %s with length %d (MLS+1 = %d)\n",indim->ToString().c_str(),length,N_MLS[N_D_MLS-1]+1);
            SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
        }
    }
    
    PetscFunctionReturn(0);
}

 
#undef __FUNCT__
#define __FUNCT__ "MLSAddPol"

/**
 * @brief	Adds a mls polarization dimension.
 * 
 * @param	ket 	the number of the dimension.
 * @param   bra     the number of the dimension.
 * @param	length	the length of the dimension.
 * 
 */

PetscErrorCode System::MLSAddPol(PetscInt ket, PetscInt bra, PetscInt length)
{
    PetscFunctionBeginUser;
    
    if( modesetup )
    {
      (*PetscErrorPrintf)("Cannot mix MLS and Mode dimension setup functions!\n");
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
    }
    else if ( N_D_MLS > 1 )
    {
        (*PetscErrorPrintf)("More than one type of MLS in use. Please use the System::MLSAddPol(MLSDim * indim, PetscInt length, PetscReal energy) function instead!\n");
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
    }
    else if ( N_D_MLS == 0 )
    {
        (*PetscErrorPrintf)("Please call MLSAdd() before adding any dimensions!\n");
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
    }
    else
    {
      if( length <= N_MLS[N_D_MLS-1]+1 )
      {
          MLSDim	*dim = new MLSDim (ket,bra,1,length,0.0);
	
          dimensions.push_back(dim);
          num_mlsdims++;
          num_dims++;
      }
      else
      {
          (*PetscErrorPrintf)("Invalid input for mls polarization!\n");
          (*PetscErrorPrintf)("Current input is: n(%d,%d) with length %d (MLS+1 = %d)\n",ket,bra,length,N_MLS[N_D_MLS-1]+1);
          SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
      }
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MLSAddPol"

/**
 * @brief	Adds a mls polarization dimension.
 * 
 * @param	indim	the MLSDim identifier.
 * @param	length	the length of the dimension.
 * 
 */

PetscErrorCode System::MLSAddPol(MLSDim * indim, PetscInt length)
{
    PetscFunctionBeginUser;
    
    MLSDim         *single  = dynamic_cast<MLSDim*> (indim);
    MultiMLSDim    *multi   = dynamic_cast<MultiMLSDim*> (indim);
    
    if( modesetup )
    {
      (*PetscErrorPrintf)("Cannot mix MLS and Mode dimension setup functions!\n");
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
    }
    else if ( N_D_MLS == 0 )
    {
        (*PetscErrorPrintf)("Please call MLSAdd() or MLSAddMulti() before adding any dimensions!\n");
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
    }
    else if ( single && !multi )
    {
        if( N_D_MLS > 1 )
        {
            (*PetscErrorPrintf)("Use of MLSDim identifier together with multiple types of MLS.\n");
            (*PetscErrorPrintf)("Please use MultiMLSDim instead.\n");
            SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
        }
        else if( length <= N_MLS[N_D_MLS-1]+1 && !indim->IsDensity() )
        {
            MLSDim	*dim = new MLSDim (*indim,1,length,0.0);
	
            dimensions.push_back(dim);
            num_mlsdims++;
            num_dims++;
        }
        else
        {
            (*PetscErrorPrintf)("Invalid input for mls polarization!\n");
            (*PetscErrorPrintf)("Current input is: %s with length %d (MLS+1 = %d)\n",indim->ToString().c_str(),length,N_MLS[N_D_MLS-1]+1);
            SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
        }
    }
    else if ( multi )
    {
        if( useMulti == 0 )
        {
            (*PetscErrorPrintf)("Use of MultiMLSDim identifier together with single MLS type operation aka. MLSAdd() call.\n");
            (*PetscErrorPrintf)("Please use MLSDim instead.\n");
            SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
        }
        else if( length <= N_MLS[N_D_MLS-1]+1 && !indim->IsDensity() )   //during setup the current MLS number maximum is called like this
        {
            MultiMLSDim    *dim = new MultiMLSDim (*multi,1,length,0.0);
            
            dimensions.push_back(dim);
            num_mlsdims++;
            num_dims++;
        }
        else
        {
            (*PetscErrorPrintf)("Invalid input for mls polarization:\n");
            (*PetscErrorPrintf)("Current input is: %s with length %d (MLS+1 = %d)\n",indim->ToString().c_str(),length,N_MLS[N_D_MLS-1]+1);
            SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
        }
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ModeAdd"

/**
 * @brief	Adds a bosonic mode. Sets mode_nums, num_modes, energies and calls twice AddDimension since a mode has two degrees of freedom.
 * 
 * @param	length		the length of the mode dim to be added, i.e. the maximum number state plus one!
 * @param	offdiag		the maximum number of offdiagonals, per side, i.e. if you want 2 offdiagonals per side then type 2 here.
 * @param	energy		the energy of an elementary excitation in the mode.
 * 
 */

PetscErrorCode System::ModeAdd(PetscInt length, PetscInt offdiag, PetscReal energy)
{
    PetscFunctionBeginUser;
    
    ModeDim	*ketdim = new ModeDim (0,num_modes,length,energy);
    ModeDim	*bradim = new ModeDim (1,num_modes,offdiag,0.0);
    
    dimensions.push_back(ketdim);
    dimensions.push_back(bradim);
    num_dims += 2;
    num_modes++;
    
    if( !modesetup )	modesetup = 1;		//set the mode flag
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FindMatch"

/**
 * @brief	Searches the name list for a match with name. Returns the number of the dimension in the order they were set by the user, starting with zero.
 * 
 * @param	name	the name of the dimension.
 * @param	n	the return type, number of the dimension
 * 
 */

PetscErrorCode System::FindMatch(Dim * name, PetscInt * n)
{
    PetscFunctionBeginUser;
    
    PetscInt		num=0;
    
    if( !(name->n00) )
    {
      std::list<Dim*>::iterator it=dimensions.begin();
      Dim * temp = *it;
    
      while ( !(temp->IsEqual(name)) )
      {
          num++;
          it++;
          temp = *it;
	
          if( it == dimensions.end() )
          {
              (*PetscErrorPrintf)("Error: No match for current dimension %s!\n",(name->ToString()).c_str());
              SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
          }
      }
    
      *n = num;
    }
    else
    {
        MLSDim          *ptr1 = dynamic_cast<MLSDim*> (name);
        MultiMLSDim     *ptr2 = dynamic_cast<MultiMLSDim*> (name);
        
        if(ptr1)        *n = -ptr1->mlsTypeNumber-1;
        else if(ptr2)   *n = -ptr2->mlsTypeNumber-1;
        else
        {
            (*PetscErrorPrintf)("Error: n00 error: %s!\n",(name->ToString()).c_str());
            SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
        }
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MLSDimMaxVal"

/**
 * @brief	Returns the length of the MLS dimension with dimension index n. If the corresponding dimension is not a MLS dimension then this returns an error.
 * 
 * @param	n	the number of the dimension
 * @param	ret	the lenght of the corresponding dimension
 * 
 */

PetscErrorCode System::MLSDimMaxVal(PetscInt n,PetscInt *ret)
{
    PetscFunctionBeginUser;
    
    PetscInt		i;
    
    if( n < num_mlsdims )
    {
      std::list<Dim*>::iterator it=dimensions.begin();
      for(i=0; i < n; i++)	it++;
    
      MLSDim		*ptr = dynamic_cast<MLSDim*>(*it);		//dynamic cast checks whether its actually a real MLSDim, should also work for MultiMLSDim
    
      if(ptr)		*ret = ptr->dimlength;
      else
      {
          (*PetscErrorPrintf)("System::MLSDimMaxVal() error. The checked dimension is not a MLS dimension! dim number = %d\n",n);
          SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
      }
    }
    else
    {
        (*PetscErrorPrintf)("System::MLSDimMaxVal() error. The dimension number is too large! dim number = %d, num mls dims = %d \n",n, num_mlsdims);
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ModeDimLen"

/**
 * @brief	Returns the length of the mode dimension with dimension index n. If the corresponding dimension is not a mode dimension then this returns an error.
 * 
 * @param	n	the number of the dimension
 * @param	ret	the lenght of the corresponding dimension
 * 
 */

PetscErrorCode System::ModeDimLen(PetscInt n,PetscInt *ret)
{
    PetscFunctionBeginUser;
    
    PetscInt		i;
    
    if( n < 2*num_modes )
    {
      std::list<Dim*>::iterator it=dimensions.begin();
      for(i=0; i < n+num_mlsdims; i++)	it++;
    
      ModeDim		*ptr = dynamic_cast<ModeDim*>(*it);		//dynamic cast checks whether its actually a real ModeDim
    
      if(ptr)		*ret = ptr->dimlength;
      else
      {
          (*PetscErrorPrintf)("System::ModeDimLen() error. The checked dimension is not an mode dimension! dim_number = %d\n",n);
          SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
      }
    }
    else
    {
        (*PetscErrorPrintf)("System::ModeDimLen() error. The mode dimension number is too large! dim number = %d, num mode dims = %d \n",n, 2*num_modes);
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "IsMLSDimPol"

/**
 * @brief	Returns 0 for a dimension, 1 for a polarization and an error if the dimension n is not a MLS dimension.
 * 
 * @param	n	the number of the dimension
 * @param	ret	the polarization flag of the corresponding dimension
 * 
 */

PetscErrorCode System::IsMLSDimPol(PetscInt n,PetscInt *ret)
{
    PetscFunctionBeginUser;
    
    PetscInt		i;
    
    if( n < num_mlsdims )
    {
      std::list<Dim*>::iterator it=dimensions.begin();
      for(i=0; i < n; i++)	it++;
    
      MLSDim		*ptr = dynamic_cast<MLSDim*>(*it);		//dynamic cast checks whether its actually a real ModeDim
    
      if(ptr)		*ret = ptr->ispol;
      else
      {
          (*PetscErrorPrintf)("System::IsMLSDimPol() error. The checked dimension is not a MLS dimension! dim number = %d\n",n);
          SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
      }
    }
    else
    {
        (*PetscErrorPrintf)("System::IsMLSDimPol() error. The dimension number is too large! dim number = %d, num mls dims = %d \n",n, num_mlsdims);
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "Energies"

/**
 * @brief	Returns the energy for a mls density dimension or the ket dimension of a mode, 0 for a mls polarization dimension of the bra dimension of a mode, and an error if the dimension n is out of bounds.
 * 
 * @param	n	the number of the dimension
 * @param	ret	the energy of the corresponding dimension
 * 
 */

PetscErrorCode System::Energies(PetscInt n,PetscReal *ret)
{
    PetscFunctionBeginUser;
    
    PetscInt		i,size = (PetscInt) dimensions.size();
    
    if( n < size )
    {
      std::list<Dim*>::iterator it=dimensions.begin();
      for(i=0; i < n; i++)	it++;
    
      *ret = (*it)->energy;
    }
    else
    {
      (*PetscErrorPrintf)("System::Energies() error. The dimension number is too large! dim number = %d, num dims = %d \n",n,dimensions.size());
	SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PrintEnergies"

/**
 * @brief	Prints the energies of all set dimensions into stdout.
 * 
 */

PetscErrorCode System::PrintEnergies()
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\ndimension energies: "); CHKERRQ(ierr);
    for(std::list<Dim*>::iterator it = dimensions.begin(); it != dimensions.end(); ++it)
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"%s: %f\t",(*it)->ToString().c_str(),(float) (*it)->energy); CHKERRQ(ierr);
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\n"); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PrintNames"

/**
 * @brief	Prints the names of all dimensions into stdout.
 * 
 */

PetscErrorCode System::PrintNames()
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\ndimensions: "); CHKERRQ(ierr);
    for(std::list<Dim*>::iterator it = dimensions.begin(); it != dimensions.end(); ++it)
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"%s\t",(*it)->ToString().c_str()); CHKERRQ(ierr);
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\n"); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PrintDimlengths"

/**
 * @brief	Prints the lengths of all dimensions into stdout.
 * 
 */

PetscErrorCode System::PrintDimlengths()
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\ndimension lengths: "); CHKERRQ(ierr);
    for(std::list<Dim*>::iterator it = dimensions.begin(); it != dimensions.end(); ++it)
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"%s: %d\t",(*it)->ToString().c_str(), (*it)->dimlength); CHKERRQ(ierr);
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\n"); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ExtractMLSDimLengths"

/**
 * @brief	Extracts an array of length num_mlsdims that contains all the dimlengths of all MLS dimensions.
 * 
 * @param	ret	the return array. gets allocated during function call.
 */

PetscErrorCode System::ExtractMLSDimLengths(PetscInt **ret)
{
    PetscFunctionBeginUser;
    
    PetscInt		i;
    PetscInt		*loc = new PetscInt [num_mlsdims];
    
    std::list<Dim*>::iterator it=dimensions.begin();
    for(i=0; i < num_mlsdims; i++)
    {
      loc[i] = (*it)->dimlength;
      it++;
    }
    
    *ret = loc;
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ExtractModeDimLengths"

/**
 * @brief	Extracts an array of length 2*num_modes that contains all the dimlengths of all mode dimensions.
 * 
 * @param	ret	the return array. gets allocated during function call.
 */

PetscErrorCode System::ExtractModeDimLengths(PetscInt **ret)
{
    PetscFunctionBeginUser;
    
    PetscInt		i;
    PetscInt		*loc = new PetscInt [2*num_modes];
    
    std::list<Dim*>::iterator it=dimensions.begin();
    for(i=0; i < num_mlsdims; i++)	it++;
    
    for(i=0; i < 2*num_modes; i++)
    {
      loc[i] = (*it)->dimlength;
      it++;
    }
    
    *ret = loc;
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ExtractMLSDimPol"

/**
 * @brief	Extracts an array of length num_mlsdims that contains all the polarization flags of all mls dimensions.
 * 
 * @param	ret	the return array. gets allocated during function call.
 */

PetscErrorCode System::ExtractMLSDimPol(PetscInt **ret)
{
    PetscFunctionBeginUser;
    
    PetscInt		i;
    PetscInt		*loc = new PetscInt [num_mlsdims];
    
    std::list<Dim*>::iterator it=dimensions.begin();
    for(i=0; i < num_mlsdims; i++)
    {
      MLSDim		*ptr = (MLSDim*) *it;
      loc[i] = ptr->ispol;
      it++;
    }
    
    *ret = loc;
    
    PetscFunctionReturn(0);
}



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  Initialize index, vectors and matrices
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

#undef __FUNCT__
#define __FUNCT__ "PQSPSetup"

/**
 * @brief	This function has to be called after all dimensions have been set and before the user does something else! Intitializes the index and the parallel layout via creating the first parallel Petsc vector.
 * 		Must be called after all the dimensions have been added. Dimensions that are added afterwards will not be included and may cause serious errors.
 * 		First the System::index is created via Index::SetupIndex(). This function sets everything that is needed for a uniprocessor Index object. 
 * 		Especially it computes the total number of degrees of freedom, i.e. the number of entries of the density matrix. This is needed to create the first vector, which in turn sets the internal Petsc parallel layout.
 * 		This however may be changed afterwards through appropriate renumbering.
 * 
 * @param	dm		the vector for e.g. density matrix
 * @param	matrices	the number of Liouvillian matrices that should be created. The user can also create more matrices using the public PQSPCreateMat() function directly
 * @param	AAs		one or possibly more than one matrices
 * 
 */

PetscErrorCode System::PQSPSetup(Vec * dm, PetscInt matrices, Mat * AAs)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    
    //print all options to stdout
    PetscBool		flg;
    
    ierr = PetscOptionsHasName(NULL,NULL,"-pqsp_options",&flg); CHKERRQ(ierr);
    if(flg)
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nOptions table for PsiQuaSP:\n"); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"  -realvaluetol: set the maximum acceptable value for the imaginary part of a hermitian observable\n"); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"  -hermitiantol: set the maximum acceptable deviation from hermitianity, when using VecIsHermitian()\n"); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"  -long_output: print long output of all setup routines\n"); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"  -prop_output: print long output of the output object setup routines\n"); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"  -liouville_output: print long output of the Liouvillian setup routines\n"); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"  -index_output: print information about the Index object\n"); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"  -tev_steps_monitor: print every ... time step into output file for TEVMonitor()\n"); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\n"); CHKERRQ(ierr);
    }
    
    //setup
    PetscInt		i,start,end;
    ierr = PQSPSetupIndex(); CHKERRQ(ierr);			//this creates the uniprocessor index and computes the total size of the liouville space
    ierr = PQSPCreateVec(dm,&start,&end); CHKERRQ(ierr);	//this size is needed to create the first vector, usually the density matrix, which in turn sets the internal Petsc parallel layout
  
    index->SetParallelLayout(start,end);			//the parallel layout is then used to set the parallel layout of the index
    
    for(i=0; i < matrices; i++)
    {
      ierr = PQSPCreateMat( &(AAs[i]) ); CHKERRQ(ierr); 
    }
    
    
    //command line convergence 
    PetscReal		value;
    
    ierr = PetscOptionsHasName(NULL,NULL,"-realvaluetol",&flg); CHKERRQ(ierr);
    if(flg)
    {
      ierr = PetscOptionsGetReal(NULL,NULL,"-realvaluetol",&value,NULL);CHKERRQ(ierr);
      real_value_tolerance = value;
    }
    ierr = PetscOptionsHasName(NULL,NULL,"-hermitiantol",&flg); CHKERRQ(ierr);
    if(flg)
    {
      ierr = PetscOptionsGetReal(NULL,NULL,"-hermitiantol",&value,NULL);CHKERRQ(ierr);
      hermitian_tolerance = value;
    }
    
    
    //command line output 
    ierr = PetscOptionsHasName(NULL,NULL,"-long_output",&flg); CHKERRQ(ierr);
    longout	= flg;
    
    ierr = PetscOptionsHasName(NULL,NULL,"-prop_output",&flg); CHKERRQ(ierr);
    propout	= flg;
    
    ierr = PetscOptionsHasName(NULL,NULL,"-liouville_output",&flg); CHKERRQ(ierr);
    liouout	= flg;
    
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PQSPSetupKSP"

/**
 * @brief	Intitializes the index, the density matrix and all needed Liouvillian matrices. Must be called after all the AddDimension  has happened. Dimensions that are added afterwards will not be included and may cause serious errors.
 * 
 * @param	dm	the vector for the density matrix
 * @param	b	the right hand side vector for the ksp steady state solver
 * @param	AA	the Liouvillian matrix for the steady state solver, there is only one, since no time dependencies are allowed.
 * 
 */

PetscErrorCode System::PQSPSetupKSP(Vec * dm, Vec * b, Mat * AA)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    PetscInt		start,end;
    
    ierr = PQSPSetupIndex(); CHKERRQ(ierr);

    ierr = PQSPCreateVec(dm,&start,&end);
    ierr = PQSPCreateVecPlus1(b,&start,&end);			//create extra space for the trace condition
    
    index->SetParallelLayout(start,end);
    
    ierr = PQSPSetupMatPlus1Row(AA); CHKERRQ(ierr); 	//create extra space for the trace condition
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PQSPSetupIndex"

/**
 * @brief	Intitializes the index. Must be called after all the AddDimension  has happened. Dimensions that are added afterwards will not be included and maybe will cause serious errors.
 * 
 */

PetscErrorCode System::PQSPSetupIndex()
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    if( num_dims )			//if there is at least one dimension and the parallel layout has been determined
    {
        PetscInt		*mlsdim_pol;
        PetscInt		*mlsdimlengths;
        PetscInt		*modedimlenghts;
      
        ierr = ExtractMLSDimPol(&mlsdim_pol); CHKERRQ(ierr);
        ierr = ExtractMLSDimLengths(&mlsdimlengths); CHKERRQ(ierr);
        ierr = ExtractModeDimLengths(&modedimlenghts); CHKERRQ(ierr);
      
        if(!useMulti)       //single type of MLS
        {
            Index	*locindex   = new Index (num_mlsdens+1,num_mlsdims,mlsdim_pol,N_MLS[0],mlsdimlengths,num_modes,modedimlenghts);
            index		        = locindex;
        }
        else                //multiple types of mls
        {
            Index    *locindex   = new Index (num_mlsdens+1,num_mlsdims,mlsdim_pol,N_MLS,mlsdimlengths,N_D_MLS,multiMLS_start,num_modes,modedimlenghts);
            index                = locindex;
        }
        
        PetscBool		flg;
        ierr = PetscOptionsHasName(NULL,NULL,"-index_output",&flg); CHKERRQ(ierr);
        if(flg)
        {
            index->PrintGenInfos();
        }
      
        delete[]	mlsdim_pol;
        delete[]	mlsdimlengths;
        delete[]	modedimlenghts;
    }
    else				//otherwise its crap
    {
        (*PetscErrorPrintf)("Index initialization requires at least one dimension and a parallel layout!\n");
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PQSPCreateVec"

/**
 * @brief	intitializes the parallel density matrix vector and specifies the local dimensions
 * 
 * @param	dm	the vector which is used to store the density matrix.
 * 
 */

PetscErrorCode System::PQSPCreateVec(Vec * dm, PetscInt *start, PetscInt *end)
{
    PetscFunctionBeginUser;
    
    PetscInt		low,high;
    PetscErrorCode	ierr;


    if(num_dims)
    {
      ierr = VecCreate(PETSC_COMM_WORLD,dm); CHKERRQ(ierr);			//create vector
      ierr = VecSetSizes(*dm,PETSC_DECIDE,index->TotalDOF()); CHKERRQ(ierr);	//let petsc decide on the parallel layout
      ierr = VecSetFromOptions(*dm); CHKERRQ(ierr);				//


      ierr = VecGetOwnershipRange(*dm,&low,&high); CHKERRQ(ierr);		//get the local boundaries
      
      if( start && end )
      {
	*start		= low;							//combined index boundaries
	*end		= high;
      }
      
      loc_size		= high-low;
      
//       parallel_layout	= 1;							//layout has been determined
    }
    else
    {
      (*PetscErrorPrintf)("Can't create vector before system dimensions have been set!\n");
      (*PetscErrorPrintf)("Use e.g. AddMLSDimension(...) or AddMode(...) etc!\n");
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PQSPCreateVecPlus1"

/**
 * @brief	Intitializes a parallel vector of the dimension of the density matrix plus one.
 * 
 * @param	dm	the vector which is used to store the density matrix.
 * 
 */

PetscErrorCode System::PQSPCreateVecPlus1(Vec * dm, PetscInt *start, PetscInt *end)
{
    PetscFunctionBeginUser;
    
    PetscInt		low,high;
    PetscErrorCode	ierr;


    if(num_dims)
    {
      ierr = VecCreate(PETSC_COMM_WORLD,dm); CHKERRQ(ierr);			//create vector
      ierr = VecSetSizes(*dm,PETSC_DECIDE,index->TotalDOF()+1); CHKERRQ(ierr);	//let petsc decide on the parallel layout, one extra element for the trace condition
      ierr = VecSetFromOptions(*dm); CHKERRQ(ierr);				//

      ierr = VecGetOwnershipRange(*dm,&low,&high); CHKERRQ(ierr);		//get the local boundaries

      if( start && end )
      {
	*start		= low;							//combined index boundaries
	*end		= high;
      }
      
      loc_size		= high-low;
      
//       parallel_layout	= 1;							//layout has been determined
    }
    else
    {
      (*PetscErrorPrintf)("Can't setup density matrix before dimensions have been set!\n");
      (*PetscErrorPrintf)("Use e.g. AddMLSDimension(...) or AddMode(...)\n");
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PQSPCreateMat"

/**
 * @brief	intitializes the matrix for Liouvillian
 * 
 * @param	AA	the matrix for storing the Liouvillian.
 * 
 */

PetscErrorCode System::PQSPCreateMat(Mat * AA)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;

    if(num_dims)
    {
      ierr = MatCreate(PETSC_COMM_WORLD,AA); CHKERRQ(ierr);
      ierr = MatSetSizes(*AA,PETSC_DECIDE,PETSC_DECIDE,index->TotalDOF(),index->TotalDOF()); CHKERRQ(ierr);
      ierr = MatSetFromOptions(*AA); CHKERRQ(ierr);
      ierr = MatSetUp(*AA); CHKERRQ(ierr);
    }
    else
    {
      (*PetscErrorPrintf)("Can't setup Liouvillian matrix before dimensions have been set!\n");
      (*PetscErrorPrintf)("Use e.g. AddMLSDimension(...) or AddMode(...)\n");
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PQSPSetupMatPlus1Row"

/**
 * @brief	Intitializes a parallel matrix of the dimensions of the Liouville superoperator plus an extra row. This results in an rectangular matrix.
 * 
 * @param	AA	the matrix for storing the Liouvillian.
 * 
 */

PetscErrorCode System::PQSPSetupMatPlus1Row(Mat * AA)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;

    if(num_dims)
    {
      ierr = MatCreate(PETSC_COMM_WORLD,AA); CHKERRQ(ierr);
      ierr = MatSetSizes(*AA,PETSC_DECIDE,PETSC_DECIDE,index->TotalDOF()+1,index->TotalDOF()); CHKERRQ(ierr);
      ierr = MatSetFromOptions(*AA); CHKERRQ(ierr);
      ierr = MatSetUp(*AA); CHKERRQ(ierr);
    }
    else
    {
      (*PetscErrorPrintf)("Can't setup Liouvillian matrix before dimensions have been set!\n");
      (*PetscErrorPrintf)("Use e.g. AddMLSDimension(...) or AddMode(...)\n");
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
    }
    
    PetscFunctionReturn(0);
}




//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  Write certain values into a vector
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

#undef __FUNCT__
#define __FUNCT__ "DMWriteDiagThermal"

/**
 * @brief	Write the diagonal entries of the density matrix with a thermal distribution. Temperature and energies need to be specified in the System class.
 * 		NOTE: Normalization is slightly off, not dramatically, c.f. MLSPartitionFunction()
 * 
 * @param	dm	the density matrix.
 * @param	beta	the beta factor 1/(k_B T)
 * 
 */

PetscErrorCode System::DMWriteDiagThermal(Vec dm, PetscReal beta)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
  //-----------------------------------------------------------------
  //local vector entries
  //-----------------------------------------------------------------
    PetscInt	locindex,i,flg = 1;
    PetscScalar	value=0;
    PetscReal	part = 0.0,energy=0.0;				//default values produce crap, so the user should be able to see that there is sth wrong here
    
    ierr = MLSPartitionFunction(beta,&part); CHKERRQ(ierr);
    
    locindex = index->InitializeLocal();
    
    while ( index->ContinueLocal() )				//loop over all local rows
    {
      if( !index->IsPol() )
      {
	value	= 1;
	
	for (i=0; i < num_mlsdims; i++)
	{
	  ierr = IsMLSDimPol(i,&flg); CHKERRQ(ierr);
	  if( !flg )
	  {
	    ierr = Energies(i,&energy); CHKERRQ(ierr);
	    value *= PetscExpReal(-beta*((PetscReal) index->MLSQN(i))*energy);	//for all polarization dims the factor equals 1
	  }
	}
	
	value /= part;						//divide by mls partition function
	
	i = num_mlsdims;
	while( i < num_dims )
	{
	  ierr = Energies(i,&energy); CHKERRQ(ierr);
	  value *= (1-PetscExpReal(-energy*beta))*PetscExpReal(-beta*((PetscReal) index->ModeQN(i))*energy);
	  i += 2;
	}
	
	ierr	= VecSetValues(dm,1,&locindex,&value,INSERT_VALUES); CHKERRQ(ierr);
      }
      
      locindex = index->Increment();
    }
    
    
  //-----------------------------------------------------------------
  //assembly
  //-----------------------------------------------------------------
    ierr = VecAssemblyBegin(dm);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(dm);CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,"denstity matrix intitialized with thermal values.\n beta: %e K\n",(double) beta); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMWriteUniformDistribution"

/**
 * @brief	Write the diagonal entries of the density matrix with an uniform distribution into the dm diagonals.
 * 
 * @param	dm	the density matrix.
 * 
 */


PetscErrorCode System::DMWriteUniformDistribution(Vec dm)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
  //-----------------------------------------------------------------
  //local vector entries
  //-----------------------------------------------------------------
    PetscInt	locindex,count=0;
    PetscScalar	value;
    
    locindex = index->InitializeGlobal();
    
    while ( index->ContinueGlobal() )						//loop over all global rows
    {
      if( !index->IsPol() )
      {
	count++;								//total number of densities in the density matrix
      }
      
      locindex = index->Increment();
    }
    
    locindex = index->InitializeLocal();
    
    while ( index->ContinueLocal() )						//loop over all local rows
    {
      if( !index->IsPol() )
      {
	value	= 1.0/((PetscReal) count);
	ierr	= VecSetValue(dm,locindex,value,INSERT_VALUES); CHKERRQ(ierr);
      }
      
      locindex = index->Increment();
    }
    
  //-----------------------------------------------------------------
  //assembly
  //-----------------------------------------------------------------
    ierr = VecAssemblyBegin(dm);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(dm);CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,"denstity matrix intitialized with uniform distribution.\n"); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


/**
 * @brief	Computes the partition function of a set of arbitrary multi-level systems with an arbitrary number of bosonic modes. Needed for thermal dm initialization. Return type is PetscReal for increased accuracy.
 * 		
 * 		NOTE:	Accuracy of float128 seems to be too low in order to correctly compute this for optical frequencies at room temperature. 
 * 			Result in this case is just 1.0, this leads to a slightly wrong normalization of the density matrix at the beginning at the order of 10^-23, which is ok, but not super nice.
 * 			Could change that by somehow bypassing the calculation of something like Z = 1 + 10^-23, which is not possible with the used accuracy.
 * 			Other possibility is to use a arbitrary precision package like "GNU Multiple Precision Arithmetic Library"
 * 
 * @param	beta	the beta factor 1/(k_B T)
 * @param	ret	the value of the partition function
 * 
 */

PetscErrorCode System::MLSPartitionFunction(PetscReal beta, PetscReal *ret)
{   
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    PetscReal		summand, part = 0, energy = 0.0;
    PetscInt		i,flg = 1;
    
    index->InitializeGlobal();		//set the index to the global beginning aka ground state
    
    while ( index->ContinueMLS() )	//for all mls dofs
    {
      if( !index->IsPol() )		//is it density like?
      {
          summand = 1.0;
	
          for(i=0; i < num_mlsdims; i++)
          {
              ierr = IsMLSDimPol(i,&flg); CHKERRQ(ierr);
              if( !flg )
              {
                  ierr = Energies(i,&energy); CHKERRQ(ierr);
                  summand *= PetscExpReal(-beta*((PetscReal) index->MLSQN(i))*energy);	//factor equals one for all polarization like dofs.
              }
          }
	
          part += summand;
      }
      
      index->Increment();		//next element
    }
    
    *ret = part;
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMWriteGroundState"

/**
 * @brief	Initialize the dm in the ground state. Assumes that the entry P[0,0,0,0,...] is the ground state.
 * 
 * @param	dm	the density matrix.
 * 
 */

PetscErrorCode System::DMWriteGroundState(Vec dm)
{  
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    ierr = VecSet(dm,0.0); CHKERRQ(ierr);
    
    if ( index->LocStart() == 0 )
    {
      ierr = VecSetValue(dm,0,1.0,INSERT_VALUES); CHKERRQ(ierr);
    }
    
    ierr = VecAssemblyBegin(dm);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(dm);CHKERRQ(ierr);
      
    ierr = PetscPrintf(PETSC_COMM_WORLD,"denstity matrix intitialized in the ground state.\n\n"); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);

}


#undef __FUNCT__
#define __FUNCT__ "DMWritePureState"

/**
 * @brief	Initialize the dm in a pure state. Returns an error message if the input corresponds to a polarization like density matrix element.
 * 
 * @param	dm		the density matrix.
 * @param	indices		the multiindex containing the quantum numbers of the desired state.
 * 
 */

PetscErrorCode System::DMWritePureState(Vec dm, PetscInt * indices)
{  
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    ierr = PetscPrintf(PETSC_COMM_WORLD,"DMWritePureState: pure state density matrix intitialization:\n"); CHKERRQ(ierr);
    
    index->SetIndexFromQN(indices);							//Initialize the index to the given input   
    
    if( !index->IsPol() )								//if the index corresponds to a density like dm entry everything is fine
    {
      if ( index->IsLocal() )								//if the index is local we may set the density matrix
      {
	ierr = VecSetValue(dm,index->DMIndex(),1.0,INSERT_VALUES); CHKERRQ(ierr);	//write a 1 into the corresponding slot
      }
    
      ierr = VecAssemblyBegin(dm);CHKERRQ(ierr);					//assemble the parallel vector
      ierr = VecAssemblyEnd(dm);CHKERRQ(ierr);						//
      
      ierr = PetscPrintf(PETSC_COMM_WORLD,"State: "); CHKERRQ(ierr);
      ierr = index->PrintIndices(); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\n"); CHKERRQ(ierr);
    }
    else										//if the input corresponds to a polarization like input we have a problem
    {
      (*PetscErrorPrintf)("Seems like DMWritePureState input refers to a polarization dof!\n");
      (*PetscErrorPrintf)("Density matrix can only be initialized in a phyiscial state, i.e. a diagonal entry!\n");
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecIsHermitian"

/**
 * @brief	Checks whether the vector/density matrix is hermitian.
 * 
 * @param	dm		the density matrix
 * @param	flg		the answer, 0 for no, 1 for yes.
 * 
 */

PetscErrorCode System::VecIsHermitian(Vec dm, PetscInt *flg)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nDMIsHermitian:\t"); CHKERRQ(ierr);
    
    //if the dm is hermitian then the sum over all (and all nondiagonal) entries is real...
    PetscInt		locindex;
    const PetscScalar	*a;
    PetscScalar		local=0, global=0;
    
    ierr = VecGetArrayRead(dm,&a);CHKERRQ(ierr);
    
    locindex	= index->InitializeLocal();
    
    while( index->ContinueLocal() )
    {
      if( index->IsPol() )
      {
	local	+= a[locindex - index->LocStart()];
      }
      
      locindex	= index->Increment();
    }
    
    ierr = VecRestoreArrayRead(dm,&a);CHKERRQ(ierr);
    
    MPI_Allreduce(&local,&global,1,MPIU_SCALAR,MPIU_SUM,PETSC_COMM_WORLD);	//add all the local subsums together
    
    ierr = PetscPrintf(PETSC_COMM_WORLD,"imag = %e\n\n",(double) PetscImaginaryPart(global)); CHKERRQ(ierr);
    if ( fabs((double) PetscImaginaryPart(global)) > hermitian_tolerance )	*flg = 0;	//is not hermitian
    else									*flg = 1;	//is hermitian
    
    PetscFunctionReturn(0);
}



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  Param routines
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

#undef __FUNCT__
#define __FUNCT__ "AddParam"

/**
 * @brief	Add a parameter to the internal list. So far only for the header in the output files. Makes it more traceable.
 * 
 * @param	pname		the name of the parameter
 * @param	pvalue		the value of the parameter
 * 
 */

PetscErrorCode System::AddParam(std::string pname, PetscReal pvalue)
{
    PetscFunctionBeginUser;
    
    paramname[numparams]	= pname;
    params[numparams]		= pvalue;
    
    numparams++;
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "GetParam"

/**
 * @brief	Retrieve a parameter value by name.
 * 
 * @param	pname		the name of the parameter
 * @param	pvalue		the value of the parameter
 * 
 */

PetscErrorCode System::GetParam(std::string pname, PetscReal *pvalue)
{
    PetscFunctionBeginUser;
    
    PetscInt	i;
    
    for(i=0; i < numparams; i++)
    {
      if( !paramname[i].compare(pname) ) *pvalue = params[i];
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "UpdateParam"

/**
 * @brief	Update the value of a parameter by name.
 * 
 * @param	pname		the name of the parameter
 * @param	pvalue		the value of the parameter
 * 
 */

PetscErrorCode System::UpdateParam(std::string pname, PetscReal pvalue)
{
    PetscFunctionBeginUser;
    
    PetscInt	i;
    
    for(i=0; i < numparams; i++)
    {
      if( !paramname[i].compare(pname) ) params[i] = pvalue;
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "UpdateParam"

/**
 * @brief	Update the value of the real value tolerance cutoff.
 * 
 * @param	value		the new tolerance value
 * 
 */

PetscErrorCode System::SetRealValueTolerance(PetscReal value)
{
    PetscFunctionBeginUser;
    
    real_value_tolerance = value;
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "UpdateParam"

/**
 * @brief	Update the value of the hermitianity tolerance cutoff
 * 
 * @param	value		the new tolerance value
 * 
 */

PetscErrorCode System::SetHermitianTolerance(PetscReal value)
{
    PetscFunctionBeginUser;
    
    hermitian_tolerance = value;
    
    PetscFunctionReturn(0);
}


//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  Usefull 
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

/**
 * @brief	factorial. Return type is PetscReal, should make it more reliable for larger numbers.
 * 
 * @param	n	for n!
 * 
 */

#undef __FUNCT__
#define __FUNCT__ "Factorial"

PetscReal System::Factorial(PetscInt n)
{
    PetscInt	i;
    PetscReal	ret=1;
    
    if( n >= 0 )
    {
      for(i=1; i<= n; i++)
      {
	ret*=i;
      }
    }
    else
    {
      ret = 0;
    }
    
    return	ret;
}


/**
 * @brief	n!/(n-m)! = n*(n-1)...(n-m+1). Return type is PetscReal, should make it more reliable for larger numbers.
 * 
 * @param	n	for n!
 * @param	m	for (n-m)!
 * 
 */

#undef __FUNCT__
#define __FUNCT__ "FactorialTrunc"

PetscReal System::FactorialTrunc(PetscInt n, PetscInt m)
{
    PetscInt	i;
    PetscReal	ret=1;
    
    if( n >= m && m >= 0 )
    {
      for(i=n; i > (n-m); i--)
      {
	ret*=i;
      }
    }
    else
    {
      ret = 0;
    }
    
    return	ret;
}


/**
 * @brief	Returns the name of the nth dimension as a c++ string.
 * 
 * @param	n	the number of the dimension
 * 
 */

#undef __FUNCT__
#define __FUNCT__ "DimName"

std::string System::DimName(PetscInt n)
{
    std::list<Dim*>::iterator it=dimensions.begin();
    
    PetscInt	i;
    for (i = 0; i < n; i++)
    {
      it++;
    }
    
    Dim	*ret = (Dim*) *it;
    
    return	ret->ToString();
}


/**
 * @brief    Checks wether the two pointers belong to the same class and if they belong to the same mls type. If not this function returns specific error messages. Also returns the type number.
 *
 * @param    ptr1   the first pointer
 * @param    ptr2   the second pointer
 *
 */

#undef __FUNCT__
#define __FUNCT__ "SameType"

PetscErrorCode System::SameType(MLSDim *Ptr1, MLSDim *Ptr2, PetscInt * type)
{
    PetscFunctionBeginUser;
    
    PetscInt        ret  = 0;
    MultiMLSDim    *ptr1 = dynamic_cast<MultiMLSDim*> (Ptr1);
    MultiMLSDim    *ptr2 = dynamic_cast<MultiMLSDim*> (Ptr2);
    
    if( (ptr1 && !ptr2) || (!ptr1 && ptr2) )            //xor, means the two belong to a different class
    {
        (*PetscErrorPrintf)("Error: Mixed use of MLSDim and MultiMLSDim objects. Use either MLSDim or MultiMLSDim!\n");
        (*PetscErrorPrintf)("The two instances are: %s and %s\n",ptr1->ToString().c_str(),ptr2->ToString().c_str());
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
    }
    else if( ptr1 && ptr2 )
    {
        if( !(ptr1->SameType(ptr2)) )
        {
            (*PetscErrorPrintf)("Error: Mixed use of different MLS type objects! Use Matrix-Multiplication to create Liouvillians that couple different MLS types!\n");
            (*PetscErrorPrintf)("The two instances are: %s and %s\n",ptr1->ToString().c_str(),ptr2->ToString().c_str());
            SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
        }
        else    ret = ptr1->mlsTypeNumber;
    }
    
    *type = ret;
    
    PetscFunctionReturn(0);
}



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  Elem 
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

/**
 * @brief	Constructor of the Elem class, which is a (very) lightweight version of the Index class, just contains the  needed for the recursive operator action stuff.
 * 		    Shouldn't do anything with the mode dofs, but they are included either way, in order to not to have to implement even more specialized  in the Index class.
 * 
 * @param	index		the Index object.
 * @param	inorder		the order of the normally ordered expectation value to be computed.
 * @param   type        the type of mls
 * 
 */

Elem::Elem(Index * index,PetscInt inorder, PetscInt type)
{
    PetscInt	i;
    
    indices	        = new PetscInt [index->NumDims()];
    for(i=0; i < index->NumDims(); i++)	indices[i] = index->Indices(i);
    
    order	        = inorder;
    length	        = index->NumDims();
    mlslength	    = index->MLSDims();
    mlsTypeNumber   = type;
    NMLS	        = index->NMls(type);
    factor  	    = 1;
    opactions	    = 0;
}


/**
 * @brief	Them copy constructor.
 * 
 */

Elem::Elem(const Elem& source)
{
    PetscInt	i;
    
    indices	        = new PetscInt [source.length];
    for(i=0; i < source.length; i++ )	indices[i] = source.indices[i];
    
    order	        = source.order;
    length	        = source.length;
    mlslength	    = source.mlslength;
    NMLS	        = source.NMLS;
    factor	        = source.factor;
    opactions	    = source.opactions;
    mlsTypeNumber   = source.mlsTypeNumber;
}


/**
 * @brief	Them destructor.
 * 
 */

Elem::~Elem()
{
    delete[] indices;
}


/**
 * @brief	Provides a compare function for the list sort algorithm.
 * 
 * @param	first		the first element of the comparison
 * @param	second		the second element of the comparison
 * 
 */

bool Elem::ElemComp(const Elem* first, const Elem* second)
{
    return	(first->dmindex < second->dmindex);
}


#undef __FUNCT__
#define __FUNCT__ "ComputeIndex"

/**
 * @brief	Computes the dmindex of the Elems in the list. Needed for comparision with the local dm elements.
 * 
 * @param	sys		the System specification object.
 * @param	clean		the list of Elems.
 * 
 */

PetscErrorCode Elem::ComputeIndex(System * sys, std::list< Elem* >* clean)
{
    PetscFunctionBeginUser;
    
    for(std::list<Elem*>::iterator it = clean->begin(); it != clean->end(); ++it)
    {
      (*it)->dmindex = sys->index->SetIndices((*it)->indices);
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PrintIndices"

/**
 * @brief	Prints the indices of the element into stdout.
 */

PetscErrorCode Elem::PrintIndices()
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    PetscInt		i;
    
    ierr = PetscPrintf(PETSC_COMM_WORLD,"indices:  "); CHKERRQ(ierr);
    for(i=0; i < mlslength; i++) ierr = PetscPrintf(PETSC_COMM_WORLD,"%d ",indices[i]); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\t"); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "CombineListElems"

/**
 * @brief	Takes the raw output form the recursive SingleElementMLSNE() call. In this output the same density matrix elements may appear multiple times. This function adds all the factors of the matching elements together and outputs a clean list, where
 * 		each element appears only once with the correct prefactor.
 * 
 * @param	clean		the output list.
 * @param	raw		the input list.
 * 
 */

PetscErrorCode Elem::CombineListElems(std::list< Elem* >* clean, std::list< Elem* >* raw)
{
    PetscFunctionBeginUser;

    PetscInt	match;										//check parameter    
    Elem	*check;										//Elem pointer
    
    while(!raw->empty())									//either raw input is empty or everything is done
    {
      check	= raw->front();									//get the first element of the list

      if(clean->empty())									//nothing in the clean list, i.e. first loop
      {
	clean->push_back(check);								//just add the element to the list and continue
      }
      else											//if there is already something in the clean list (i.e. not the first loop)
      {
	match = 0;										//set control to zero
	for(std::list<Elem*>::iterator it = clean->begin(); it != clean->end(); ++it)		//for every element in clean
	{
	  if( (*it)->EqualIndices(*check) )							//if they have matching indices
	  {
	    (*it)->factor	+= check->factor;						//add the factors together
	    delete		check;								//then delete the element
	    match++;										//increase control by one
	    break;
	  }
	}
	if( match == 0 )									//no matching elements in clean
	{
	  clean->push_back(check);								//so we put it into the list
	}
      }
      
      raw->pop_front();										//and delete it from the list (I think it calls destructor, so it has to be at the end of each 
    }
    
    PetscFunctionReturn(0);
}



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  Vector initialization routines
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

#undef __FUNCT__
#define __FUNCT__ "VecMLSGroundStateModeTraceout"

/**
 * @brief	Write a vector |a>> that represents the mls ground state projector while tracing over the mode dofs
 * 
 * @param	a		the vector.
 * 
 */

PetscErrorCode System::VecMLSGroundStateModeTraceout(Vec a)
{  
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    PetscInt		locindex;
    
    ierr = VecSet(a,0.0); CHKERRQ(ierr);
    
    locindex	= index->InitializeLocal();
    
    while( index->ContinueLocal() )
    {
      if( index->IsMLSGroundState() && index->IsModeDensity() )
      {
	ierr = VecSetValue(a,locindex,1.0,INSERT_VALUES); CHKERRQ(ierr);
      }
      
      locindex	= index->Increment();
    }
    
    ierr = VecAssemblyBegin(a);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(a);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecTrace"

/**
 * @brief	Writes the trace operation into the vector i.e. writes the entries of a in a way that  < a|rho > = tr[rho] = 1 (if rho is a valid density matrix) 
 * 
 * @param	a		the trace operator
 * 
 */

PetscErrorCode System::VecTrace(Vec a)
{  
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    PetscInt		locindex;
    
    ierr = VecSet(a,0.0); CHKERRQ(ierr);
    
    locindex	= index->InitializeLocal();
    
    while( index->ContinueLocal() )
    {
      if( !index->IsPol() )
      {
	ierr = VecSetValue(a,locindex,1.0,INSERT_VALUES); CHKERRQ(ierr);
      }
      
      locindex	= index->Increment();
    }
    
    ierr = VecAssemblyBegin(a);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(a);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecContractReal"

/**
 * @brief	Writes all nonzero real elements into the indices and factors arrays, while allocating the storage for them. Useful for saving memory when the vectors conatain mainly zeros.
 * 		Writes the local vector indices! sth. like a sparse vector format...
 * 
 * @param	a		the vector.
 * @param	num		pointer to the number of nonzeros.
 * @param	indices		pointer to the array that stores the local indices of the nonzero elements
 * @param	factors		pointer to the array that stores the acutal nonzero elements
 * 
 */

PetscErrorCode System::VecContractReal(Vec a, PetscInt *num, PetscInt ** indices, PetscReal ** factors)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    const PetscScalar	*entries;
    PetscInt		i,count=0;
    
    ierr = VecGetArrayRead(a,&entries); CHKERRQ(ierr);
    
    for(i = 0; i < loc_size; i++)
    {
      if( PetscRealPart(entries[i]) != 0.0 )	count++;
    }
    
    *num	= count;
    *indices	= new PetscInt  [count];
    *factors	= new PetscReal [count];
    
    count=0;
    for(i = 0; i < loc_size; i++)
    {
      if( PetscRealPart(entries[i]) != 0.0 )
      {
	(*indices)[count]	= i;
	(*factors)[count++]	= PetscRealPart(entries[i]);
      }
    }
    
    ierr = VecRestoreArrayRead(a,&entries); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecContractScalar"

/**
 * @brief	Writes all nonzero complex elements into the indices and factors arrays, while allocating the storage for them. Useful for saving memory when the vectors conatain mainly zeros.
 * 		sth. like a sparse vector format...
 * 
 * @param	a		the vector.
 * @param	num		pointer to the number of nonzeros.
 * @param	indices		pointer to the array that stores the indices of the nonzero elements
 * @param	factors		pointer to the array that stores the acutal nonzero elements
 * 
 */

PetscErrorCode System::VecContractScalar(Vec a, PetscInt *num, PetscInt ** indices, PetscScalar ** factors)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    const PetscScalar	*entries;
    PetscInt		i,count=0;
    
    ierr = VecGetArrayRead(a,&entries); CHKERRQ(ierr);
    
    for(i = 0; i < loc_size; i++)
    {
      if( PetscAbsScalar(entries[i]) > 0.0 )	count++;
    }
    
    *num	= count;
    *indices	= new PetscInt    [count];
    *factors	= new PetscScalar [count];
    
    count=0;
    for(i = 0; i < loc_size; i++)
    {
      if( PetscAbsScalar(entries[i]) > 0.0 )
      {
	(*indices)[count]	= i;
	(*factors)[count++]	= entries[i];
      }
    }
    
    ierr = VecRestoreArrayRead(a,&entries); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

