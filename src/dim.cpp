
/**
 * @file	dim.cpp
 * 
 * 		This contains all the memeber function definitions of the Dim class family
 * 
 * @author	Michael Gegg
 * 
 */

#include"../include/dim.hpp"

//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  MLSDim member functions
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

/**
 * @brief	dimension constructor. Sets all aspects of the dimension. Needed for the initial setup of the dimensionality of the considered problem.
 * 
 */

MLSDim::MLSDim(PetscInt left, PetscInt right, PetscInt polflag, PetscInt inlength, PetscReal inenergy)
{
    ket		        = left;
    bra		        = right;
    ispol	        = polflag;
    dimlength	    = inlength;
    energy	        = inenergy;
    mlsTypeNumber   = 0;
    
    if( ket != 0 || bra != 0 )	n00	= 0;
    else			n00	= 1;		// the n00 case
}


/**
 * @brief	dimension constructor using a name. Sets all aspects of the dimension. Needed for the initial setup of the dimensionality of the considered problem.
 * 
 */

MLSDim::MLSDim(const MLSDim& dim, PetscInt polflag, PetscInt indimlength, PetscReal inenergy)
{
    ket		        = dim.ket;
    bra		        = dim.bra;
    ispol	        = polflag;
    dimlength	    = indimlength;
    energy	        = inenergy;
    mlsTypeNumber   = 0;
    
    if( ket != 0 || bra != 0 )	n00	= 0;
    else			n00	= 1;		// the n00 case
}


/**
 * @brief	Name constructor. Just sets the name part of the MLSDim. Needed for identifying existing dimensions, setting physical vector entries and Liouville operators. c.f. the AddLiouvillian() and Observables::SetupXXX functions.
 * 
 */

MLSDim::MLSDim(PetscInt left, PetscInt right)
{
    ket	            = left;
    bra	            = right;
    mlsTypeNumber   = 0;
    
    if( ket != 0 || bra != 0 )	n00	= 0;
    else			n00	= 1;		// the n00 case
}


/**
 * @brief	Swap "constructor". Creates a name object that has swapped ket and bra values.
 * 
 */

MLSDim MLSDim::Swap(MLSDim swap)
{
    MLSDim	ret (swap.bra,swap.ket);
    return	ret;
}

/**
 * @brief	Density constructor, creates a density dimenstion corresponding to the first or second index of the input name
 * 
 */

MLSDim::MLSDim(PetscInt which, const MLSDim &name)
{
    if( !which )
    {
      ket		= name.ket;
      bra		= name.ket;
    }
    else
    {
      ket		= name.bra;
      bra		= name.bra;
    }
    
    mlsTypeNumber   = 0;
    
    if( ket || bra )	n00 = 0;
    else		        n00 = 1;
}


/**
 * @brief	Density constructor, creates a density name corresponding to the first or second index of the input name
 * 
 */

MLSDim::MLSDim(const MLSDim& ketdim, const MLSDim& bradim)
{
    ket		        = ketdim.ket;
    bra		        = bradim.ket;
    mlsTypeNumber   = 0;
    
    if( ket == 0 && bra == 0 )	n00 = 1;
    else			n00 = 0;
}


/**
 * @brief	Simple compare function.
 * 
 */

PetscInt MLSDim::IsEqual(Dim * input)
{
    MLSDim	*ptr = dynamic_cast<MLSDim*> (input);
    
    if( ptr && ket == ptr->ket && bra == ptr->bra )	return 1;
    else						                    return 0;
}


/**
 * @brief	Checks whether the current MLSDim is a density degree of freedom.
 * 
 */

PetscInt MLSDim::IsDensity()
{
    if( ket == bra )	return	1;
    else		        return	0;
}


/**
 * @brief	Simple print name function. Prints the name of the MLS dimension into std_out.
 * 
 */

PetscErrorCode MLSDim::PrintName()
{
    PetscFunctionBeginUser;
  
    PetscErrorCode	ierr;
    
    ierr = PetscPrintf(PETSC_COMM_WORLD,"mls dimension: n(%d,%d)\n",ket,bra); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


/**
 * @brief	Create a string representation of the MLSDim, i.e. the "name"
 * 
 */

std::string MLSDim::ToString()
{
    return	"n"+std::to_string(ket)+","+std::to_string(bra);
}



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  MultiMLSDim member functions
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

/**
 * @brief    dimension constructor. Sets all aspects of the dimension. Needed for the initial setup of the dimensionality of the considered problem.
 *
 */

MultiMLSDim::MultiMLSDim(PetscInt left, PetscInt right, PetscInt polflag, PetscInt inlength, PetscReal inenergy, PetscInt typeNumber) :
MLSDim(left, right, polflag, inlength, inenergy)
{
    mlsTypeNumber = typeNumber;
}


/**
 * @brief    dimension constructor using a name. Sets all aspects of the dimension. Needed for the initial setup of the dimensionality of the considered problem.
 *
 */

MultiMLSDim::MultiMLSDim(const MultiMLSDim& dim, PetscInt polflag, PetscInt indimlength, PetscReal inenergy) :
MLSDim(dim.ket, dim.bra, polflag, indimlength, inenergy)
{
     mlsTypeNumber = dim.mlsTypeNumber;
}


/**
 * @brief    Name constructor. Just sets the name part of the MLSDim. Needed for identifying existing dimensions, setting physical vector entries and Liouville operators. c.f. the AddLiouvillian() and Observables::SetupXXX functions.
 *
 */

MultiMLSDim::MultiMLSDim(PetscInt left, PetscInt right, PetscInt typeNumber) : MLSDim(left,right)
{
    mlsTypeNumber = typeNumber;
}


/**
 * @brief    Swap "constructor". Creates a name object that has swapped ket and bra values.
 *
 */

MultiMLSDim MultiMLSDim::Swap(MultiMLSDim swap)
{
    MultiMLSDim     ret (swap.bra,swap.ket,swap.mlsTypeNumber);
    return          ret;
}

/**
 * @brief    Density constructor, creates a density dimenstion corresponding to the first or second index of the input name
 *
 */

MultiMLSDim::MultiMLSDim(PetscInt which, const MultiMLSDim &name) : MLSDim(which,name)
{
    mlsTypeNumber = name.mlsTypeNumber;
}


/**
 * @brief    Density constructor, creates a density name corresponding to the first or second index of the input name
 *
 */

MultiMLSDim::MultiMLSDim(const MultiMLSDim& ketdim, const MultiMLSDim& bradim) :
MLSDim(ketdim, bradim)
{
    mlsTypeNumber = ketdim.mlsTypeNumber;
}


/**
 * @brief    Simple compare function.
 *
 */

PetscInt MultiMLSDim::IsEqual(Dim * input)
{
    MultiMLSDim    *ptr = dynamic_cast<MultiMLSDim*> (input);
    
    PetscInt ret = 0;
    
    if(ptr)
    {
        if( ket == ptr->ket && bra == ptr->bra && mlsTypeNumber == ptr->mlsTypeNumber )    ret++;
    }
    
    return ret;
}


/**
 * @brief    Simple print name function. Prints the name of the MLS dimension into std_out.
 *
 */

PetscErrorCode MultiMLSDim::PrintName()
{
    PetscFunctionBeginUser;
    
    PetscErrorCode    ierr;
    
    ierr = PetscPrintf(PETSC_COMM_WORLD,"mls dimension: n(%d,%d), mls type: %d\n",ket,bra,mlsTypeNumber); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


/**
 * @brief    Create a string representation of the MLSDim, i.e. the "name"
 *
 */

std::string MultiMLSDim::ToString()
{
    return    "n"+std::to_string(ket)+","+std::to_string(bra)+"_"+std::to_string(mlsTypeNumber);
}


/**
 * @brief    Simple compare function.
 *
 */

PetscInt MultiMLSDim::SameType(Dim * input)
{
    MultiMLSDim    *ptr = dynamic_cast<MultiMLSDim*> (input);
    
    PetscInt ret = 0;
    if(ptr)
    {
        if( mlsTypeNumber == ptr->mlsTypeNumber )    ret++;
    }
    
    return ret;
}



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  ModeDims member functions
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

/**
 * @brief	Constructor. So simple that it does not need error handling.
 * 
 */

ModeDim::ModeDim(PetscInt choose, PetscInt n, PetscInt indimlenght, PetscReal inenergy)
{
    ket		= choose;
    number	= n;
    n00		= 0;
    dimlength	= indimlenght;
    
    if( !ket )	energy	= inenergy;		//only the ket dim of the mode carries the information about the elementary excitation energy
    
}


/**
 * @brief	Name constructor. Just sets the name part of the ModeDim. Needed for identifying the right dimensions for setting Liouvillians etc.
 * 
 */

ModeDim::ModeDim(PetscInt choose, PetscInt n)
{
    ket		= choose;
    number	= n;
    n00		= 0;
}


/**
 * @brief	Compares two ModeDim objects.
 * 
 */

PetscInt ModeDim::IsEqual(Dim* input)
{
    ModeDim	*ptr = dynamic_cast<ModeDim*> (input);
    
    if( ptr && number == ptr->number && ket == ptr->ket )	return	1;
    else							return	0;
}


/**
 * @brief	Print the name of the ModeDim into stdout.
 * 
 */

PetscErrorCode ModeDim::PrintName()
{
    PetscFunctionBeginUser;
  
    PetscErrorCode	ierr;
    
    if( ket )
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"individual mode dimension: |m%d>\n",number); CHKERRQ(ierr);
    }
    else
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"individual mode dimension: <m%d|\n",number); CHKERRQ(ierr);
    }
    
    PetscFunctionReturn(0);
}


/**
 * @brief	Print the name of the whole mode into stdout.
 * 
 */

PetscErrorCode ModeDim::PrintMode()
{
    PetscFunctionBeginUser;
  
    PetscErrorCode	ierr;

    ierr = PetscPrintf(PETSC_COMM_WORLD,"mode no: %d\n",number); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


/**
 * @brief	Create a string representation of the ModeDim, i.e. the "name"
 * 
 */

std::string ModeDim::ToString()
{
    if( ket )	return	"|m"+std::to_string(number)+">";
    else	return	"<m"+std::to_string(number)+"|";
}
