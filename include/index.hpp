
/**
 * @file	index.hpp
 * 
 * 		Definintion of the Index class. PsiQuaSP declares the density matrix in a vectorized form. The index keeps track of which element in the vector corresponds to which set of quantum numbers.
 * 		The loop functionality of the class is declared as inline functions for (hopefully) better performance, therefore these functions are also included here in the header.
 * 
 * @author	Michael Gegg
 * 
 */


#ifndef _Index
#define _Index

#include"system.hpp"
#include"dim.hpp"

/**
 * @brief	Index class. Manages all indexing routines of the many mls method including the modes. <br>
 * 
 * 		The indexing of the density matrix is managed in a one dimensional vector without any "dead" entries, meaning every entry in the Petsc vector corresponds to an actual, physical entry in the density matrix.
 * 		The actual working principle is hidden, because it is complicated. For most purposes the user will not need to work with this class directly.<br>
 * 		In a multiprocessor setup every processor has its own full copy of the index. Only the local start and endvalues of the vector index are stored: Petsc cuts parallel vectors into snippets of approximately the same size (at least using the default options).
 * 		The snippets are distributed across the processor and every processor only has the local snippet and not a copy of the entire vector (distributed memory). The start and end values of the index refer to the global index that just counts the 
 *		vector entries starting with zero at the first entry. So far this has not caused any storage overhead problems.<br>
 * 		There are almost no error checks in the utility functions, since it should be kept fast.
 * 
 * - The class provides functionalities for everything needed in an application:
 *   + increment capabilities: going the vector element by element requires a function that keeps track of the associated quantum numbers of the elements
 *   + logical statements like, can this dimension be incremented/decremented? has the local/global index reached the local/global end?
 *   + Pitches for setting up the matrix of the master equation, like letting a constructor act on a density matrix element results in a new element, which is pitch vector entries away
 *   + truncation of all occuring dimensions, mls and mode!
 *   + some read and write utilities for checking/debugging
 * 
 */


class Index
{
protected:
  
    //parameters
    PetscInt	**blocksizes;			//!< the sizes of all superblocks of the different dimensions
    PetscInt	*blocksizes_max;		//!< the number of different superblocks for each dimension, i.e. the lengths of the blocksizes[i] arrays
    PetscInt	*blockindices;			//!< the indices referring to the blocksizes arrays
    PetscInt	*indices;			    //!< the normal indices referring to the dimensions
    PetscInt	*mlsdim_maxvalues;		//!< the maximum value each mls dimension can have, important for mls offdiag truncation
    PetscInt	*mlsdim_pol;			//!< 0 for a density like mls dimension, 1 for a polarization
    PetscInt	dmindex;			    //!< the combinded running index, i.e. the index in the dm vector, starting with zero at the global beginning of the vector!
    PetscInt	mlsindex;			    //!< the running index of the current mls subblock, repeats for every mode dof...
    PetscInt	isend;				    //!< are we at the end of all superblocks?
    PetscInt	num_dims;			    //!< the number of different dimensions
    PetscInt	firstmodedim;			//!< the index of the first mode dimension in the arrays above
    
    PetscInt    N_D_MLS;                //!< the number of different mls
    PetscInt    *multiMLS_start;        //!< the index of the frist dimension of the mls kind in the indices array
    
    PetscInt	num_levels;			    //!< the number of levels per mls
    PetscInt	*mls_dof;			    //!< the mls total degrees of freedom
    PetscInt	*mode_dofs;			    //!< the dofs for each mode
    PetscInt	*NMLS;				    //!< the number of different mls
    PetscInt	total_dof;			    //!< total degrees of freedom
    
    PetscInt	loc_start;			    //!< the lower parallel boundary for Index::dmindex
    PetscInt	loc_end;			    //!< one more than the upper parallel boundary for Index::dmindex
    PetscInt	mls_start;			    //!< the parallel lower boundary for the mlsindex
    PetscInt	*loc_starters;			//!< the local start values for the indices
    PetscInt	*loc_blockstarters;		//!< the local start values for the blockindices
    
    PetscInt	*isfake;			    //!< is it a fake dimension or not
    PetscInt	**rule;				    //!< the rule for computing the value of the fake dimension form the other dimensions, only for fake dimensions, obviously
    
    
    //some helper functions
    PetscInt		BinomDim(PetscInt n, PetscInt order);		//Computes the recurring dimensionality binomial coefficient
    PetscInt		SetBlockZeros(PetscInt * dimlenghts);		//sets the blocksizes[0] entries to zero according to the dimlenghts specification array
    inline void		IncrementInternal();                        //!< increases the internal counters for the case of one type of mls
    inline void     IncrementInternalMultiMLS();                //!< increases the internal counters for the case of more than one type of mls
    
public:
  
    //constructors/destructors
    Index(PetscInt nlevels, PetscInt nummlsdims, PetscInt * mlspol, PetscInt N, PetscInt * dimlenghts, PetscInt modes, PetscInt *modedimlengths);
    Index(PetscInt nlevels, PetscInt nummlsdims, PetscInt * mlspol, PetscInt * N, PetscInt * dimlenghts, PetscInt n_d_mls, PetscInt * multimls_start, PetscInt modes, PetscInt *modedimlengths);
    ~Index();								//default destructor
    
    //set parallel layout
    void		SetParallelLayout(PetscInt start, PetscInt end);
  
    //the actual loop based functionalities, like initialize, increment, and so on
    inline PetscInt	InitializeLocal();				//set the Index to the local start values, needed for parallelization
    inline PetscInt	InitializeGlobal();				//set the Index to the global start values, needed for parallelization
    inline PetscInt	Increment();					//increase the running index by one and the blockindex accordingly
    inline PetscInt	ContinueLocal();				//true so long the index is within the local boundaries
    inline PetscInt	ContinueGlobal();				//true so long the index is within the global boundaries
    inline PetscInt ContinueMLS();                  //true so lone we reached the end of the first mls block of all types
    inline PetscInt	ContinueMLS(PetscInt i);		//true so lone we reached the end of the first mls block of type i
    
    
    inline PetscInt	MLSIPitch(PetscInt dim);			        //the pitch for increasing a single mls dimension
    inline PetscInt	MLSDPitch(PetscInt dim);    			    //the pitch for decreasing a single mls dimension
    inline PetscInt	MLSCPitch(PetscInt up, PetscInt down);		//pitch for increasing mls dim up by one and simultaneously decreasing mls down by one
    inline PetscInt	ModeIPitch(PetscInt modedim);			    //pitch for increasing the modedim dim by one
    inline PetscInt	ModeDPitch(PetscInt modedim);			    //pitch for decreasing the modedim dim by one
    
    inline PetscInt GetType(PetscInt i);                        //return the mls type number to a given dimension i
    inline PetscInt	OutOfBounds(PetscInt * index);			    //check whether a given index is allowed or not
    inline PetscInt	MLSOutOfBounds(PetscInt * index);		    //check whether a given mls index is allowed or not
    inline PetscInt	CanIncrement(PetscInt dim);			        //can I increment the dimension dim?
    inline PetscInt	CanDecrement(PetscInt dim);			        //can I decrement the dimension dim?
    inline PetscInt	IsLocal();					                //is the current index position within the local boundaries?
    inline PetscInt	IsLocal(PetscInt index);		        	//is the index parameter local?
    inline PetscInt	IsLocal(PetscInt *indices);		        	//are the indices local?
    
    inline PetscInt	IsPol();									                //is the current element a polarization or not?
    inline PetscInt	IsMLSDensity();								        	    //are the mls dofs density like?
    inline PetscInt	IsModeDensity();							        	    //are the mode dims density like?
    inline PetscInt	IsMLSOneDimFirstOffdiag(PetscInt dim);						//is the dim mls dimension a frist offdiag and the rest density like? as in <J_->
    inline PetscInt	IsMLSOneDimNumberOffdiag(PetscInt dim, PetscInt number);    //is the dim mls dimension a #number offdiag and the rest density like? as in <(J_-)^k>
    inline PetscInt	IsMLSTwoDimFirstOffdiag(PetscInt dim1,PetscInt dim2);		//are the mls dims frist offdiags for dim1 and dim2 and the rest density like?
    inline PetscInt	IsMLSTwoDimNumberOffdiag(PetscInt dim1, PetscInt dim2, PetscInt number);	//are the mls dims #number offdiags for dim1 and dim2 and the rest density like?
    inline PetscInt	IsModeOneDimFirstLeftOffdiag(PetscInt dim);					//is the dim mode dimension a first left offdiag and the rest density like? as in <b>
    inline PetscInt	IsModeOneDimFirstRightOffdiag(PetscInt dim);				//is the dim mode dimension a first right offdiag and the rest density like? as in <b^+>
    inline PetscInt	IsMLSGroundState();								            //is the mls index in the ground state
    
    inline PetscInt	MLSQN(PetscInt dim);				            //return the mls dimension quantum number, also for the n00 case
    inline PetscInt	ModeQN(PetscInt dim);				            //return the mode dimension quantum number
    inline PetscInt	MaxQN(PetscInt dim);				            //Return the maximum allowed quantum number for the dim dimension
    inline PetscInt	Indices(PetscInt dim);
    
    inline void		SetIndex(PetscInt index);			    //set the Index to a specific location via an index
    inline PetscInt	SetIndexLocal(PetscInt index);			//set the Index to a specific location via a local index, starting with zero at the first local element
    inline PetscInt	SetIndices(PetscInt *inindices);		//set the index to a specific location via inindices
    PetscErrorCode	SetIndexFromQN(PetscInt *inqns);		//set the index from an array that contains the actual desired quantum numbers 
    
    
    //I/O functionalities
    PetscErrorCode	PrintElements();				    //Print all elements into stdout
    PetscErrorCode	PrintDiagonals();				    //Print all diagonals into stdout
    PetscErrorCode	PrintBlockSizes();				    //Print the blocksizes, see whether its junk
    PetscErrorCode	PrintIndex();					    //Print the current dmindex value of the Index
    PetscErrorCode	PrintIndices();					    //Print the current indices values
    PetscErrorCode	PrintBlockIndices();				//Print the current block indices
    PetscErrorCode	PrintMLSDimMaxValues();				//Print the dim_maxvalues array into stdout
    PetscErrorCode	PrintGenInfos();
    
    PetscInt		LocStart()		        { return loc_start; }			//!< Return the local start value of the index
    PetscInt		LocEnd()		        { return loc_end; }	    		//!< Return the position of one past the local end of the index
    PetscInt		TotalDOF()		        { return total_dof; }			//!< Return the total number of density matrix elements
    PetscInt		NumDims()		        { return num_dims; }			//!< Return the total number of dimensions
    PetscInt		DMIndex()		        { return dmindex; }		    	//!< Return the dmindex
    PetscInt		MLSIndex()		        { return mlsindex; }			//!< Return the position in the current mls subblock
    PetscInt		MLSDims()		        { return firstmodedim; }		//!< Return the number of the first mode dimension, which is also the total number of mls dimensions
    PetscInt		NMls(PetscInt i)	    { return NMLS[i]; }			    //!< Return the number of mls
    PetscInt        MlsStart(PetscInt i)    { return multiMLS_start[i]; }   //!< Return the number of mls
};



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  inline pitch functions
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

/**
 * @brief	Calculate the pitch for the Index to increase the value of the dim dimension by one.
 * 
 * @param	dim	the dimension to be increased.
 * 
 */

inline PetscInt Index::MLSIPitch(PetscInt dim)
{
    PetscInt	ret = 1 , upper = 1;
    
    if( N_D_MLS == 1 )
    {
        if(dim)
        {
            PetscInt	i = dim-1,j,pitch = 0;
            
            upper = blocksizes[i][blockindices[i]];
            
            while(i)
            {
                for(j=0; j < upper; j++)
                {
                    pitch += blocksizes[i-1][blockindices[i-1]+j];
                }
                upper = pitch;
                pitch = 0;
                i--;
            }
        }
    }
    else
    {
        PetscInt type = GetType(dim);
        
        if( dim - multiMLS_start[type] )
        {
            PetscInt    i = dim-1,j,pitch = 0;
            
            upper = blocksizes[i][blockindices[i]];
            
            while( i > multiMLS_start[type] )                           //only the blocks inside the respective mls type
            {
                for(j=0; j < upper; j++)
                {
                    pitch += blocksizes[i-1][blockindices[i-1]+j];
                }
                upper = pitch;
                pitch = 0;
                i--;
            }
        }
        if(type) upper *= mls_dof[type-1];                              //multiply times the dofs of all preceding mls types
    }
    
    return	ret*upper;
}


/**
 * @brief	Calculate the pitch for the Index to decrease the value of the dim dimension by one.
 * 
 * @param	dim	the dimension to be increased.
 * 
 */

inline PetscInt Index::MLSDPitch(PetscInt dim)
{
    PetscInt	ret = 1 , upper = 1;
    
    if( N_D_MLS == 1 )
    {
        if(dim)
        {
            PetscInt	i = dim-1,j,pitch = 0;
            
            upper = blocksizes[i][blockindices[i]-1];
            
            while(i)
            {
                for(j=0; j < upper; j++)
                {
                    pitch += blocksizes[i-1][blockindices[i-1]+j-upper];
                }
                upper = pitch;
                pitch = 0;
                i--;
            }
        }
    }
    else
    {
        PetscInt type = GetType(dim);
        
        if( dim - multiMLS_start[type] )
        {
            PetscInt    i = dim-1,j,pitch = 0;
            
            upper = blocksizes[i][blockindices[i]-1];
            
            while( i > multiMLS_start[type] )
            {
                for(j=0; j < upper; j++)
                {
                    pitch += blocksizes[i-1][blockindices[i-1]+j-upper];
                }
                upper = pitch;
                pitch = 0;
                i--;
            }
        }
        if(type) upper *= mls_dof[type-1];                              //multiply times the dofs of all preceding mls types
    }
    
    return	-ret*upper;
}


/**
 * @brief	Calculate the mls pitch when the up dimension is increased by one while the down dimension is decreased by one.
            Since the up and down pitches depend on each other, this is not just the addition of MLSIPitch() and MLSDPitch(). Thereofore there is an extra function for this.
 * 
 * @param	down	the down dimension.
 * @param	up	    the up dimension.
 * 
 */

inline PetscInt Index::MLSCPitch(PetscInt down, PetscInt up)
{
    PetscInt	i,ret = 1 , upper = 1, upper2 = 1;
    
    PetscInt	newblockindices [firstmodedim];                                 //this is the difference: the increase pitch needs to be calculated from the position where the down pitch leads
    for(i=0; i < firstmodedim; i++)	newblockindices[i] = blockindices[i];       //to (or vice versa). However this function does not change the index position, it just calculates the pitch
    
    if( N_D_MLS == 1 )
    {
        if(down)		//start with downward pitch
        {
            i = down-1;
            PetscInt	j,pitch = 0;
            
            newblockindices[i]	= blockindices[i]-1;			                //this is said position, where the decrement pitch leads to
            upper			    = blocksizes[i][blockindices[i]-1];	            //
            
            while(i)
            {
                newblockindices[i-1]	= blockindices[i-1] - upper;
                for(j=0; j < upper; j++)
                {
                    pitch += blocksizes[i-1][blockindices[i-1]+j-upper];
                }
                upper = pitch;
                pitch = 0;
                i--;
            }
        }
        
        if(up)
        {
            i = up-1;
            PetscInt	j,pitch = 0;
            
            upper2 = blocksizes[i][newblockindices[i]];                         //which is used as start for the increment pitch
            
            while(i)
            {
                for(j=0; j < upper2; j++)
                {
                    pitch += blocksizes[i-1][newblockindices[i-1]+j];
                }
                upper2 = pitch;
                pitch = 0;
                i--;
            }
        }
    }
    else
    {
        PetscInt type = GetType(down);
        
        if( down-multiMLS_start[type] )        //start with downward pitch
        {
            i = down-1;
            PetscInt    j,pitch = 0;
            
            newblockindices[i]  = blockindices[i]-1;                            //this is said position, where the decrement pitch leads to
            upper               = blocksizes[i][blockindices[i]-1];             //
            
            while( i > multiMLS_start[type] )
            {
                newblockindices[i-1]    = blockindices[i-1] - upper;
                for(j=0; j < upper; j++)
                {
                    pitch += blocksizes[i-1][blockindices[i-1]+j-upper];
                }
                upper = pitch;
                pitch = 0;
                i--;
            }
        }
        
        if( up-multiMLS_start[type] )
        {
            i = up-1;
            PetscInt    j,pitch = 0;
            
            upper2 = blocksizes[i][newblockindices[i]];                         //which is used as start for the increment pitch
            
            while( i > multiMLS_start[type] )
            {
                for(j=0; j < upper2; j++)
                {
                    pitch += blocksizes[i-1][newblockindices[i-1]+j];
                }
                upper2 = pitch;
                pitch = 0;
                i--;
            }
        }
        if(type)
        {
            upper *= mls_dof[type-1];                              //multiply times the dofs of all preceding mls types
            upper2 *= mls_dof[type-1];
        }
    }

    return	ret*(upper2-upper);
}


/**
 * @brief	Computes the pitch for increasing the value of modedim by one
 * 
 * @param	modedim		the mode dim.
 * 
 */

inline PetscInt Index::ModeIPitch(PetscInt modedim)
{
    PetscInt	coarse,fine,i;
    
    coarse = mls_dof[N_D_MLS-1];
    for(i=0; i < (modedim-firstmodedim)/2; i++)	coarse *= mode_dofs[i];
      
    if( !((modedim-firstmodedim)%2) )			//ket style (left,first) mode dim
    {
      fine = 1;
    }
    else						//bra style (right, second) mode dim
    {
      fine = blocksizes[modedim-1][blockindices[modedim-1]];
      if( blocksizes[modedim-1][0]-1 <= blockindices[modedim-1] )	fine--;
    }
    
    return	coarse*fine;
}


/**
 * @brief	Computes the pitch for increasing the value of modedim by one
 * 
 * @param	modedim		the mode dim.
 * 
 */

inline PetscInt Index::ModeDPitch(PetscInt modedim)
{
    PetscInt	coarse,fine,i;
    
    coarse = mls_dof[N_D_MLS-1];
    for(i=0; i < (modedim-firstmodedim)/2; i++)
    {
      coarse *= mode_dofs[i];
    }
    if( !((modedim-firstmodedim)%2) )			//ket style (left,first) mode dim
    {
      fine = 1;
    }
    else						//bra style (right, second) mode dim
    {
      fine = blocksizes[modedim-1][blockindices[modedim-1]-1];
      if( blocksizes[modedim-1][0]-1 < blockindices[modedim-1] )	fine--;
    }
    
    return	-coarse*fine;
}





//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  inline loop utilities
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

/**
 * @brief	Initializes the Index to the current local layout snippet. Returns the local start value.
 * 
 */

inline PetscInt Index::InitializeLocal()
{
    PetscInt	i;
    
    for(i=0; i < num_dims; i++) blockindices[i]	= loc_blockstarters[i];
    for(i=0; i < num_dims; i++) indices[i]	= loc_starters[i];
    
    dmindex	= loc_start;
    mlsindex	= mls_start;
    isend	= 0;
    
    return	loc_start;
}


/**
 * @brief	Initializes the Index to the global start value. Returns the global start value, i.e. 0.
 * 
 */

inline PetscInt Index::InitializeGlobal()
{
    PetscInt	i;
    
    for(i=0; i < num_dims; i++) blockindices[i]	= 0;
    for(i=0; i < num_dims; i++) indices[i]	= 0;
    
    dmindex	= 0;
    mlsindex	= 0;
    isend	= 0;
    
    return	0;
}


/**
 * @brief	Increments density matrix index dmindex by one and increments the internal structures to the next allowed value by calling IncrementInternal().
 * 
 */

inline PetscInt Index::Increment()
{
    dmindex++;
    
    mlsindex++;					                        //seems quite simple
    if( mlsindex == mls_dof[N_D_MLS-1] )	mlsindex = 0;	//
    
    if( N_D_MLS <= 1 )          IncrementInternal();          //one or zero types of mls
    else                        IncrementInternalMultiMLS();  //more types of mls
    
    return dmindex;
}


/**
 * @brief	Increments the internal arrays indices and blockindices to the next allowed value. Becomes recursive in case of mls offdiagonal truncation.
 * 
 */

inline void Index::IncrementInternal()
{
    PetscInt	i,j;
    
    indices[0]++;
    i = 0;
    while( indices[i] >= blocksizes[i][blockindices[i]] )
    {
      if( i < firstmodedim-1 )			//increment mls dim values
      {
          indices[i] = 0;
          indices[i+1]++;
          blockindices[i]++;
          i++;
      }
      else if ( firstmodedim != num_dims )	//there is at least one mode
      {
          if ( i == firstmodedim-1 )		//transition between mls dims and mode dims
          {
              for(j=0; j < firstmodedim; j++)	blockindices[j] = 0;
              indices[i] = 0;               //TODO: this seems to be obsolete
              i++;
              indices[i]++;
          }
          else if( i < num_dims-1 )
          {
              if( !((i-firstmodedim)%2) )		//multiple of 2, means first mode dim
              {
                  indices[i] = 0;
                  indices[i+1]++;
                  blockindices[i]++;
                  i++;
              }
              else					//not a multiple of 2, means second mode dim
              {
                  indices[i] = 0;
                  indices[i+1]++;
                  blockindices[i-1] = 0;
                  i++;
              }
          }
          else					//the end of the line
          {
              isend = 1;
              break;
          }
      }
      else					//the end of the line
      {
          isend = 1;
          break;
      }
    }
    if( !isend ) if( !blocksizes[0][blockindices[0]] ) IncrementInternal();	//if the current block in dim 0 has blocksize zero, do it again, because there is no allowed entry here, due to truncation
}


/**
 * @brief    Increments the internal arrays indices and blockindices to the next allowed value. Becomes recursive in case of mls offdiagonal truncation.
 *
 */

inline void Index::IncrementInternalMultiMLS()
{
    PetscInt    i,j,k;
    
    indices[0]++;
    i = 0;
    k = 1;
    
    while( indices[i] >= blocksizes[i][blockindices[i]] )
    {
        if( i < firstmodedim-1 )            //increment mls dim values
        {
            if( i < multiMLS_start[k]-1 )   //inside the current mls type group
            {
                indices[i] = 0;
                indices[i+1]++;
                blockindices[i]++;
                i++;
            }
            else if ( i == multiMLS_start[k]-1 )    //at the transition between two types of mls
            {
                for(j=0; j < multiMLS_start[k]; j++)    blockindices[j] = 0;
                indices[i] = 0;
                i++;
                k++;                                //TODO: check wether we need an overflow check for this variable
                indices[i]++;
            }
        }
        else if ( firstmodedim != num_dims )    //there is at least one mode
        {
            if ( i == firstmodedim-1 )        //transition between mls dims and mode dims
            {
                for(j=0; j < firstmodedim; j++)    blockindices[j] = 0;
                indices[i] = 0;                 //TODO: this seems to be obsolete
                i++;
                indices[i]++;
            }
            else if( i < num_dims-1 )
            {
                if( !((i-firstmodedim)%2) )        //multiple of 2, means first mode dim
                {
                    indices[i] = 0;
                    indices[i+1]++;
                    blockindices[i]++;
                    i++;
                }
                else                    //not a multiple of 2, means second mode dim
                {
                    indices[i] = 0;
                    indices[i+1]++;
                    blockindices[i-1] = 0;
                    i++;
                }
            }
            else                    //the end of the line
            {
                isend = 1;
                break;
            }
        }
        else                    //the end of the line
        {
            isend = 1;
            break;
        }
    }
    if( !isend )        //if the current block in dim 0 has blocksize zero, do it again, because there is no allowed entry here, due to truncation
    {
        PetscInt    check = 0;
        for(i=0;i<N_D_MLS;i++)
        {
            if( !blocksizes[multiMLS_start[i]][blockindices[multiMLS_start[i]]] ) check++;
        }
        if(check)   IncrementInternalMultiMLS();
    }
}


/**
 * @brief	Checks whether a while loop can be continued with the current Index. Says no at the global and local ends.
 * 
 */

inline PetscInt Index::ContinueLocal()
{
    PetscInt	ret = 1;
    
    if (isend)			ret = 0;
    if (dmindex == loc_end)	ret = 0;
    
    return	ret;
}


/**
 * @brief	Checks whether a while loop can be continued with the current Index. Says no at the global and local ends.
 * 
 */

inline PetscInt Index::ContinueGlobal()
{
    PetscInt	ret = 1;
    
    if (isend)			ret = 0;
    if (dmindex == total_dof)	ret = 0;
    
    return	ret;
}


/**
 * @brief    Checks whether a while loop can be continued with the current Index. Says no at the end of the first mls block of all types. Makes only sense in combination with InitializeGlobal().
 *
 */

inline PetscInt Index::ContinueMLS()
{
    PetscInt    ret = 1;
    
    if (dmindex >= mls_dof[N_D_MLS-1])    ret = 0;
    
    return    ret;
}


/**
 * @brief	Checks whether a while loop can be continued with the current Index. Says no at the end of the first mls block of type i. Makes only sense in combination with InitializeGlobal().
 * 
 */

inline PetscInt Index::ContinueMLS(PetscInt i)
{
    PetscInt	ret = 1;
    
    if (dmindex >= mls_dof[i])	ret = 0;
    
    return	ret;
}


//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  logical statements
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

/**
 * @brief	Checks whether the dim dimension can be incremented by one. Without checking the n00 part!!!
 * 
 * @param	dim	the dimension
 * 
 */

inline PetscInt Index::CanIncrement(PetscInt dim)
{
    PetscInt	ret = 0;
    
    if( dim < firstmodedim )
    {
      if( (mlsdim_maxvalues[dim] != indices[dim]) )						//mls dims: quantum number is not maximal
      {
          ret++;
          goto stop;
      }
    }
    else
    {
      if( !((dim-firstmodedim)%2) && indices[dim] < blocksizes[dim][blockindices[dim]]-1 )	//mode dims
      {
          ret++;
          goto stop;
      }
      else if( indices[dim] < blocksizes[dim][blockindices[dim]]-1 && (indices[dim-1] != 0 || blockindices[dim-1] < blocksizes[dim-1][0]-1 ) )
      {
          ret++;
          goto stop;
      }
    }
    
    stop:
    return	ret;
}


/**
 * @brief	Checks whether the dim dimension can be decremented by one.
 * 
 * @param	dim	the dimension
 * 
 */

inline PetscInt Index::CanDecrement(PetscInt dim)
{
    PetscInt	ret = 0;
    
    if( dim < firstmodedim )
    {
      if( indices[dim] )
      {
          ret++;
          goto stop;
      }
    }
    else
    {
      if( !((dim-firstmodedim)%2) )
      {
          if( indices[dim] )
          {
              ret++;
              goto stop;
          }
      }
      else
      {
          if( indices[dim] && !(indices[dim] < blocksizes[dim][0] - blocksizes[dim-1][0] + 1 && indices[dim-1] == blocksizes[dim-1][blockindices[dim-1]]-1) )
          {
              ret++;
              goto stop;
          }
          else if( indices[dim] < blocksizes[dim][0] - blocksizes[dim-1][0] + 1 )
          {
              //TODO: check if this is right
          }
      }
    }
    
    stop:
    return	ret;
}


/**
 * @brief	Checks whether an element given by index is inlcuded in the used excerpt of the density matrix or not.
 * 
 * @param	index	the multiindex of the element to be checked.
 * 
 */

inline PetscInt Index::OutOfBounds(PetscInt* index)
{
    PetscInt	i,ret = 0;
    
    for(i=0; i < firstmodedim; i++)
    {
      if( index[i] > mlsdim_maxvalues[i] )
      {
	ret++;
	goto stop;
      }
    }
    
    i = firstmodedim;
    
    while( i < num_dims )
    {
      if( index[i] >= blocksizes[i][index[i]+1] )	
      {
	ret++;
	goto stop;
      }
      i++;
      if( index[i] >= blocksizes[i][0] )
      {
	ret++;
	goto stop;
      }
      i++;
    }
    
    stop:
    return	ret;
}


/**
 * @brief	Checks whether an element given by mls index is inlcuded in the used excerpt of the density matrix or not. Mainly useful for the recursive setup of the gnfcts.
 * 
 * @param	index	the multiindex of the element to be checked.
 * 
 */

inline PetscInt Index::MLSOutOfBounds(PetscInt* index)
{
    PetscInt	i,ret = 0;
    
    for(i=0; i < firstmodedim; i++)
    {
      if( indices[i] > mlsdim_maxvalues[i] )
      {
	ret++;
	break;
      }
    }

    return	ret;
}


/**
 * @brief	Checks whether the current element is polarization like or not. Returns 1 if yes and 0 if not.
 * 
 */

inline PetscInt Index::IsPol()
{
    PetscInt	i,ret = 0;
    
    for(i=0; i < firstmodedim; i++)			//check them mls dims!
    {
      if( mlsdim_pol[i] && indices[i] )			//if it is polarization like and unequal to zero it is a polarization
      {
	ret++;
	goto stop;
      }
    }
    
    i = firstmodedim;					//redundant but better to read
    while( i < num_dims )				//check them mode dims
    {
      if( MAX(0,blockindices[i]-blocksizes[i][0]+1) + indices[i] != indices[i+1] )	//if ml != mr then it is a polarization
      {
	ret++;
	goto stop;
      }
      i += 2;
    }
    
    stop:
    return	ret;					//zero if not a polarization, 1 if yes
}


/**
 * @brief	Checks whether the current position of the index belongs to the local density matrix snippet or not. Returns 1 if yes, 0 if not.
 * 
 */

inline PetscInt Index::IsLocal()
{
    PetscInt	ret = 0;
    
    if( dmindex >= loc_start && dmindex < loc_end )	ret++;
    
    return	ret;
}


/**
 * @brief	Checks whether the element corresponding to index is local or not.
 * 
 */

inline PetscInt Index::IsLocal(PetscInt index)
{
    PetscInt	ret = 0;
    
    if( index >= loc_start && index < loc_end )	ret++;
    
    return	ret;
}


/**
 * @brief	Checks whether the element corresponding to index is local or not. Sets the index to the indices position while doing so. Slow!
 * 
 */

inline PetscInt Index::IsLocal(PetscInt * indices)
{   
    SetIndices(indices);
    
    return	IsLocal();
}


/**
 * @brief	Are the mls degrees of freedom density like? 1 if yes, 0 if not. Checks only the mls dofs! Is intended to be used with mode specific checks!
 * 
 */

inline PetscInt Index::IsMLSDensity()
{
    PetscInt	i,ret = 1;
    
    for(i=0; i < firstmodedim; i++)			//check them mls dims!
    {
      if( mlsdim_pol[i] && indices[i] )			//if it is polarization like and unequal to zero it is a polarization
      {
	ret = 0;
	break;
      }
    }
    
    return	ret;					//zero if not a polarization, 1 if yes
}


/**
 * @brief	Are the mode degrees of freedom density like? 1 if yes, 0 if not. Checks only the mode dofs! Is intended to be used with mls specific checks!
 * 
 */

inline PetscInt Index::IsModeDensity()
{
    PetscInt	i,ret = 1;
    
    i = firstmodedim;
    
    while( i < num_dims )									//check them mls dims!
    {
      if( MAX(0,blockindices[i]-blocksizes[i][0]+1) + indices[i] != indices[i+1] )		//if ml != mr then its polarization like
      {
	ret = 0;
	break;
      }
      i += 2;
    }
    
    return	ret;					//zero if not a polarization, 1 if yes
}


/**
 * @brief	Are the mls degrees of freedom in the ground state? 1 if yes, 0 if not. Checks only the mls dofs! Is intended to be used with mls specific checks!
 * 		Is the same as ( 0 == index->MLSIndex() )
 * 
 */

inline PetscInt Index::IsMLSGroundState()
{   
    if(mlsindex)	return	0;
    else		return	1;
}

/**
 * @brief	Checks whether the current element is a first offdiagonal in the dim mls dimension and density like in all the other dimensions. Checks for dm entries that occur in the <J_-> expectation value,
 *		i.e. the type of elements the action of the mls lowering operator on diagonal (trace) elements produces... 
 * 
 * @param	dim	type PetscInt; the dimension to be checked.
 * 
 */

inline PetscInt Index::IsMLSOneDimFirstOffdiag(PetscInt dim)
{
    PetscInt	i,ret = 0;
    
    for(i=0; i < firstmodedim; i++)			//check them mls dims!
    {
      if( mlsdim_pol[i] && indices[i] && i != dim )	//if it is polarization like && unequal to zero && not the dim dimension
      {
	ret = 0;
	goto stop;
      }
      if( mlsdim_pol[i] && i == dim )			//a bit of a failsafe
      {
	if( indices[i] == 1 )
	{
	  ret++;
	}
	else
	{
	  ret = 0;
	  goto stop;
	}
      }
    }
    
    stop:
    return	ret;					//zero if not a polarization, 1 if yes
}


/**
 * @brief	Checks whether the current element is a #number offdiagonal in the dim mls dimension and density like in all the other dimensions. Checks for dm entries that occur in the <(J_xy)^k> expectation values,
 *		i.e. the type of elements the action of the mls lowering operator on diagonal (trace) elements produces... 
 * 
 * @param	dim	the dimension to be checked.
 * @param	number	the order of the offdiagonal element
 * 
 */

inline PetscInt Index::IsMLSOneDimNumberOffdiag(PetscInt dim, PetscInt number)
{
    PetscInt	i,ret = 0;
    
    for(i=0; i < firstmodedim; i++)			//check them mls dims!
    {
      if( mlsdim_pol[i] && indices[i] && i != dim )	//if it is polarization like && unequal to zero && not the dim dimension
      {
	ret = 0;
	goto stop;
      }
      if( mlsdim_pol[i] && i == dim )			//a bit of a failsafe
      {
	if( indices[i] == number )
	{
	  ret++;
	}
	else
	{
	  ret = 0;
	  goto stop;
	}
      }
    }
    
    stop:
    return	ret;					//zero if not a polarization, 1 if yes
}


/**
 * @brief	Checks whether the current element is a first offdiagonal in the dim1 and dim2 mls dimensions and density like in all the other dimensions. Checks for dm entries that occur in the <J_-> expectation value,
 *		i.e. the type of elements the action of the mls lowering operator on diagonal (trace) elements produces... 
 * 
 * @param	dim1	the first dimension to be checked.
 * @param	dim2	the second dimension to be checked.
 * 
 */

inline PetscInt Index::IsMLSTwoDimFirstOffdiag(PetscInt dim1, PetscInt dim2)
{
    PetscInt	i,ret = 0;
    
    for(i=0; i < firstmodedim; i++)			//check them mls dims!
    {
      if( mlsdim_pol[i] && !indices[i] && i != dim1 && i != dim2 )	//if it is polarization like and unequal to zero it is a polarization and not the dim1/2 dims
      {
	ret = 0;
	goto stop;
      }
      if( mlsdim_pol[i] && i == dim1 )			//a bit of a failsafe
      {
	if( indices[i] == 1 )
	{
	  ret++;
	}
	else
	{
	  ret = 0;
	  goto stop;
	}
      }
      if( mlsdim_pol[i] && i == dim2 )			//a bit of a failsafe
      {
	if( indices[i] == 1 )
	{
	  ret++;
	}
	else
	{
	  ret = 0;
	  goto stop;
	}
      }
    }
    
    stop:
    return	ret/2;					//zero if not a polarization, 1 if yes
}


/**
 * @brief	Checks whether the current element is a #number offdiagonal in the dim1 and dim2 mls dimensions and density like in all the other dimensions. Checks for dm entries that occur in the offdiag mls distributions. 
 * 
 * @param	dim1	the first dimension to be checked.
 * @param	dim2	the second dimension to be checked.
 * @param	number	the offdiagonality index
 * 
 */

inline PetscInt Index::IsMLSTwoDimNumberOffdiag(PetscInt dim1, PetscInt dim2, PetscInt number)
{
    PetscInt	i,ret = 0;
    
    for(i=0; i < firstmodedim; i++)			//check them mls dims!
    {
      if( mlsdim_pol[i] && !indices[i] && i != dim1 && i != dim2 )	//if it is polarization like and unequal to zero it is a polarization and not the dim1/2 dims
      {
	ret = 0;
	goto stop;
      }
      if( mlsdim_pol[i] && i == dim1 )			//a bit of a failsafe
      {
	if( indices[i] == number )
	{
	  ret++;
	}
	else
	{
	  ret = 0;
	  goto stop;
	}
      }
      if( mlsdim_pol[i] && i == dim2 )			//a bit of a failsafe
      {
	if( indices[i] == number )
	{
	  ret++;
	}
	else
	{
	  ret = 0;
	  goto stop;
	}
      }
    }
    
    stop:
    return	ret/2;					//zero if not a polarization, 1 if yes
}


/**
 * @brief	Checks whether current index position has the mode given by dim in a first left diagonal state i.e. a |m-1>< m| state. 1 if true, 0 if not. 
 * 
 * @param	dim	the mode dim, first mode dim (of the two)!!!
 * 
 */

inline PetscInt Index::IsModeOneDimFirstLeftOffdiag(PetscInt dim)
{
    PetscInt	i,ret = 0;
    
    i = firstmodedim;
    while( i < num_dims )				//check them mode dims
    {
      if( MAX(0,blockindices[i]-blocksizes[i][0]+1) + indices[i] != indices[i+1] && i != dim )	//if ml != mr then it is a polarization
      {
	ret = 0;
	goto stop;
      }
      if( i == dim )										//if its the dim dim dim dim...
      {
	if( MAX(0,blockindices[i]-blocksizes[i][0]+1) + indices[i] == indices[i+1]-1 )		//if ml == mr-1 then its right
	{
	  ret++;
	}
	else
	{
	  ret = 0;
	  goto stop;
	}
      }
      i += 2;
    }
    
    stop:
    return	ret;					//zero if not a polarization, 1 if yes
}


/**
 * @brief	Checks whether current index position has the mode given by dim in a first right diagonal state i.e. a |m+1>< m| state. 1 if true, 0 if not. 
 * 
 * @param	dim	the mode dim, first mode dim (of the two)!!!
 * 
 */

inline PetscInt Index::IsModeOneDimFirstRightOffdiag(PetscInt dim)
{
    PetscInt	i,ret = 0;
    
    i = firstmodedim;
    while( i < num_dims )				//check them mode dims
    {
      if( MAX(0,blockindices[i]-blocksizes[i][0]+1) + indices[i] != indices[i+1] && i != dim )	//if ml != mr then it is a polarization
      {
	ret = 0;
	goto stop;
      }
      if( i == dim )										//if its the dim dim dim dim...
      {
	if( MAX(0,blockindices[i]-blocksizes[i][0]+1) + indices[i] - 1 == indices[i+1] )	//if ml-1 == mr then its right
	{
	  ret++;
	}
	else
	{
	  ret = 0;
	  goto stop;
	}
      }
      i += 2;
    }
    
    stop:
    return	ret;					//zero if not a polarization, 1 if yes
}



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  general helper functions
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------


/**
 * @brief    Returns the mls type number of a given mls dimension i. Does not work for mode dimensions! Be careful!
 *
 * @param    dim    the mls dimension to be checked.
 *
 */

inline PetscInt Index::GetType(PetscInt dim)
{
    PetscInt type = 0;
    
    if( dim >= 0 )
    {
        while ( dim >= multiMLS_start[type+1] ) type++;
    }
    else
    {
        type = -(dim+1);
    }
    
    return type;
}


/**
 * @brief	Returns the quantum number of the dim dimension of the current element.
 * 
 * @param	dim	the mls dimension to be checked.
 * 
 */

inline PetscInt Index::MLSQN(PetscInt dim)
{
    PetscInt	i,ret = 0;
    
    if( N_D_MLS == 1 )
    {
        if( dim == -1 )
        {
            ret	= NMLS[0];
            for(i=0; i < firstmodedim; i++)	ret -= indices[i];
        }
        else	ret	= indices[dim];
    }
    else
    {
        PetscInt type = GetType(dim);
        
        if( dim == -type-1 )
        {
            ret    = NMLS[type];
            for(i=multiMLS_start[type]; i < multiMLS_start[type+1]; i++)    ret -= indices[i];
        }
        else    ret    = indices[dim];
    }

    return	ret;
}


/**
 * @brief	Compute and return the mode dimension quantum number.
 * 
 * @param	dim	the mode dim to be checked.
 * 
 */

inline PetscInt Index::ModeQN(PetscInt dim)
{
    PetscInt	ret;
    
    if( !((dim-firstmodedim)%2) )			//ket style (left,first) mode dim
    {
      ret = MAX(0,blockindices[dim]-blocksizes[dim][0]+1) + indices[dim];
    }
    else						//bra style (right, second) mode dim
    {
      ret = indices[dim];
    }
    
    return	ret;
}


/**
 * @brief	Return the indices. Returns n00 for -1.
 * 
 * @param	dim	the dim.
 * 
 */

inline PetscInt Index::Indices(PetscInt dim)
{
    PetscInt	ret;
    
    if( dim == -1 )	ret = MLSQN(-1);
    else		    ret = indices[dim];
    
    return	ret;
}

/**
 * @brief	Return the maximum allowed quantum number for the respective dimension. Works also for n00, then dim has to be -1.
 * 
 * @param	dim	the dimension to be checked.
 * 
 */

inline PetscInt Index::MaxQN(PetscInt dim)
{
    PetscInt	ret,dummy,type = GetType(dim);
    
    if( dim == -type-1 )
    {
      ret	= NMLS[type] +1;
    }
    else if( dim < firstmodedim )
    {
      ret	= mlsdim_maxvalues[dim];
    }
    else
    {
      dummy	= (dim-firstmodedim)/2;
      ret	= blocksizes[dim+1-dummy][0] - 1;
    }
    
    return	ret;
}


/**
 * @brief	Set the Index object to the index given by inindices. Rather slow. Global function, the index does not neccessarily belong to the local snippet.
 * 
 * @param	index	the wanted index.
 * 
 */

inline void Index::SetIndex(PetscInt index)
{
    PetscInt	check = 0;
    
    InitializeGlobal();					//start from the beginning
    
    if( index == dmindex )	check++;		//if its a match set check
    
    while(!check)					//while not equal
    {
      Increment();					//increment
      
      if( index == dmindex )	check++;		//if its a match set check
    }
}


/**
 * @brief	Set the Index object to the index given by index. Local function, does only work with local indices! Produces crap otherwise. Returns the global index.
 * 
 * @param	index	the wanted index.
 * 
 */

inline PetscInt Index::SetIndexLocal(PetscInt index)
{
    PetscInt	check = 0;
    
    InitializeLocal();					//start from the beginning
    
    if( index+loc_start == dmindex )	check++;	//if its a match set check
    
    while( !check && IsLocal() )			//while not equal
    {
      Increment();					//increment
      
      if( index+loc_start == dmindex )	check++;	//if its a match set check
    }
    
    return	dmindex;
}


/**
 * @brief	Set the Index object to the index given by inindices. Rather slow. Global function, the index does not neccessarily belong to the local snippet.
 * 
 * @param	inindices	the wanted indices.
 * 
 */

inline PetscInt Index::SetIndices(PetscInt* inindices)
{
    PetscInt	check = 0,i;
    
    InitializeGlobal();					//start from the beginning
    
    for(i=0; i < num_dims; i++)				//test the first element
    {
      if(inindices[i] == indices[i])	check++;
    }
    check /= num_dims;					//0 if not equal, 1 if equal
    
    while( ContinueGlobal() && !check )			//while not equal
    {
      Increment();					//increment
      
      for(i=0; i < num_dims; i++)			//and check again
      {
	if(inindices[i] == indices[i])	check++;
      }
      check /= num_dims;
    }
    
    return	dmindex;				//return value is the dmindex
}


#endif		// _Index
