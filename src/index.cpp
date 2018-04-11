

/**
 * @file	index.cpp
 * 
 * 		This file contains the all functionalities that are not used in loops.
 * 		Therefore they are not inline, have a errorchecking and so on. Not very efficient, but convenient.
 * 		Especially contains the constructor and destructor of the Index class.
 * 
 * @author	Michael Gegg
 * 
 */

#include"../include/index.hpp"

//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  I/O functionalities
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

#undef __FUNCT__
#define __FUNCT__ "PrintElements"

/**
 * @brief	Print all elements of the index into stdout
 * 
 */

PetscErrorCode Index::PrintElements()
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    PetscInt		i;

    InitializeGlobal();					    //initialize the index
        
    while( ContinueGlobal() )				//is it possible to continue?
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"dmindex: %d\t mlsindex: %d\t",dmindex,mlsindex); CHKERRQ(ierr);		//dmindex and mlsindex
      
      ierr = PetscPrintf(PETSC_COMM_WORLD,"n00: %d\t",MLSQN(-1)); CHKERRQ(ierr);					            //the n00 value
      
      ierr = PetscPrintf(PETSC_COMM_WORLD,"indices: "); CHKERRQ(ierr);							                //normal indices
      for(i=0; i < num_dims; i++)	ierr = PetscPrintf(PETSC_COMM_WORLD,"%d ",indices[i]); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\t"); CHKERRQ(ierr);
      
      ierr = PetscPrintf(PETSC_COMM_WORLD,"blockindices: "); CHKERRQ(ierr);						                //block indices
      for(i=0; i < num_dims; i++)	ierr = PetscPrintf(PETSC_COMM_WORLD,"%d ",blockindices[i]); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\n"); CHKERRQ(ierr);
      
      Increment();				//increment the index by one
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PrintDiagonals"

/**
 * @brief	Print all diagonal elements of the complete index into stdout
 * 
 */

PetscErrorCode Index::PrintDiagonals()
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    PetscInt		i;

    InitializeGlobal();					//initialize the index
        
    while( ContinueGlobal() )				//is it possible to continue?
    {
      if( !IsPol() )
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"dmindex: %d\t mlsindex: %d\t",dmindex,mlsindex); CHKERRQ(ierr);		//dmindex and mlsindex
      
	ierr = PetscPrintf(PETSC_COMM_WORLD,"n00: %d\t",MLSQN(-1)); CHKERRQ(ierr);					//n00 value
      
	ierr = PetscPrintf(PETSC_COMM_WORLD,"indices: "); CHKERRQ(ierr);						//normal indices
	for(i=0; i < num_dims; i++)	ierr = PetscPrintf(PETSC_COMM_WORLD,"%d ",indices[i]); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\t"); CHKERRQ(ierr);
      
	ierr = PetscPrintf(PETSC_COMM_WORLD,"blockindices: "); CHKERRQ(ierr);						//block indices
	for(i=0; i < num_dims; i++)	ierr = PetscPrintf(PETSC_COMM_WORLD,"%d ",blockindices[i]); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\n"); CHKERRQ(ierr);
      }
      
      Increment();				//increment the index by one
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PrintBlockSizes"

/**
 * @brief	Print all the block sizes to stdout
 * 
 */

PetscErrorCode Index::PrintBlockSizes()
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    PetscInt		i,j;
    
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nblocksizes:\n"); CHKERRQ(ierr);
    for(i=0; i < num_dims; i++)
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"dim %d: blocks %d:\t",i,blocksizes_max[i]); CHKERRQ(ierr);
      for(j=0; j < blocksizes_max[i]; j++)
      {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"%d ",blocksizes[i][j]); CHKERRQ(ierr);
      }
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\n"); CHKERRQ(ierr);
    }
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PrintIndex"

/**
 * @brief	Print the current value of dmindex to stdout
 * 
 */

PetscErrorCode Index::PrintIndex()
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    ierr = PetscPrintf(PETSC_COMM_WORLD,"dmindex = %d\t mlsindex = %d\n",dmindex,mlsindex); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PrintIndices"

/**
 * @brief	Print the current value of dmindex to stdout.
 * 
 */

PetscErrorCode Index::PrintIndices()
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    PetscInt		i;
    
    ierr = PetscPrintf(PETSC_COMM_WORLD,"indices: "); CHKERRQ(ierr);
    for(i=0; i < num_dims; i++) ierr = PetscPrintf(PETSC_COMM_WORLD,"%d ",indices[i]); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\t"); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PrintBlockIndices"

/**
 * @brief	Print the current value of dmindex to stdout.
 * 
 */

PetscErrorCode Index::PrintBlockIndices()
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    PetscInt		i;
    
    ierr = PetscPrintf(PETSC_COMM_WORLD,"blockindices: "); CHKERRQ(ierr);
    for(i=0; i < num_dims; i++) ierr = PetscPrintf(PETSC_COMM_WORLD,"%d ",blockindices[i]); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n"); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PrintMLSDimMaxValues"

/**
 * @brief	Print the mlsdim_maxvalues into stdout.
 * 
 */

PetscErrorCode Index::PrintMLSDimMaxValues()
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    PetscInt		i;
    
    ierr = PetscPrintf(PETSC_COMM_WORLD,"dim_maxvalues: "); CHKERRQ(ierr);
    for(i=0; i < firstmodedim; i++)	ierr = PetscPrintf(PETSC_COMM_WORLD,"%d ",mlsdim_maxvalues[i]);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SetIndexFromQN"

/**
 * @brief	Set the Index object to the index corresponding to the quantum numbers in the inqns array. Rather slow.
 * 
 * @param	inqns	the array containing the desired quantum numbers.
 * 
 */

PetscErrorCode Index::SetIndexFromQN(PetscInt* inqns)
{
    PetscFunctionBeginUser;
    
    PetscInt	desired[num_dims];						//the actual indices in the internal Index class format
    PetscInt	i;
    
    for(i=0; i < firstmodedim; i++)						//get the mls indices
    {
      if( inqns[i] <= mlsdim_maxvalues[i] ) desired[i] = inqns[i];		//mls indices correspond to the qunatum numbers...
      else
      {
	(*PetscErrorPrintf)("Requestet density matrix entry is not contained in the used excerpt of the density matrix!\n");
	(*PetscErrorPrintf)("MLS dimension: %d, polarization %d; max value: %d, requested value: %d\n",i,mlsdim_pol[i],mlsdim_maxvalues[i],inqns[i]);
	SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
	break;
      }
    }
    while( i < num_dims )							//get the mode indices
    {
      if( inqns[i] < blocksizes[i+1][0] && inqns[i+1] < blocksizes[i+1][0] && abs(inqns[i]-inqns[i+1]) < blocksizes[i][0] )
      {
	desired[i+1] = inqns[i+1];						//this corresponds to blockindex[i]!
	desired[i] = inqns[i] - MAX(0, desired[i+1]-blocksizes[i][0]+1);	//there is a little shift here
	i += 2;
      }
      else
      {
	(*PetscErrorPrintf)("Requestet density matrix entry is not contained in the used excerpt of the density matrix!\n");
	(*PetscErrorPrintf)("Mode dimension: %d, max value: %d, offidagonals: %d, requested values: |%d><%d|\n",i,blocksizes[i+1][0]-1,blocksizes[i][0]-1,inqns[i],inqns[i+1]);
	SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
	break;
      }
    }
    
    SetIndices(desired);							//set the Index object to the desired indices
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PrintGenInfos"

/**
 * @brief	Print general information about the index into stdout.
 * 
 */

PetscErrorCode Index::PrintGenInfos()
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    PetscInt		i;
    
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nIndex general infos:\n"); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"collection of %d %d-level systems: %d mls dimensions:\n",NMLS,num_levels,firstmodedim); CHKERRQ(ierr);
    for(i=0; i < firstmodedim; i++) ierr = PetscPrintf(PETSC_COMM_WORLD,"dim: maximum value %d, polarization %d\n",mlsdim_maxvalues[i],mlsdim_pol[i]); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n%d modes:\n",(num_dims-firstmodedim)/2); CHKERRQ(ierr);
    while( i < num_dims )
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"mode %d: maximum number state order %d, maximum number of offidagonals %d\n",(i-firstmodedim)/2,blocksizes[i+1][0]-1,blocksizes[i][0]-1); CHKERRQ(ierr);
      i += 2;
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\ntotal degrees of freedom: %d, mls degrees of freedom: %d\n",total_dof,mls_dof[0]); CHKERRQ(ierr); //TODO: make this meaninful for multi mls
    
    ierr = PrintBlockSizes(); CHKERRQ(ierr);
    
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\n"); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  constructors and destructors
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

/**
 * @brief	This constructor initializes the Index class with full support for truncation and uniprocessor use. The parallel layout needs to be set after creating the density matrix using SetParallelLayout()
 * 
 * @param	nlevels		the number of levels per mls.
 * @param	nummlsdims	the number of mls dimensions.
 * @param	mlspol		logical array that contains the information which mls dimension is density like and which is polarization like
 * @param	N		the total number of mls
 * @param	dimlenghts	the lenghts of the respective mls dimensions, for truncation
 * @param	modes		the number of modes
 * @param	modedimlengths	the sizes of the mode dimensions: {mode1order,dmr1,mode2order,dmr2,...}
 * 
 */

Index::Index(PetscInt nlevels, PetscInt nummlsdims, PetscInt * mlspol, PetscInt N, PetscInt * dimlenghts, PetscInt modes, PetscInt *modedimlengths)
{
    PetscInt	i;
    
    num_levels		= nlevels;
    num_dims		= nummlsdims+2*modes;					//total number of dimensions
    firstmodedim	= nummlsdims;						    //the index of the first mode dimension
    N_D_MLS         = 1;                                    //number of different MLS types
    
    PetscInt    *loc_multistart = new PetscInt [N_D_MLS];   //dummy values
    loc_multistart[0]   = 0;                                //only really needed for multi mls type usage
    multiMLS_start      = loc_multistart;                   //
    
    PetscInt    *loc_nmls = new PetscInt [N_D_MLS];         //only one type of mls in this case
    loc_nmls[0]     = N;                                    //the number of mls
    NMLS            = loc_nmls;                             //
    
    PetscInt    *loc_mlsdof = new PetscInt [N_D_MLS];       //only one type of mls in this case
    loc_mlsdof[0]   = BinomDim(NMLS[0],nummlsdims);         //the untruncated value for the total mls dofs
    mls_dof         = loc_mlsdof;                           //
    
    PetscInt	*locdim_maxvalues	= new PetscInt [firstmodedim];		    //the maximum value of the mls dimensions
    for(i=0; i < firstmodedim; i++)	locdim_maxvalues[i] = NMLS[0];		    //initialize it to NMLS, gets truncated later
    mlsdim_maxvalues			= locdim_maxvalues;
    
    
  //-----------------------------------------------------------------
  //allocate blocks
  //-----------------------------------------------------------------
    PetscInt	*locblocksizes_max	= new PetscInt [num_dims];		//the lengths of the blocksizes arrays
    blocksizes_max			= locblocksizes_max;

    PetscInt	**locblocksizes		= new PetscInt* [num_dims];		//the block sizes of the respective dimensions
    blocksizes	= locblocksizes;
    
    for(i=0; i < firstmodedim; i++)						//set the values for the mls dims
    {
      blocksizes_max[i]	= BinomDim(NMLS[0],firstmodedim-1-i);			//the number of blocks is given by the binomial coefficient, which is neat
      blocksizes[i]	= new PetscInt [blocksizes_max[i]];			//allocate the respective number of entries in the blocksizes array
    }
    while( i < num_dims )							//set the values for the mode dims
    {
      blocksizes_max[i]	= modedimlengths[i-firstmodedim];			//fixed number of blocks for the first dim
      blocksizes[i]	= new PetscInt [blocksizes_max[i]];
      i++;
      blocksizes_max[i]	= 1;							//and a single block for the second dim
      blocksizes[i]	= new PetscInt [blocksizes_max[i]];
      i++;
    }
   
   
  /**
   * <hr>
   * <b>Mls indexing</b><br>
   * There will be an explanation elsewhere for this routine. Probably in the appendix of my dissertation.<br>
   */
  
    blocksizes[firstmodedim-1][0] = NMLS[0] + 1;					//for the last mls dimension there is only one block, given by the lowest order binomial coefficient (NMLS+1 over NMLS)

    PetscInt	j,k,count;							                //from the untruncated last blocksize we can iteratively generate the blocksizes of the preceding dimensions.
    for(i=firstmodedim-2; i >= 0 ; i--)						        //for every mls dimenstion starting from the second last
    {
      count = 0;
      for(k=0; k < blocksizes_max[i+1]; k++)					    //take all the blocksizes of the following dimension
      {
          j = 0;
          while( blocksizes[i+1][k]-j > 0 )					            //lower them by one until they are zero
          {
              blocksizes[i][count]	= blocksizes[i+1][k]-j;				//and write all possible results into the current blocksizes dimension
              j++;
              count++;
          }
      }
    }										//you can use PrintBlockSizes to see what this does

    i = firstmodedim;								//now for the mode dimensions
    count = 0;
    while( i < num_dims )							//until we are past the last dimension
    {
	/**
	 * <hr>
	 * <b>compressed diagonal mode storage format:</b><br>
	 * example:<br>
	 * number of offidagonals:	dm = 3<br>
	 * max allowed number state:	mo = 7	--> modedimlengths[count] = blocksizzes_max[count+firstmodedim] = mo+1<br>
	 * <br>
	 * <table>
	 * <tr> <th>x </th> <th>x </th> <th>x </th> <th>00</th> <th>10</th> <th>20</th> <th>30</th> <th>-> 0 + 1 + 3 = 4</th> </tr>
	 * <tr> <th>x </th> <th>x </th> <th>01</th> <th>11</th> <th>21</th> <th>31</th> <th>41</th> <th>-> 1 + 1 + 3 = 5</th> </tr>
	 * <tr> <th>x </th> <th>02</th> <th>12</th> <th>22</th> <th>32</th> <th>42</th> <th>52</th> <th>-> 2 + 1 + 3 = 6</th> </tr>
	 * <tr> <th>03</th> <th>13</th> <th>23</th> <th>33</th> <th>43</th> <th>53</th> <th>63</th> <th>-> 3 + 1 + 3 = 7</th> </tr>
	 * <tr> <th>14</th> <th>24</th> <th>34</th> <th>44</th> <th>54</th> <th>64</th> <th>74</th> <th>-> 3 + 1 + 3 = 7</th> </tr>
	 * <tr> <th>25</th> <th>35</th> <th>45</th> <th>55</th> <th>65</th> <th>75</th> <th>x </th> <th>-> 3 + 1 + 2 = 6</th> </tr>
	 * <tr> <th>36</th> <th>46</th> <th>56</th> <th>66</th> <th>76</th> <th>x </th> <th>x </th> <th>-> 3 + 1 + 1 = 5</th> </tr>
	 * <tr> <th>47</th> <th>57</th> <th>67</th> <th>77</th> <th>x </th> <th>x </th> <th>x </th> <th>-> 3 + 1 + 0 = 4</th> </tr>
	 * </table>
	 *    ==> 8 = mo+1 blocks of varying size<br>
	 * <br>
	 * written in 1d as:	| 00 10 20 30 | 01 11 21 31 41 | 02 12 22 32 42 52 | 03 13 23 33 43 53 63 | 14 24 34 44 54 64 74 | ...<br>
	 * <br>
	 * i.e. 8 blocks of sizes {4,5,6,7,7,6,5,4}, which is stored in blocksizes[i+1] and blocksizes[i] respectively.<br>
	 * <hr>
	 */
	
      for(j=0; j < modedimlengths[count]; j++)					//we have a number of modedimlengths[count] of different blocks
      {
          blocksizes[i][j]	= MIN(j,modedimlengths[count+1]) + 1 + MIN(modedimlengths[count]-1-j,modedimlengths[count+1]);
      }
      i++;
      blocksizes[i][0]	= modedimlengths[count];
      i++;
      count+=2;
    }
    

  //-----------------------------------------------------------------
  // general : indices, blockindices and mode dofs.
  //-----------------------------------------------------------------
    PetscInt		*locblockindices	= new PetscInt [num_dims];	//the blockindex, specifies the position in the blocksizes array
    blockindices	= locblockindices;
    
    PetscInt		*locloc_blockstarters	= new PetscInt [num_dims];	//the local start values of the blockindices
    loc_blockstarters	= locloc_blockstarters;
    
    PetscInt		*locindices		= new PetscInt [num_dims];	//the actual multiindex, in case of mls dimensions these numbers are the quantum numbers
    indices		= locindices;
    
    PetscInt		*locloc_starters	= new PetscInt [num_dims];	//the local start values of the indices
    loc_starters	= locloc_starters;
    
    PetscInt		*locmlsdim_pol		= new PetscInt [firstmodedim];	//logical array that contains the information which dim is density and which is polarization like
    mlsdim_pol		= locmlsdim_pol;
    for(i=0; i < firstmodedim; i++)	mlsdim_pol[i] = mlspol[i];		//set the values
    
    PetscInt		*locmode_dofs		= new PetscInt [modes];		//the number of dofs for each mode, i.e. (mo+1)(2*dm+1) - dm*(dm+1)
    mode_dofs		= locmode_dofs;
    for(i=0; i < modes; i++)	mode_dofs[i] = modedimlengths[2*i]*(2*modedimlengths[2*i+1]+1)-modedimlengths[2*i+1]*(modedimlengths[2*i+1]+1);	//set the values
    
    
  //-----------------------------------------------------------------
  // Before truncation: setup stage
  // every processor still has the whole index set. No parallel layout so far!
  // setup of the quantities belwo is necessary for SetBlockZeros() to work!
  //-----------------------------------------------------------------
    //this is wrong and it works anyway so the entire thing seems fo be obsolete
//    mls_dof        = BinomDim(NMLS[0],num_dims);
    
    total_dof		= mls_dof[0];				                //the total number of dof is the mls_dof
    for(i=0; i < modes; i++)	total_dof *= mode_dofs[i];  	//multiplied with all the mode dofs
    
    loc_end		= total_dof;				//"serial" operation stage
    loc_start		= 0;					//
    
    for(i=0; i < num_dims; i++) loc_blockstarters[i] = 0;
    for(i=0; i < num_dims; i++) loc_starters[i] = 0;
    
    
  //-----------------------------------------------------------------
  // Truncation: first call SetBlockZeros() and then upadate the global system sizes.
  //-----------------------------------------------------------------
    total_dof		= SetBlockZeros(dimlenghts);		//now the truncation procedure of the mls dimensions, return type is the total_dof
    
    PetscInt		check=0;                            //sanity check counter
    mls_dof[0]		= total_dof;				        //the mls_dof is the total_dof ...
    for(i=0; i < modes; i++)
    {
      check += mls_dof[0] % mode_dofs[i];				//little check if this acutally works
      mls_dof[0] /= mode_dofs[i];					    // ... divided by all the mode dofs
    }
    if(check)
    {
      PetscPrintf(PETSC_COMM_WORLD,"Error! Index setup messed up!\n");
    }
}


/**
 * @brief    This constructor initializes the Index class with full support for truncation and uniprocessor use. The parallel layout needs to be set after creating the density matrix using SetParallelLayout()
 *
 * @param    nlevels            the number of levels per mls.
 * @param    nummlsdims         the number of mls dimensions.
 * @param    mlspol             logical array that contains the information which mls dimension is density like and which is polarization like
 * @param    N                  the array for total number of mls for each type
 * @param    dimlenghts         the lenghts of the respective mls dimensions, for truncation
 * @param    n_d_mls            the number of different mls types
 * @param    multimls_start     the index of the first dimension for each mls type, the mls dims are ordered
 * @param    modes              the number of modes
 * @param    modedimlengths     the sizes of the mode dimensions: {mode1order,dmr1,mode2order,dmr2,...}
 *
 */

Index::Index(PetscInt nlevels, PetscInt nummlsdims, PetscInt * mlspol, PetscInt * N, PetscInt * dimlenghts, PetscInt n_d_mls, PetscInt * multimls_start, PetscInt modes, PetscInt *modedimlengths)
{
    PetscInt    i,j;
    
    num_levels        = nlevels;
    num_dims        = nummlsdims+2*modes;                    //total number of dimensions
    firstmodedim    = nummlsdims;                            //the index of the first mode dimension
    N_D_MLS         = n_d_mls;                               //number of different MLS types
    
    PetscInt    *loc_multistart = new PetscInt [N_D_MLS+1];                     //the index of the first dimension of each mls type
    for(i=0; i < N_D_MLS; i++) loc_multistart[i]   = multimls_start[i];         //only needed for multi mls type usage
    loc_multistart[N_D_MLS]     = firstmodedim;                                 //
    multiMLS_start              = loc_multistart;                               //
    
    PetscInt    *loc_nmls = new PetscInt [N_D_MLS];         //the total number of mls for each type
    for(i=0; i < N_D_MLS; i++) loc_nmls[i]     = N[i];      //
    NMLS            = loc_nmls;                             //
    
    PetscInt    *loc_mlsdof = new PetscInt [N_D_MLS];                                                                               //only one type of mls in this case
    loc_mlsdof[0]   = BinomDim(NMLS[0],multiMLS_start[1]);                                                                          //the first number is just the number of dofs
    for(i=1; i < N_D_MLS; i++) loc_mlsdof[i]   = BinomDim(NMLS[i],multiMLS_start[i+1]-multiMLS_start[i])*loc_mlsdof[i-1];           //the following numbers are the dofs of all preceding dims
    mls_dof         = loc_mlsdof;                                                                                                   //these are the untruncated values
    
    PetscInt    *locdim_maxvalues    = new PetscInt [firstmodedim];             //the maximum value of the mls dimensions
    for(i=0; i < N_D_MLS; i++)                                                  //initialize it to NMLS, gets truncated later
    {
        for(j=multiMLS_start[i]; j < multiMLS_start[i+1];j++) locdim_maxvalues[j] = NMLS[i];
    }
    mlsdim_maxvalues            = locdim_maxvalues;
    
    
    //-----------------------------------------------------------------
    //allocate blocks
    //-----------------------------------------------------------------
    PetscInt    *locblocksizes_max    = new PetscInt [num_dims];            //the lengths of the blocksizes arrays
    blocksizes_max            = locblocksizes_max;
    
    PetscInt    **locblocksizes        = new PetscInt* [num_dims];          //the block sizes of the respective dimensions
    blocksizes    = locblocksizes;
    
    //set the values for the mls dims
    for(i=0; i < N_D_MLS; i++)
    {
        for(j=0; j < multiMLS_start[i+1]-multiMLS_start[i];j++)
        {
            blocksizes_max[j+multiMLS_start[i]]       = BinomDim(NMLS[i],multiMLS_start[i+1]-multiMLS_start[i]-1-j);           //the number of blocks is given by the binomial coefficient, which is neat
            blocksizes[j+multiMLS_start[i]]           = new PetscInt [blocksizes_max[j+multiMLS_start[i]]];                    //allocate the respective number of entries in the blocksizes array
        }
    }
    
    i = firstmodedim;
    //set the values for the mode dims
    while( i < num_dims )
    {
        blocksizes_max[i]       = modedimlengths[i-firstmodedim];            //fixed number of blocks for the first dim
        blocksizes[i]           = new PetscInt [blocksizes_max[i]];
        i++;
        blocksizes_max[i]       = 1;                                         //and a single block for the second dim
        blocksizes[i]           = new PetscInt [blocksizes_max[i]];
        i++;
    }
    
    
    /**
     * <hr>
     * <b>Mls indexing</b><br>
     * There will be an explanation elsewhere for this routine. Probably in the appendix of my dissertation.<br>
     */
    
    PetscInt    k,l,count;
    for(i=0; i < N_D_MLS; i++)
    {
        blocksizes[multiMLS_start[i+1]-1][0] = NMLS[i] + 1;                 //for the last mls dimension there is only one block, given by the lowest order binomial coefficient (NMLS+1 over NMLS)
    
        //from the untruncated last blocksize we can iteratively generate the blocksizes of the preceding dimensions.
        for(j=multiMLS_start[i+1]-2; j >= multiMLS_start[i] ; j--)          //for every mls dimenstion starting from the second last
        {
            count = 0;
            for(k=0; k < blocksizes_max[j+1]; k++)                          //take all the blocksizes of the following dimension
            {
                l = 0;
                while( blocksizes[j+1][k]-l > 0 )                           //lower them by one until they are zero
                {
                    blocksizes[j][count]    = blocksizes[j+1][k]-l;         //and write all possible results into the current blocksizes dimension
                    l++;
                    count++;
                }
            }
        }
    }
    
    //you can use PrintBlockSizes to see what this does
    
    i = firstmodedim;                                //now for the mode dimensions
    count = 0;
    while( i < num_dims )                            //until we are past the last dimension
    {
        /**
         * <hr>
         * <b>compressed diagonal mode storage format:</b><br>
         * example:<br>
         * number of offidagonals:    dm = 3<br>
         * max allowed number state:    mo = 7    --> modedimlengths[count] = blocksizzes_max[count+firstmodedim] = mo+1<br>
         * <br>
         * <table>
         * <tr> <th>x </th> <th>x </th> <th>x </th> <th>00</th> <th>10</th> <th>20</th> <th>30</th> <th>-> 0 + 1 + 3 = 4</th> </tr>
         * <tr> <th>x </th> <th>x </th> <th>01</th> <th>11</th> <th>21</th> <th>31</th> <th>41</th> <th>-> 1 + 1 + 3 = 5</th> </tr>
         * <tr> <th>x </th> <th>02</th> <th>12</th> <th>22</th> <th>32</th> <th>42</th> <th>52</th> <th>-> 2 + 1 + 3 = 6</th> </tr>
         * <tr> <th>03</th> <th>13</th> <th>23</th> <th>33</th> <th>43</th> <th>53</th> <th>63</th> <th>-> 3 + 1 + 3 = 7</th> </tr>
         * <tr> <th>14</th> <th>24</th> <th>34</th> <th>44</th> <th>54</th> <th>64</th> <th>74</th> <th>-> 3 + 1 + 3 = 7</th> </tr>
         * <tr> <th>25</th> <th>35</th> <th>45</th> <th>55</th> <th>65</th> <th>75</th> <th>x </th> <th>-> 3 + 1 + 2 = 6</th> </tr>
         * <tr> <th>36</th> <th>46</th> <th>56</th> <th>66</th> <th>76</th> <th>x </th> <th>x </th> <th>-> 3 + 1 + 1 = 5</th> </tr>
         * <tr> <th>47</th> <th>57</th> <th>67</th> <th>77</th> <th>x </th> <th>x </th> <th>x </th> <th>-> 3 + 1 + 0 = 4</th> </tr>
         * </table>
         *    ==> 8 = mo+1 blocks of varying size<br>
         * <br>
         * written in 1d as:    | 00 10 20 30 | 01 11 21 31 41 | 02 12 22 32 42 52 | 03 13 23 33 43 53 63 | 14 24 34 44 54 64 74 | ...<br>
         * <br>
         * i.e. 8 blocks of sizes {4,5,6,7,7,6,5,4}, which is stored in blocksizes[i+1] and blocksizes[i] respectively.<br>
         * <hr>
         */
        
        for(j=0; j < modedimlengths[count]; j++)                    //we have a number of modedimlengths[count] of different blocks
        {
            blocksizes[i][j]    = MIN(j,modedimlengths[count+1]) + 1 + MIN(modedimlengths[count]-1-j,modedimlengths[count+1]);
        }
        i++;
        blocksizes[i][0]    = modedimlengths[count];
        i++;
        count+=2;
    }
    
    
    //-----------------------------------------------------------------
    // general : indices, blockindices and mode dofs.
    //-----------------------------------------------------------------
    PetscInt        *locblockindices    = new PetscInt [num_dims];    //the blockindex, specifies the position in the blocksizes array
    blockindices    = locblockindices;
    
    PetscInt        *locloc_blockstarters    = new PetscInt [num_dims];    //the local start values of the blockindices (MPI)
    loc_blockstarters    = locloc_blockstarters;
    
    PetscInt        *locindices        = new PetscInt [num_dims];    //the actual multiindex, in case of mls dimensions these numbers are the quantum numbers
    indices        = locindices;
    
    PetscInt        *locloc_starters    = new PetscInt [num_dims];    //the local start values of the indices (MPI)
    loc_starters    = locloc_starters;
    
    PetscInt        *locmlsdim_pol        = new PetscInt [firstmodedim];    //logical array that contains the information which dim is density and which is polarization like
    mlsdim_pol        = locmlsdim_pol;
    for(i=0; i < firstmodedim; i++)    mlsdim_pol[i] = mlspol[i];        //set the values
    
    PetscInt        *locmode_dofs        = new PetscInt [modes];        //the number of dofs for each mode, i.e. (mo+1)(2*dm+1) - dm*(dm+1)
    mode_dofs        = locmode_dofs;
    for(i=0; i < modes; i++)    mode_dofs[i] = modedimlengths[2*i]*(2*modedimlengths[2*i+1]+1)-modedimlengths[2*i+1]*(modedimlengths[2*i+1]+1);    //set the values
    
    
    //-----------------------------------------------------------------
    // Before truncation: setup stage
    // every processor still has the whole index set. No parallel layout so far!
    // setup of the quantities below is necessary for SetBlockZeros() to work!
    //-----------------------------------------------------------------
    //this part is wrong in the single mls type index constructor and it works anyway so the entire thing seems to be obsolete
//    mls_dof        = 1;
//    for(i=0; i < N_D_MLS; i++) mls_dof *= BinomDim(NMLS[i],multiMLS_start[i+1]-multiMLS_start[i]);
    
    total_dof        = mls_dof[N_D_MLS-1];                      //the total number of dof is the mls_dof
    for(i=0; i < modes; i++)    total_dof *= mode_dofs[i];      //multiplied with all the mode dofs
    
    loc_end        = total_dof;                //"serial" operation stage
    loc_start        = 0;                    //
    
    for(i=0; i < num_dims; i++) loc_blockstarters[i] = 0;
    for(i=0; i < num_dims; i++) loc_starters[i] = 0;
    
    
    //-----------------------------------------------------------------
    // Truncation: first call SetBlockZeros() and then upadate the global system sizes.
    //-----------------------------------------------------------------
//    total_dof        = SetBlockZeros(dimlenghts);        //now the truncation procedure of the mls dimensions, return type is the total_dof
    
    //this step is a bit redundant here, but it provides a good sanity check
    PetscInt    check = 0;
    mls_dof[N_D_MLS-1]        = total_dof;                //the mls_dof is the total_dof ...
    for(i=0; i < modes; i++)
    {
        check += mls_dof[N_D_MLS-1] % mode_dofs[i];                //little check if this acutally works
        mls_dof[N_D_MLS-1] /= mode_dofs[i];                        // ... divided by all the mode dofs
    }
    if(check)
    {
        PetscPrintf(PETSC_COMM_WORLD,"Error! Index setup messed up!\n");
    }
    
    //count the new mls_dof values
    PetscInt counter = 0;
    PetscInt compare = mls_dof[N_D_MLS-1];
    for(i=0; i < N_D_MLS; i++)
    {
        InitializeGlobal();
        
        while(ContinueMLS(i))
        {
            counter++;
            Increment();
        }
        mls_dof[i] = counter;
        counter = 0;
    }
    
    //sanity check
    check = 0;
    for(i=0; i < N_D_MLS-1; i++)
    {
        check += mls_dof[i+1] % mls_dof[i];         //this should remain zero if everything works fine
    }
    if( compare != mls_dof[N_D_MLS-1] ) check++;   //this should be the same as before
    
    if(check)
    {
        PetscPrintf(PETSC_COMM_WORLD,"Error! Index setup messed up!\n");
    }
}


/**
 * @brief	This is a generic truncation helper function. It recieves an array of the dimlenghts of the dimensions truncates the indexing accordingly.
 * 		This is done by setting all blocksizes[0] elements to zero that correspond to a block that is not allowed. This way all the other functionalities of the Index class can be left unchanged for a truncated density matrix.
 * 
 * @param	dimlenghts	the array containing the lengths of the respective dimensions.
 * 
 */

PetscInt Index::SetBlockZeros(PetscInt* dimlenghts)
{
    PetscInt	i, ret;
    
    for(i=0; i < firstmodedim; i++)	mlsdim_maxvalues[i] = dimlenghts[i]-1;
    
    InitializeGlobal();					    //initialize the index
    
    while( ContinueGlobal() )				//is it possible to continue?
    {
      for(i=0; i < firstmodedim; i++)
      {
          if( indices[i] >= dimlenghts[i] )
          {
              blocksizes[0][blockindices[0]] = 0;	//if a single one of the indices exceeds the allowed dimlength, set the corresponding block size to zero
          }
      }
      Increment();					        //increment the index by one
    }
    
    ret=0;						            //count the new number of mls dofs
    InitializeGlobal();					    //by starting at the beginning
    while( ContinueGlobal() )			    //and doing it the brute force way, too lazy to come up with something fancy
    {
      ret++;
      Increment();
    }
    
    return	ret;					//new number of mls dofs...
}


/**
 * @brief	Default destructor
 * 
 */

Index::~Index()
{
    PetscInt	i;
    for(i=0; i < num_dims; i++)
    {
      delete[] blocksizes[i];
    }
    delete[] blocksizes;
    delete[] mlsdim_maxvalues;
    delete[] mode_dofs;
    delete[] blocksizes_max;
    delete[] blockindices;
    delete[] indices;
    delete[] loc_starters;
    delete[] loc_blockstarters;
    delete[] mlsdim_pol;
    delete[] multiMLS_start;
    delete[] mls_dof;
}


//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  generic helper functions
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

#undef __FUNCT__
#define __FUNCT__ "SetParallelLayout"

/**
 * @brief	Sets the parallel layout of the index.
 * 
 * @param	start		the local start value of dmindex
 * @param	end		one more than the local end of the dmindex
 * 
 */

void Index::SetParallelLayout(PetscInt start, PetscInt end)
{
    PetscInt		i;
    
    SetIndex(start);								//find the first element of the local snippet
    
    loc_start	= dmindex;							//set the local start values
    mls_start	= mlsindex;							//
    
    for(i=0; i < num_dims; i++)	loc_starters[i]		= indices[i];		//
    for(i=0; i < num_dims; i++)	loc_blockstarters[i]	= blockindices[i];	//
    
    loc_end	= end;								//the end value is rather simple...
}

/**
 * @brief	Computes the recurring dimensionality binomial coefficient @f$ {n + order \choose n } @f$
 * 
 */

PetscInt Index::BinomDim(PetscInt n, PetscInt order)
{
    if( order == 0 )
    {
      return	1;
    }
    else if( order == 1 )
    {
      return	n+1;
    }
    else if( n == 0 )
    {
      return	1;
    }
    else if( n == 1 )
    {
      return	order+1;
    }
    else
    {
      return	BinomDim(n,order-1)+BinomDim(n-1,order);
    }
}
