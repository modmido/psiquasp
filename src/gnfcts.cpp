

/**
 * @file	gnfcts.cpp
 * 
 * 		Implementation of the recursive initializing and computing of mls and mode correlation functions of arbitrary order.
 * 
 * @author	Michael Gegg
 * 
 */

#include"../include/gnfcts.hpp"
#include"../include/dim.hpp"

//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  constructors/destructors
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

/**
 * @brief	Default constructor.
 */

Gnfct::Gnfct()
{
    isherm = 1;
    alloc = 0;
    
    PetscInt	**locnums	= new PetscInt* [2];
    numbers			= locnums;
    
    PetscReal	**locfactors	= new PetscReal* [2];
    factors			= locfactors;
    
    PetscInt	*loclengths	= new PetscInt [2];
    lengths			= loclengths;
}


#undef __FUNCT__
#define __FUNCT__ "AllocateLocStorage"

/**
 * @brief	Allocates the storage
 * 
 * @param	count		the number of local dm elements, i.e. how long do the arrays need to be.
 * @param	choose		0 means allocate the numerator , 1 or higher means allocate the denominator 
 * 
 */

PetscErrorCode Gnfct::AllocateLocStorage(PetscInt count,PetscInt choose)
{
    PetscFunctionBeginUser;

    lengths[choose]			= count;
      
    PetscInt	*locnumbers		= new PetscInt [count];
    numbers[choose]			= locnumbers;
    
    PetscReal	*locfactors		= new PetscReal [count];
    factors[choose]			= locfactors;
    
    PetscFunctionReturn(0);
}



/**
 * @brief	Default destructor.
 */

Gnfct::~Gnfct()
{
    PetscInt	i;
    
    for(i=0; i<2; i++)
    {
      delete[]	numbers[i];
      delete[]	factors[i];
    }
    delete[]	numbers;
    delete[]	factors;
    
    delete[]	lengths;
}


//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  compute
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

#undef __FUNCT__
#define __FUNCT__ "Compute"

/**
 * @brief	Compute function for the Gnfct class. Usually called by the monitor function. Only the first processor gets the global result.
 * 
 * @param	dm		    the density matrix.
 * @param	time		the time of the time integration algorithm
 * @param	ret	    	the global return value, only first processor gets it tough...
 * @param	number		not needed here.
 * 
 */

PetscErrorCode Gnfct::Compute(Vec dm, PetscReal time, PetscScalar* ret, PetscInt number)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    PetscInt		i;
    PetscScalar		locnum_local=0.,locdenom_local=0.,locnum_global=0.,locdenom_global=0.,dummy=1.0;
    const PetscScalar	*a;
  
    ierr = VecGetArrayRead(dm,&a);CHKERRQ(ierr);
    
    for(i=0; i < lengths[0]; i++)								//local numerator snippet
    {
      locnum_local	+= factors[0][i]*a[numbers[0][i]];
    }
    
    for(i=0; i < lengths[1]; i++)								//local denominator snippet
    {
      locdenom_local	+= factors[1][i]*a[numbers[1][i]];
    }
    
    ierr = VecRestoreArrayRead(dm,&a);CHKERRQ(ierr);
    
    MPI_Reduce(&locnum_local,&locnum_global,1,MPIU_SCALAR,MPIU_SUM,0,PETSC_COMM_WORLD);		//locnum_global		= <(J_+)^n (J_-)^n>
    MPI_Reduce(&locdenom_local,&locdenom_global,1,MPIU_SCALAR,MPIU_SUM,0,PETSC_COMM_WORLD);	//locdenom_global	= <J_+ J_->
    
    for(i=0; i<order; i++)	dummy *= locdenom_global;					//dummy			= <J_+ J_->^n
    
    *ret	= locnum_global/dummy;								//*global		= <(J_+)^n (J_-)^n> / <J_+ J_->^n
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ComputeAll"

/**
 * @brief	Compute function for the Gnfct class. All processors get the global result.
 * 
 * @param	dm		    the density matrix.
 * @param	time		the time of the time integration algorithm
 * @param	ret 		the global return value, only first processor gets it tough...
 * @param	number		not needed here.
 * 
 */

PetscErrorCode Gnfct::ComputeAll(Vec dm, PetscReal time, PetscScalar* ret, PetscInt number)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    PetscInt		i;
    PetscScalar		locnum_local=0,locdenom_local=0,locnum_global=0,locdenom_global=0,dummy=1;
    const PetscScalar	*a;
  
    ierr = VecGetArrayRead(dm,&a);CHKERRQ(ierr);
    
    for(i=0; i < lengths[0]; i++)								//local numerator snippet
    {
      locnum_local	+= factors[0][i]*a[numbers[0][i]];
    }
    
    for(i=0; i < lengths[1]; i++)								//local denominator snippet
    {
      locdenom_local	+= factors[1][i]*a[numbers[1][i]];
    }
    
    ierr = VecRestoreArrayRead(dm,&a);CHKERRQ(ierr);
    
    MPI_Allreduce(&locnum_local,&locnum_global,1,MPIU_SCALAR,MPIU_SUM,PETSC_COMM_WORLD);		//locnum_global		= <(J_+)^n (J_-)^n>
    MPI_Allreduce(&locdenom_local,&locdenom_global,1,MPIU_SCALAR,MPIU_SUM,PETSC_COMM_WORLD);	//locdenom_global	= <J_+ J_->
    
    for(i=0; i<order; i++)	dummy *= locdenom_global;					//dummy			= <J_+ J_->^n
    
    *ret	= locnum_global/dummy;								//*global		= <(J_+)^n (J_-)^n> / <J_+ J_->^n
    
    PetscFunctionReturn(0);
}



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  mode correlation function 
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

#undef __FUNCT__
#define __FUNCT__ "SetupModeGnfct"

/**
 * @brief	Setup of the nth order correlation function of one of the bosonic modes.
 * 
 * @param	sys		the system specification object.
 * @param	modenumber	the number of the mode, starting from zero in the order in which they were set
 * @param	inorder		the order of the correlation function.
 * 
 */

PetscErrorCode Gnfct::SetupModeGnfct(System* sys, PetscInt modenumber, PetscInt inorder)
{
    PetscFunctionBeginUser;
  
    PetscErrorCode	ierr;
    
    
    //finding the dimensions  
    PetscInt	dim = 0;
    ModeDim	modedim (0,modenumber);
    
    ierr = sys->FindMatch(&modedim,&dim); CHKERRQ(ierr);
    
    
    //basic properties
    order			= inorder;
    real_value_tolerance	= sys->RealValueTolerance();				//hermitian observables need a tolerance for their realvaluedness
    name			= "g^("+ std::to_string(inorder) + ")_ph"+std::to_string(modenumber)+ ">";
    
      
    //how many local dm entries
    PetscInt		locindex,count0 = 0, count1 = 0;
    locindex		= sys->index->InitializeLocal();
      
    while( sys->index->ContinueLocal() )
    {
      if( !sys->index->IsPol() )
      {
	if( sys->index->ModeQN(dim) >= 1 )	count1++; 
	if( sys->index->ModeQN(dim) >= order )	count0++;		//if its a density && n >= m in b^m|n> since otherwise its zero 
      }
      locindex	= sys->index->Increment();
    }
    
    
    //allocate storage
    ierr = AllocateLocStorage(count0,0); CHKERRQ(ierr);
    ierr = AllocateLocStorage(count1,1); CHKERRQ(ierr);
    
    
    //fill it with values
    locindex		= sys->index->InitializeLocal();
    count0		= 0;
    count1		= 0;
      
    while( sys->index->ContinueLocal() )
    {
      if( !sys->index->IsPol() )
      {
	if( sys->index->ModeQN(dim) >= 1 )
	{
	  numbers[1][count1]		= locindex - sys->index->LocStart();
	  factors[1][count1]		= sys->index->ModeQN(dim);
	  count1++;
	}
	if( sys->index->ModeQN(dim) >= order )
	{
	  numbers[0][count0]		= locindex - sys->index->LocStart();
	  factors[0][count0]		= sys->FactorialTrunc(sys->index->ModeQN(dim),order);
	  count0++;
	}
      }
      locindex	= sys->index->Increment();
    }
       
       
    //ouput part
    if(sys->LongOut() || sys->PropOut())
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nMode correlation function initialization: mode %s,  order = %d\n",modedim.ToString().c_str(),inorder); CHKERRQ(ierr);
    }
    
    
    PetscFunctionReturn(0);
}



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  mls correlation function 
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

#undef __FUNCT__
#define __FUNCT__ "SetupMLSGnfct"

/**
 * @brief	Setup the correlation function object.
 * 
 * @param	sys		    the system specification object.
 * @param	inorder		order to the expectation value.
 * @param	destructor	the name of the destructor in the normally ordered expectation value; uniquely determines the quantity.
 * 
 */

PetscErrorCode Gnfct::SetupMLSGnfct(System * sys,MLSDim * destructor,PetscInt inorder)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    
    //basic properties
    order			= inorder;
    real_value_tolerance	= sys->RealValueTolerance();				//hermitian observables need a tolerance for their realvaluedness
    name			= "g^("+ std::to_string(inorder) + ")_MLS>";
    
    
    //numerator
    ierr = MLSNormalorderedExpecationvalue(sys,inorder,destructor,0); CHKERRQ(ierr);		//set numerator
    
    
    //denominator
    ierr = MLSNormalorderedExpecationvalue(sys,1,destructor,1); CHKERRQ(ierr);			//set denominator
    
    
    //ouput part
    if(sys->LongOut() || sys->PropOut())
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nMLS correlation function initialization: destructor %s,  order = %d\n",destructor->ToString().c_str(),inorder); CHKERRQ(ierr);
    }
    
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MLSNormalorderedExpecationvalue"

/**
 * @brief	Compute all local entries and corresponding prefactors for the normally ordered excitonic density expectation value <(J_+)^n (J_-)^n>.
 * 
 * 		First computes all entries using SingleElementMLSNE(), then adds up all factors for same entries and thus cleaning the list using CombineSummands(),
 * 		and then checks whether the entries belong to the local part of the density matrix vector and stores them in the numbers, factors and length pointers.
 * 
 * @param	sys		        the system specification object.
 * @param	order		    the order n of the expectation value.
 * @param	mlspol1_name	the name of the density corresponding to the destructor in the normally ordered expectation value
 * @param	choose		    states whether the function call is for the numerator (choose = 0) or the denominator (choose > 0) in the correlation function
 * 
 */

PetscErrorCode	Gnfct::MLSNormalorderedExpecationvalue(System * sys, PetscInt order, MLSDim * mlspol1_name, PetscInt choose)
{
    PetscFunctionBeginUser;

    PetscErrorCode	ierr;
    
    
    //finding the dimensions  
    PetscInt		dens1=0, dens2=0, pol1 = 0, pol2 = 0;
    MLSDim		    mlspol2_name = mlspol1_name->Swap(*mlspol1_name);		//swap constructor
    MLSDim		    mlsdens1_name (1,*mlspol1_name);
    MLSDim		    mlsdens2_name (0,*mlspol1_name);
    
    ierr = sys->FindMatch(&mlsdens1_name,&dens1); CHKERRQ(ierr);
    ierr = sys->FindMatch(&mlsdens2_name,&dens2); CHKERRQ(ierr);
    ierr = sys->FindMatch(mlspol1_name,&pol1); CHKERRQ(ierr);
    ierr = sys->FindMatch(&mlspol2_name,&pol2); CHKERRQ(ierr);


    //multi mls functionality
    PetscInt    mlstype = mlspol1_name->TypeNumber();
    
    
    //basic properties
    isherm	= 1;								                        //it is an observable that should be real valued.
    
    
    //get all dm entries and prefactors for the normally ordered expectation value
    std::list<Elem*>	*result = new std::list<Elem*>;					//the storage for the resulting elements
    PetscInt		locindex;
    locindex		= sys->index->InitializeGlobal();				    //global start aka mr. groundstate
    
    while ( sys->index->ContinueMLS() )							        //loop over all mls dofs
    {
      if( !sys->index->IsPol() )							            //is it density like?
      {
          std::stack<Elem*>	input;							            //recursion stack
          Elem	*newelem = new Elem (sys->index,order,mlstype);					//start element aka first element of the stack
	  
          input.push(newelem);								            //put it into the stack
	  
          ierr = SingleElementMLSNO(sys,dens1,pol2,pol1,dens2,result,&input);CHKERRQ(ierr);	//do the recursion
      }
      locindex	= sys->index->Increment();
    }

      
    //clean the result
    std::list<Elem*>	*clean = new std::list<Elem*>;					    //storage for the clean list
      
    ierr	= Elem::CombineListElems(clean,result); CHKERRQ(ierr);			//make the clean list
    delete	result;
    
    ierr	= Elem::ComputeIndex(sys,clean); CHKERRQ(ierr);				    //compute each dmindex value and check whether the element exists in the index
      
    clean->sort(Elem::ElemComp);							                //sort the list in ascending order, this way the number of comparisions in the next while loop is minimized
    
    
    //allocate storage and transfer it into the proper form
    PetscInt	count	= 0;
    if( clean->empty() )								                    //if there is nothing here this means we need dummy values..
    {
      ierr = AllocateLocStorage(count,choose); CHKERRQ(ierr);
    }
    else
    {
      locindex	= sys->index->InitializeLocal();				            //start with the first local element
      
      std::list<Elem*>::iterator it	= clean->begin();				        //set the iterator to the first list element
      while( (*it)->dmindex < sys->index->MLSIndex() ) 
      {
          (*it)->PrintIndices();
          it++;										                        //then find the first one that is contained in the local snippet
          if( it == clean->end() )							                //if the first one contained in the local snippet is larger than the last one in the list
          {
              it = clean->begin();							                //then it is again the first one
              break;									                    //and we are done
          }
      }
      while( sys->index->ContinueLocal() )						            //while there are local elements
      {
          if( (*it)->dmindex == sys->index->MLSIndex() && sys->index->IsModeDensity() )	//if the index of the element is equal to the MLSIndex of the Index object then the element should be taken
          {
              count++;									            //so we increase count, i.e. the number of local elements
              it++;										            //and we take the next list element, since also the index goes in ascending order, minimizes number of comparisions!
              if( it == clean->end() )	it = clean->begin();		//when we are at the end of the list we start anew.
          }
	
          sys->index->Increment();
      }
      
      
      //allocate storage
      ierr = AllocateLocStorage(count,choose); CHKERRQ(ierr);
      
      
      //which elements?
      count		= 0;
      locindex		= sys->index->InitializeLocal();				//start with the first local element
      
      it = clean->begin();								            //set the iterator to the first list element
      while( (*it)->dmindex < sys->index->MLSIndex() ) 
      {
          it++;										                //then find the first one that is contained in the local snippet
          if( it == clean->end() )							        //if the first one contained in the local snippet is larger than the last one in the list
          {
              it = clean->begin();								    //then it is again the first one
              break;									            //and we are done
          }
      }
      while( sys->index->ContinueLocal() )						    //while there are local elements
      {
          if( (*it)->dmindex == sys->index->MLSIndex() && sys->index->IsModeDensity() )	//if the index of the element is equal to the MLSIndex of the Index object then the element should be taken, but only if the modes are density like!!!
          {
              numbers[choose][count]	= locindex - sys->index->LocStart();		//store necessary
              factors[choose][count]	= (*it)->factor;				            //
	  
              count++;									            //so we increase count, i.e. the number of local elements
              it++;										            //and we take the next list element, since also the index goes in ascending order, minimizes number of comparisions!
              if( it == clean->end() )	it = clean->begin();		//when we are at the end of the list we start anew.
          }
	
          locindex = sys->index->Increment();
      }
      
      for (it=clean->begin(); it != clean->end(); ++it)			    //delete clean list
      {
          delete *it;
      }
      delete clean;
    }

    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SingleElementMLSNO"

/**
 * @brief	Computes recursively the action of normally ordered mls raising/lowering operators on a diagonal element of the density matrix, i.e. @f$ (J_{xy})^n (J_{yx})^n \hat{P}[...nxx,nxy=0,nyx=0,nyy...] @f$.
 * 		Returns a list containing all the elements that occur through the action of the operators. These elements are given by the Elem class.
 * 		NOTE: dens == n00 makes no sense, there is nothing below n00.
 * 
 * @param	sys		the system specification class.
 * @param	dens		the number of the density dof whose correlation is desired, i.e. nxx. Number here means the positio of the dof in the indices multiindex array.
 * @param	leftpol		the number of the left polarisation associated to the raising operators, i.e. nxy
 * @param	rightpol	the number of the right polarisation associated to the lowering operators, i.e. nyx
 * @param	lowerdens	the number of the lower density dof, i.e. nyy. Can be set to -1 for n00.
 * @param	result		the list of elements occuring due to the action of the operators. This is the function output.
 * @param	input		the stack for the recursion. Also determines the input since at the beginning of the recursion the stack has to have only one entry - the (diagonal) start entry.
 * 
 */

PetscErrorCode Gnfct::SingleElementMLSNO(System * sys, PetscInt dens, PetscInt leftpol, PetscInt rightpol, PetscInt lowerdens, std::list<Elem*> * result, std::stack<Elem*> * input)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    if(!input->empty())								//stack is empty means recursion is finished
    { 
      Elem	loc (*(input->top()));						//copy current top element into loc
      delete	input->top();
      input->pop();								//delete current element from stack
      
      
      if( loc.opactions < loc.order )						//first stage: action of the destruction operators i.e.  J_yx |... nxx ... nxy = 0 ... nyy ...><... nxx ... nyx ... nyy... |  with y < x
      {
          if( loc.indices[dens] > 0 )						//we can ony destroy an excitation if there is one
          {
              Elem	*newelem = new Elem (loc);					//first copy old element
	  
              newelem->indices[dens]--;						//the excitation is destroyed on the left (ket) side of the element nxx -> nxx-1
              newelem->indices[rightpol]++;						//this means that nyx -> nyx+1
	  
              newelem->factor *= newelem->indices[rightpol];			//something like factor *= (nyx+1)
              newelem->opactions++;							//number of operator actions has to be increased
	
              input->push(newelem);							//put it onto the stack
          }
	
          ierr = SingleElementMLSNO(sys,dens,leftpol,rightpol,lowerdens,result,input); CHKERRQ(ierr); 		//commence recursion
      }
      else if( loc.opactions < 2*loc.order )					//second stage: action of the creation operators
      {
          if( loc.indices[rightpol] > 0 )						//creating an excitation on the left side which is already present on the right side
          {
              Elem	*newelem = new Elem (loc);					//first copy old element
	  
              newelem->indices[dens]++;						//the excitation is created on the left (ket) side of the element while it is already present on the right (bra) side
              newelem->indices[rightpol]--;						//this means that nxx -> nxx+1 and nyx -> nyx-1
	  
              newelem->factor *= newelem->indices[dens];				//something like factor *= (nxx+1)
              newelem->opactions++;							//number of operator actions has to be increased

              input->push(newelem);							//put it onto the stack
          }
          if( lowerdens != -1 && loc.indices[lowerdens] > 0 )			//create an excitation on the left which is not present on the right side
          {
              Elem	*newelem = new Elem (loc);					//first copy the old element
	  
              newelem->indices[lowerdens]--;					//the excitation is created on the left (ket) side of the element while it is not present on the right (bra) side
              newelem->indices[leftpol]++;						//this means that nxy -> nxy+1 and nyy -> nyy-1
	  
              newelem->factor *= newelem->indices[leftpol];				//something like factor *= (nxy+1)
              newelem->opactions++;							//number of operator actions has to be increased
	
              input->push(newelem);							//put it onto the stack
          }
          else if ( lowerdens == -1 && loc.n00() > 0 )				//create an excitation on the left which is not present on the right side && lowerdens == n00
          {
              Elem	*newelem = new Elem (loc);					//first copy the old element
	  
              newelem->indices[leftpol]++;						//the excitation is created on the left (ket) side of the element while it is not present on the right (bra) side and nyy is n00 and therefore omitted
	  
              newelem->factor *= newelem->indices[leftpol];				//something like factor *= (nxy+1)
              newelem->opactions++;							//number of operator actions has to be increased
	  
              input->push(newelem);							//put it onto the stack
          }
	
          ierr = SingleElementMLSNO(sys,dens,leftpol,rightpol,lowerdens,result,input); CHKERRQ(ierr); 		//commence recursion
      }
      else									//third stage: all operators have acted, copy into output
      {
          Elem	*newelem = new Elem (loc);					//first copy the old element, with using new!
	
          if( !sys->index->MLSOutOfBounds(newelem->indices) )
          {
              result->push_back(newelem);						//put it into the output list
          }
          else
          {
              (*PetscErrorPrintf)("Error: Elements of correlation function are not contained in the used density matrix boundaries!\n");				//means that we messed up something
              (*PetscErrorPrintf)("Either increase system size/number of offdiagonals or reduce order of the correlation function!\n");				//means that we messed up something
              SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");													//so we abort
          }
	
          ierr = SingleElementMLSNO(sys,dens,leftpol,rightpol,lowerdens,result,input); CHKERRQ(ierr); 								//commence recursion
      }
    }
    PetscFunctionReturn(0);
}




