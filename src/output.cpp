

/**
 * @file	output.cpp
 * 
 * 		Contains member function definitions of the classes defined in the output.hpp file.
 * 
 * @author	Michael Gegg
 * 
 */

#include"../include/output.hpp"
#include"../include/distributions.hpp"
#include"../include/dim.hpp"



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  OFile member functions
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

/**
 * @brief	Virtual default destructor. Calls the desturctors of all the objects in the list.
 * 
 */

OFile::~OFile()
{
    for (std::list<PropBase*>::iterator it=elements.begin(); it!=elements.end() ; ++it)
    {
      delete *it;
    }

    PetscFClose(PETSC_COMM_WORLD,file);		//no error checking since destructor has no return type and this will only be called after everything is done.
}


#undef __FUNCT__
#define __FUNCT__ "SetOFile"

/**
 * @brief	Open the file in Petsc style using the PETSC_COMM_WORLD MPI communicator for parallel output.
 * 
 * @param	sys		pointer to the user specified System (polymorphic).
 * @param	filename	name of the file.
 * 
 */

PetscErrorCode OFile::SetOFile(System* sys, std::string filename)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    fname = filename;
    ierr = PetscFOpen(PETSC_COMM_WORLD,filename.c_str(),"w",&file); CHKERRQ(ierr);		//seems like petsc wants a FILE ** pointypointer
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddElem"

/**
 * @brief	Add Element to OFile. Element here means a child of PropBase
 * 
 * @param	elem		pointer to the element (observable, distribution, gnfct,...) to be added (polymorphic).
 * @param	name		name of the element (obs, dist, etc. )
 * 
 */

PetscErrorCode OFile::AddElem(PropBase* elem, std::string name)
{
    PetscFunctionBeginUser;
    
    elements.push_back(elem);	//put element at the end of the list
    names.push_back(name);	//as well as the name
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "WriteSystemParameters"

/**
 * @brief	Write system parameters to the output file (as a header).
 * 
 * @param	sys		pointer to the user specified System (polymorphic).
 * 
 */

PetscErrorCode OFile::WriteSystemParameters(System * sys)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    PetscInt		i;
    PetscInt		dummy = 0;
    PetscReal		rummy = 0.0;
    
    
  //-------------------------------------------------------------------------------------
  //general information
  //-------------------------------------------------------------------------------------
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"#System: %d %d-level-systems, %d mls dofs, %d modes\n\n",sys->NMls(),sys->NLevels(),sys->NumMlsdims(),sys->NumModes());CHKERRQ(ierr);
    
    
  //-------------------------------------------------------------------------------------
  //mls information
  //-------------------------------------------------------------------------------------
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"#Mls dofs and maxvalues:  ");CHKERRQ(ierr);						//mls dofs
    for(i=0; i < sys->NumMlsdims(); i++)
    {
      ierr = sys->MLSDimMaxVal(i,&dummy); CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"%s: %d,\t",sys->DimName(i).c_str(), dummy);CHKERRQ(ierr);
    }
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"\n");CHKERRQ(ierr);
    
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"#Mls engergy levels: n00 = 0.0 (default),\t");CHKERRQ(ierr);				//mls energies
    for(i=0; i < sys->NumMlsdims(); i++)
    {
      ierr = sys->IsMLSDimPol(i,&dummy); CHKERRQ(ierr);
      if( !dummy )
      {
	ierr = sys->Energies(i,&rummy); CHKERRQ(ierr);
	ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"%s = %f,\t",sys->DimName(i).c_str(),(float) rummy);CHKERRQ(ierr);
      }
    }
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"\n\n");CHKERRQ(ierr);
    
  //-------------------------------------------------------------------------------------
  //mode information
  //-------------------------------------------------------------------------------------
    i = 0;
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"#Mode dofs: ");CHKERRQ(ierr);								//mode dofs
    while( i < sys->NumModes() )
    {
      ierr = sys->ModeDimLen(2*i,&dummy); CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"maxvalue %s: %d",sys->DimName(2*i + sys->NumMlsdims()).c_str(),dummy-1);CHKERRQ(ierr);
      ierr = sys->ModeDimLen(2*i+1,&dummy); CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_WORLD,file,", no of offdiags %s: %d\t",sys->DimName(2*i+1+sys->NumMlsdims()).c_str(),dummy);CHKERRQ(ierr);
      i++;
    }
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"\n");CHKERRQ(ierr);
    
    i = sys->NumMlsdims();
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"#Mode elementary excitation energies: ");CHKERRQ(ierr);					//mode energies
    while( i < sys->NumDims() )
    {
      ierr = sys->Energies(i,&rummy); CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"%s = %f,\t",sys->DimName(i).c_str(),(float) rummy);CHKERRQ(ierr);
      i += 2;
    }
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"\n\n");CHKERRQ(ierr);
    
    
  //-------------------------------------------------------------------------------------
  //system specific simulation parameters
  //-------------------------------------------------------------------------------------
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"#Simulation parameters:\n#");CHKERRQ(ierr);						//simulation parameters
    for(i=0; i < sys->NumParams(); i++)
    {
      ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"%s = %e",sys->PName(i).c_str(),(float) PetscRealPart(sys->PValue(i)));CHKERRQ(ierr);
      if( PetscImaginaryPart(sys->PValue(i)) > 0.0 )
      {
	ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"+i*%e",(float) PetscImaginaryPart(sys->PValue(i)));CHKERRQ(ierr);
      }
      ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"   ");CHKERRQ(ierr);
    }
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"\n\n");CHKERRQ(ierr);
    

  //-------------------------------------------------------------------------------------
  //system specific simulation parameters
  //-------------------------------------------------------------------------------------
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"#Convergence parameters:\n#");CHKERRQ(ierr);						//simulation parameters
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"real_value_tolerance = %e, hermitian_tolerance = %e",(double) sys->RealValueTolerance(),(double) sys->HermitianTolerance());CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"\n\n");CHKERRQ(ierr);
    
    
    PetscFunctionReturn(0);
}


//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  PropFile class member functions
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

#undef __FUNCT__
#define __FUNCT__ "MakeHeaderTEV"

/**
 * @brief	Makes a header line explaining the output. Prints #time first and then the names list
 * 
 */

PetscErrorCode	PropFile::MakeHeaderTEV()
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"#time\t\t");CHKERRQ(ierr);
    
    for (std::list<std::string>::iterator it=names.begin(); it!=names.end() ; ++it)
    {
      ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"#%s\t\t",(*it).c_str());CHKERRQ(ierr);
    }
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"\n");CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MakeHeaderGen"

/**
 * @brief	Makes a header line explaining the output. Prints #var and then the names list.
 * 
 * @param	var	the name of the parameter to be varied in the output file
 * 
 */

PetscErrorCode	PropFile::MakeHeaderGen(std::string var)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    std::string		out = "#" + var;
    
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"%s\t\t",out.c_str());CHKERRQ(ierr);
    
    for (std::list<std::string>::iterator it=names.begin(); it!=names.end() ; ++it)
    {
      ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"#%s\t\t",(*it).c_str());CHKERRQ(ierr);
    }
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"\n");CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PrintLine"

/**
 * @brief	Prints a single line into the output file by iterating through the Observables list and computing the corresponding values. Does a real value check for hermitian operator expectation values.
 * 
 * @param	dm	the density matrix.
 * @param	time	the current integration time of the solver.
 * 
 */

PetscErrorCode	PropFile::PrintLine(Vec dm, PetscReal time)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    PetscScalar		dummy;
    
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"%.5e\t",(double) time);CHKERRQ(ierr);
    
    for (std::list<PropBase*>::iterator it=elements.begin(); it!=elements.end() ; ++it)				//iterate through the list
    {
      ierr = (*it)->Compute(dm,time,&dummy,0); CHKERRQ(ierr);							//compute each observables' value
      
      if( (*it)->IsHerm() )											//check hermitian observables for real valuedness
      {
	ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"%.12e\t",(double) PetscRealPart(dummy) );CHKERRQ(ierr);	//print it
	if(  fabs( (double) PetscImaginaryPart(dummy) ) > (*it)->RealValueTolerance() )				//of course it comes with a tolerance
	{
	  ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"\n");CHKERRQ(ierr);					//line ends with a newline character, even if it crashes, so that you can acutally see something useful in the output file
	  (*PetscErrorPrintf)("Imaginary part of %s is too large.\n",(*it)->Name().c_str());			//petsc error handling 
	  (*PetscErrorPrintf)("Tolerance value= %.10e\t Im[%s]= %.10e\n",(double) (*it)->RealValueTolerance(),(*it)->Name().c_str(),(double) PetscImaginaryPart(dummy) );
	  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MAX_VALUE,"");
	}
      }
      else
      {
	ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"%.12e\t%.12e\t",(double) PetscRealPart(dummy), (double) PetscImaginaryPart(dummy) );CHKERRQ(ierr);	//print it
      }
    }
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"\n");CHKERRQ(ierr);						//line ends with a newline character
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PrintLineSteady"

/**
 * @brief	Prints a single line into the output file by iterating through the Observables list and computing the corresponding values. Does a real value check for hermitian operator expectation values.
 * 		This function is for steady state computations.
 * 
 * @param	dm	the density matrix.
 * @param	param	the current value of the parameter that is varied.
 * 
 */

PetscErrorCode	PropFile::PrintLineSteady(Vec dm, PetscReal param)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    PetscScalar		dummy;
    
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"%.5e\t",(double) param);CHKERRQ(ierr);
    
    for (std::list<PropBase*>::iterator it=elements.begin(); it!=elements.end() ; ++it)				//iterate through the list
    {
      ierr = (*it)->Compute(dm,0,&dummy,0); CHKERRQ(ierr);							//compute each observables' value
      
      if( (*it)->IsHerm() )											//check hermitian observables for real valuedness
      {
	ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"%.12e\t",(double) PetscRealPart(dummy) );CHKERRQ(ierr);	//print it
	if(  fabs( (double) PetscImaginaryPart(dummy) ) > (*it)->RealValueTolerance() )				//of course it comes with a tolerance
	{
	  ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"\n");CHKERRQ(ierr);					//line ends with a newline character, even if it crashes, so that you can acutally see something useful in the output file
	  (*PetscErrorPrintf)("Imaginary part of %s is too large.\n",(*it)->Name().c_str());			//petsc error handling 
	  (*PetscErrorPrintf)("Tolerance value= %.10e\t Im[%s]= %.10e\n",(double) (*it)->RealValueTolerance(),(*it)->Name().c_str(),(double) PetscImaginaryPart(dummy) );
	  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MAX_VALUE,"");
	}
      }
      else
      {
	ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"%.12e\t%.12e\t",(double) PetscRealPart(dummy), (double) PetscImaginaryPart(dummy) );CHKERRQ(ierr);	//print it
      }
    }
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"\n");CHKERRQ(ierr);						//line ends with a newline character
    
    PetscFunctionReturn(0);
}


//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  DistFile class member functions
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

#undef __FUNCT__
#define __FUNCT__ "PrintLine"

/**
 * @brief	Print a single line into the distributions output file.
 * 
 * @param	dm	the density matrix vector.
 * @param	time	the current time of the integration algorithm
 * 
 */

PetscErrorCode DistFile::PrintLine(Vec dm, PetscReal time)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    PetscInt		i;
    PetscScalar		dummy;
    
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"%.5e\t",(double) time);CHKERRQ(ierr);
    
    std::list<PropBase*>::iterator	it=elements.begin();
    
    for(i = 0; i < length; i++ )
    {
      ierr = (*it)->Compute(dm,time,&dummy,i); CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"%.10Le\t",(long double) PetscRealPart(dummy));CHKERRQ(ierr);	//print it
      
      if( (*it)->IsHerm() )											//check hermitian observables for real valuedness
      {
	if(  fabs( (double) PetscImaginaryPart(dummy) ) > (*it)->RealValueTolerance() )				//of course it comes with a tolerance
	{
	  ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"\n");CHKERRQ(ierr);					//line ends with a newline character, even if it crashes, so that you can acutally see something useful in the output file
	  (*PetscErrorPrintf)("Imaginary part of distribution %s, number state %d is too large.\n",(*it)->Name().c_str(),i);					//petsc error handling 
	  (*PetscErrorPrintf)("Tolerance value= %.10e\t Im[<|%d><%d|>]= %.10e\n",(double) (*it)->RealValueTolerance(),i,i,(double) PetscImaginaryPart(dummy) );
	  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MAX_VALUE,"");
	}
      }
    }
    
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"\n");CHKERRQ(ierr);	//end of line
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PrintLineSteady"

/**
 * @brief	Print a single line into the distributions output file, for steady state calculations.
 * 
 * @param	dm	the density matrix vector.
 * @param	param	the current value of the parameter that is varied.
 * 
 */

PetscErrorCode DistFile::PrintLineSteady(Vec dm, PetscReal param)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    PetscInt		i;
    PetscScalar		dummy;
    
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"%.5e\t",(double) param);CHKERRQ(ierr);
    
    std::list<PropBase*>::iterator	it=elements.begin();
    
    for(i = 0; i < length; i++ )
    {
      ierr = (*it)->Compute(dm,0,&dummy,i); CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"%.10Le\t",(long double) PetscRealPart(dummy));CHKERRQ(ierr);	//print it
      
      if( (*it)->IsHerm() )											//check hermitian observables for real valuedness
      {
	if(  fabs( (double) PetscImaginaryPart(dummy) ) > (*it)->RealValueTolerance() )				//of course it comes with a tolerance
	{
	  ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"\n");CHKERRQ(ierr);					//line ends with a newline character, even if it crashes, so that you can acutally see something useful in the output file
	  (*PetscErrorPrintf)("Imaginary part of distribution %s, number state %d is too large.\n",(*it)->Name().c_str(),i);					//petsc error handling 
	  (*PetscErrorPrintf)("Tolerance value= %.10e\t Im[<|%d><%d|>]= %.10e\n",(double) (*it)->RealValueTolerance(),i,i,(double) PetscImaginaryPart(dummy) );
	  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MAX_VALUE,"");
	}
      }
    }
    
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"\n");CHKERRQ(ierr);	//end of line
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MakeHeaderTEV"

/**
 * @brief	Make generic header for the distributions files for time evolution
 * 
 */

PetscErrorCode DistFile::MakeHeaderTEV()
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    PetscInt		i;
    
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file, "#time\t\t"); CHKERRQ(ierr);
    for(i=0; i<=7; i++) 
    {
      ierr = PetscFPrintf(PETSC_COMM_WORLD,file, "#<|%d><%d|>\t\t", i,i);CHKERRQ(ierr);
    }
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file, "...\n");CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MakeHeaderGen"

/**
 * @brief	Make generic header for the distributions files for some other parameter to be changed
 * 
 */

PetscErrorCode DistFile::MakeHeaderGen(std::string var)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    PetscInt		i;
    std::string		out = "#" + var;
    
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"%s\t\t",out.c_str());CHKERRQ(ierr);
    
    for(i=0; i<=7; i++) 
    {
      ierr = PetscFPrintf(PETSC_COMM_WORLD,file, "#<|%d><%d|>\t\t", i,i);CHKERRQ(ierr);
    }
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file, "...\n");CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SetupMLSDistFile"

/**
 * @brief	Setup for the mls distribution file. This function has return type PetscErrorCode for Petsc error handling, which is why I prefer this instead of a constructor.
 * 
 * @param	system		the system specification object (polymorphic). Needed mainly for the header.
 * @param	filename	the filename
 * @param	mlsdens		the name of the density.
 * 
 */

PetscErrorCode DistFile::SetupMLSDistFile(System * system, std::string filename, MLSDim * mlsdens)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
  //-------------------------------------------------------------------------------------
  //open file, make gen. header
  //-------------------------------------------------------------------------------------
    ierr = SetOFile(system,filename); CHKERRQ(ierr);		//Open File, set file name
    ierr = WriteSystemParameters(system); CHKERRQ(ierr);	//write system parameters into the output file
    
    
    Distribution	*mlsdist = new Distribution();
    
    ierr = mlsdist->SetupMLSDensityDistribution(system,mlsdens); CHKERRQ(ierr);
    
    length = mlsdist->PrintTotalNum();
    
    ierr = AddElem(mlsdist,mlsdens->ToString() + " distribution"); CHKERRQ(ierr);
    
    ierr = MakeHeaderTEV(); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SetupMLSOffdiagDistFile"

/**
 * @brief	Setup for the mls distribution file. This function has return type PetscErrorCode for Petsc error handling, which is why I prefer this instead of a constructor.
 * 
 * @param	system		the system specification object (polymorphic). Needed mainly for the header.
 * @param	filename	the name of the file
 * @param	mlspol		the name of the density
 * @param	number		the order of the offdiagonal
 * 
 */

PetscErrorCode DistFile::SetupMLSOffdiagDistFile(System * system, std::string filename, MLSDim * mlspol, PetscInt number)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    //open file, make gen. header
    std::ostringstream	convert;
    convert << number;
    
    ierr = SetOFile(system,filename); CHKERRQ(ierr);					//Open File, set file name
    ierr = WriteSystemParameters(system); CHKERRQ(ierr);				//write system parameters into the output file
    
    
    Distribution	*mlsdist = new Distribution();
    
    ierr = mlsdist->SetupMLSOffdiagDistribution(system,mlspol,number); CHKERRQ(ierr);
    
    length = mlsdist->PrintTotalNum();
    
    ierr = AddElem(mlsdist,mlspol->ToString() + " offdiag " + convert.str() + " distribution"); CHKERRQ(ierr);
    
    ierr = MakeHeaderTEV(); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SetupModeDistFile"

/**
 * @brief	Setup for the mode distribution file. This function has return type PetscErrorCode for Petsc error handling, which is why I prefer this instead of a constructor.
 * 
 * @param	system		    the system specification object (polymorphic). Needed mainly for the header.
 * @param	modenumber		the name of the mode.
 * 
 */

PetscErrorCode DistFile::SetupModeDistFile(System * system, std::string name, PetscInt modenumber)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    std::ostringstream	convert;
    convert << modenumber;
    
    ierr = SetOFile(system,name); CHKERRQ(ierr);		//Open File, set file name
    ierr = WriteSystemParameters(system); CHKERRQ(ierr);	//write system parameters into the output file
    
    
    Distribution	*modedist = new Distribution();
    
    ierr = modedist->SetupModeDistribution(system,modenumber); CHKERRQ(ierr);
    
    length = modedist->PrintTotalNum();
    
    ierr = AddElem(modedist,convert.str() + " distribution"); CHKERRQ(ierr);
    
    ierr = MakeHeaderTEV(); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SetupMLSDistFile"

/**
 * @brief	Setup for the mls distribution file. This function has return type PetscErrorCode for Petsc error handling, which is why I prefer this instead of a constructor.
 * 
 * @param	system		the system specification object (polymorphic). Needed mainly for the header.
 * @param	filename	the name of the file
 * @param	mlsdens		the name of the density.
 * @param	var		the name of the parameter to be changed in the output file
 * 
 */

PetscErrorCode DistFile::SetupMLSDistFile(System * system, std::string filename, MLSDim * mlsdens, std::string var)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
  //-------------------------------------------------------------------------------------
  //open file, make gen. header
  //-------------------------------------------------------------------------------------
    ierr = SetOFile(system,filename); CHKERRQ(ierr);		//Open File, set file name
    ierr = WriteSystemParameters(system); CHKERRQ(ierr);	//write system parameters into the output file
    
    
    Distribution	*mlsdist = new Distribution();
    
    ierr = mlsdist->SetupMLSDensityDistribution(system,mlsdens); CHKERRQ(ierr);
    
    length = mlsdist->PrintTotalNum();
    
    ierr = AddElem(mlsdist,mlsdens->ToString() + " distribution"); CHKERRQ(ierr);
    
    ierr = MakeHeaderGen(var); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SetupMLSOffdiagDistFile"

/**
 * @brief	Setup for the mls distribution file. This function has return type PetscErrorCode for Petsc error handling, which is why I prefer this instead of a constructor.
 * 
 * @param	system		the system specification object (polymorphic). Needed mainly for the header.
 * @param	filename	the name of the file
 * @param	mlspol		the name of the polarization/offdiagonal
 * @param	var		the name of the parameter to be changed in the output file
 * 
 */

PetscErrorCode DistFile::SetupMLSOffdiagDistFile(System * system, std::string filename, MLSDim * mlspol, PetscInt number, std::string var)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
  //-------------------------------------------------------------------------------------
  //open file, make gen. header
  //-------------------------------------------------------------------------------------
    std::ostringstream	convert;
    convert << number;
    
    ierr = SetOFile(system,filename); CHKERRQ(ierr);					//Open File, set file name
    ierr = WriteSystemParameters(system); CHKERRQ(ierr);				//write system parameters into the output file
    
    
    Distribution	*mlsdist = new Distribution();
    
    ierr = mlsdist->SetupMLSOffdiagDistribution(system,mlspol,number); CHKERRQ(ierr);
    
    length = mlsdist->PrintTotalNum();
    
    ierr = AddElem(mlsdist,mlspol->ToString() + " offdiag " + convert.str() + " distribution"); CHKERRQ(ierr);
    
    ierr = MakeHeaderGen(var); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SetupModeDistFile"

/**
 * @brief	Setup for the mode distribution file. This function has return type PetscErrorCode for Petsc error handling, which is why I prefer this instead of a constructor.
 * 
 * @param	system		the system specification object (polymorphic). Needed mainly for the header.
 * @param	filename	the name of the file
 * @param	modenumber	the number of the mode.
 * @param	var		the name of the parameter to be changed in the output file
 * 
 */

PetscErrorCode DistFile::SetupModeDistFile(System * system, std::string filename, PetscInt modenumber, std::string var)
{
    PetscFunctionBeginUser;
    
    PetscErrorCode	ierr;
    
    std::ostringstream	convert;
    convert << modenumber;
    
    ierr = SetOFile(system,filename); CHKERRQ(ierr);		//Open File, set file name
    ierr = WriteSystemParameters(system); CHKERRQ(ierr);	//write system parameters into the output file
    
    Distribution	*modedist = new Distribution();
    
    ierr = modedist->SetupModeDistribution(system,modenumber); CHKERRQ(ierr);
    
    length = modedist->PrintTotalNum();
    
    ierr = AddElem(modedist,"mode " + convert.str() + " distribution"); CHKERRQ(ierr);
    
    ierr = MakeHeaderGen(var); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  Output member functions
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------


/**
 * @brief	Constructor, only sets the tev_steps_monitor variable
 */

Output::Output()
{
    PetscBool		flg;
    PetscInt		value;
    
    tev_steps_monitor	= 30;		//default value
    
    PetscOptionsHasName(NULL,NULL,"-tev_steps_monitor",&flg);
    if(flg)
    {
      PetscOptionsGetInt(NULL,NULL,"-tev_steps_monitor",&value,NULL);
      tev_steps_monitor = value;
    }
}


/**
 * @brief	Virtual default destructor. Calls the virtual destructors of the all the output files in the list.
 * 
 */

Output::~Output()
{
    for (std::list<OFile*>::iterator it=files.begin(); it!=files.end() ; ++it)
    {
      delete (*it);
    }
}


#undef __FUNCT__
#define __FUNCT__ "AddOFile"

/**
 * @brief	Adds an output file object to the output file list.
 * 
 * @param	newfile		pointer to the file object (polymorphic). Can be either ObsFile, DistFile or some other user derived class objects.
 * 
 */

PetscErrorCode	Output::AddOFile(OFile * newfile)
{
    PetscFunctionBeginUser;
    
    files.push_back(newfile);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TEVMonitor"

/**
 * @brief	The generic monitor function which is given to the Petsc TS solver. Just iterates through the OFile list and prints a line each with PrintLine().
 * 
 * @param	ts		the Petsc time stepper context.
 * @param	steps		the number of the current time step.
 * @param	time		the integration time of the solver.
 * @param	dm		the density matrix.
 * @param	ptr		void pointer to the output object, a bit of a petsc hack for c++
 * 
 */

PetscErrorCode Output::TEVMonitor(TS ts,PetscInt steps,PetscReal time,Vec dm,void* ptr)
{
    PetscFunctionBeginUser;
    
    Output * output	= (Output*) ptr;
    
    if( steps%output->Steps() == 0 )				//print only every 25th integration step, so save memory
    {
      for (std::list<OFile*>::iterator it=output->files.begin(); it!=output->files.end() ; ++it)
      {
	OFile * temp = *it;
	temp->PrintLine(dm,time);
      }
    }
    
    PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "SteadyStateMonitor"

/**
 * @brief	Monitor function for steady state calculations. Remaining unitary time dependencies of observables, e.g. in a rotating frame description is disregarded, i.e. it is set time=0 in all these cases.
 * 
 * @param	param		can be any real valued parameter that has been altered. This is somewhat a hack, output for explicitly time dependent (exponential like) quantities is not meaningful, but thats not so important.
 * @param	dm		the density matrix.
 * 
 */

PetscErrorCode Output::SteadyStateMonitor(PetscReal param,Vec dm)
{
    PetscFunctionBeginUser;
    
    for (std::list<OFile*>::iterator it=files.begin(); it!=files.end() ; ++it)
    {
      (*it)->PrintLineSteady(dm,param);
    }
    
    PetscFunctionReturn(0);
}

