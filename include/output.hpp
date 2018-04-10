

/**
 * @file	output.hpp
 * 
 * Header file for the program ouput utilities.<br>
 * <b>Includes: </b>
 * - PropBase class, which is the base class for all printable system properties, like observables, correlation functions and distributions
 * - OFile class, which is the base class for all output files.
 * - Two child classes of OFile: DistFile for Distribution and custom DModular objects and PropFile for Observable, GnFct and custom PModular objects.
 * - Output class, which is the base class for the whole program output, user needs to implement a child class and specify 
 * 
 * @author	Michael Gegg
 * 
 */
 
#ifndef _Output
#define _Output

#include"system.hpp"

//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  PropBase class
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

/**
 * @brief	Abstract base class for the system properties, like observables, correlation functions and distributions
 * 
 */

class PropBase
{
  protected:
    std::string	name;					//!< name of the quantity
    PetscInt	isherm;					//!< isherm: 0 if it should not be real valued, 1 if it should
    PetscInt	alloc;					//!< alloc: 0 if its not allocated yet, 1 if yes
    PetscReal	real_value_tolerance;			//!< cutoff value for the maximum value of an imaginary part of an hermitian observable. is checked when output is printed
    
  public:
    PropBase()	{ real_value_tolerance = 1.e-10; name = "noname"; }
    virtual	~PropBase() { }										//!< empty, virtual destructor, seems to work, get specified in the derived classes
    
    PetscInt		IsHerm() { return isherm; }							//!< return isherm
    std::string		Name() { return name; }								//!< return name
    PetscReal		RealValueTolerance() { return real_value_tolerance; }				//!< return the real_value_tolerance 
    
    virtual	PetscErrorCode	Compute (Vec dm,PetscReal time,PetscScalar * ret,PetscInt number) =0;		//!< pure virtual compute function, only rank=0 process gets the answer
    virtual	PetscErrorCode	ComputeAll (Vec dm,PetscReal time,PetscScalar * ret,PetscInt number) =0;	//!< pure virtual compute function, every process gets the answer
};



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  OFile and children
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------


/**
 * @brief	Abstract base class for output files. The idea is to create a single object per output file which then can be used with obj.PrintLine() in order to create output
 * 
 */

class OFile		
{
  protected:
    std::list<PropBase*>	elements;			//!< The elements of the OFile. In case of observables and gnfcts there are generally a few, in case of distributions there is (should/makes sense) only one element. 
    std::list<std::string>	names;				//!< The names of the respective quantities, as written in the output file
    std::string			    fname;				//!< the name of the output file to be generated
    FILE			        *file;				//!< the FILE pointer
    
  public:
    virtual ~OFile();								//make it virtual so that derived file classes can have a desructor as well
    PetscErrorCode	SetOFile(System * sys,std::string filename);		//open file, set fname
    PetscErrorCode	WriteSystemParameters(System * sys);			//write the system parameters to the output file
    PetscErrorCode	AddElem(PropBase * elem, std::string propname);		//add a new element to the list, set name
 
    virtual PetscErrorCode	MakeHeaderTEV() =0;				//!< Pure virtual function for writing the file header for time evolution
    virtual PetscErrorCode	MakeHeaderGen(std::string var) =0;		//!< Pure virtual function for writing the a more generic file header
    virtual PetscErrorCode	PrintLine(Vec dm, PetscReal time) =0;		//!< Pure virtual function for printing a single line into the file
    virtual PetscErrorCode	PrintLineSteady(Vec dm, PetscReal param) =0;	//!< Pure virtual function for printing a single line into the file for steady state calculations
};


/**
 * @brief	Properties output file. Child of OFile. Used for standard Observable, Gnfct and custom made PModular objects. 
 * 		Specifies the virtual functions MakeHeader() and PrintLine() for specific use in observable output files. Needs to be further specified by the user through the use of a child class (cf. MyObsFile in mylevel.hpp)
 * 
 */

class PropFile: public OFile
{
  public:
    PetscErrorCode	MakeHeaderTEV();				//print the names of the Observables into a single line of the output file with the time as the first argument
    PetscErrorCode	MakeHeaderGen(std::string var);			//print the names of the Observables into a single line of the output file with var as the first argument
    PetscErrorCode	PrintLine(Vec dm, PetscReal time);		//print a single line into the output file, i.e. the time and the values of the observables
    PetscErrorCode	PrintLineSteady(Vec dm, PetscReal param);	//print a single line into the output file for steady state calculations
};


/**
 * @brief	Distribution output file. Child of OFile. Can be used as such.
 * 		Sets the length parameter, which is the number of different number states in the distribution. Sets some other functions like MakeHeader() and PrintLine() and also SetupMLSDistFile() and SetupModeDistFile().
 * 		For more complicated/problem dependent distributions, like dressed states, please use the DModular class
 * 
 */

class DistFile: public OFile
{
  protected:
    PetscInt	length;					//!< the number of different number states
    
  public:
    PetscErrorCode	MakeHeaderTEV();
    PetscErrorCode	MakeHeaderGen(std::string var);
    PetscErrorCode	PrintLine(Vec dm, PetscReal time);
    PetscErrorCode	PrintLineSteady(Vec dm, PetscReal param);
    
    PetscErrorCode	SetupMLSDistFile(System * system, std::string filename, MLSDim * mlsdens);
    PetscErrorCode	SetupMLSOffdiagDistFile(System * system, std::string filename, MLSDim * mlspol, PetscInt number);
    PetscErrorCode	SetupModeDistFile(System * system, std::string filename, PetscInt modenumber);

    PetscErrorCode	SetupMLSDistFile(System * system, std::string filename, MLSDim * mlsdens, std::string var);
    PetscErrorCode	SetupMLSOffdiagDistFile(System* system, std::string filename, MLSDim * mlspol, PetscInt number, std::string var);
    PetscErrorCode	SetupModeDistFile(System * system, std::string filename, PetscInt modename, std::string var);
};


//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  Output
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------


/**
 * @brief	Base class for the whole program output, needs to be specified for the specific usage.
 * 		Contains the files list which contains all output file ojbects and two methods AddOFile which adds a file to the list and GenMonitor which prints a single line to all the output files.
 * 
 */

class Output
{
  protected:
    PetscInt		    tev_steps_monitor;				//!< print every ... time step into the ouput file in case of time evolution
    std::list<OFile*>	files;						    //!< All output files of the program
    
  public:
    virtual ~Output();
    Output();
    PetscErrorCode		    AddOFile(OFile * newfile);
    static PetscErrorCode	TEVMonitor(TS ts,PetscInt n,PetscReal time,Vec dm,void* out);		//print a single line to each of the files
    PetscErrorCode		    SteadyStateMonitor(PetscReal param,Vec dm);
    
    PetscInt			Steps() { return tev_steps_monitor; }					//return the tev_steps_monitor variable
};


#endif		// _Output

