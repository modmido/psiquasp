

/**
 * @file	system.hpp
 *
 * 		Definition of the System class
 * 		and some inline functions.
 *
 * @author	Michael Gegg
 *
 */

/**
 * TEST:
 *   - make sure that the old way of coding is preserved, what happens without mls, what happens with one type of mls
 *
 *
 */

/**
 * @mainpage	PsiQuaSP -- Permutation symmetry for identical Quantum Systems Package
 *
 * @section	one General information:
 *
 * The PsiQuaSP library provides all functionality that is needed to quickly set up a simulation for the permutation symmetric many multi-level system (MLS) method into code. The method is introduced in the three publications<br>
 * [1] [M. Richter, M. Gegg, TS. Theuerholz and A. Knorr, Phys. Rev. B 91, 035306 (2015)] (https://journals.aps.org/prb/abstract/10.1103/PhysRevB.91.035306)<br>
 * [2] [M. Gegg, M. Richter: New J. Phys. 18, 043037 (2016)] (http://iopscience.iop.org/article/10.1088/1367-2630/18/4/043037)<br>
 * [3] [M. Gegg, M. Richter: arXiv 1707.01079 (2017)] (https://arxiv.org/abs/1707.01079) <br>
 * The last publication serves as a short manual for this library. All occuring equations are solved using the PETSc and SLEPc packages, https://www.mcs.anl.gov/petsc/ and http://slepc.upv.es/ <br>
 * PsiQuaSP solely provides the functionality for setting up the master equation and the program output. Some understanding of PETSc and the permutationally symmetric methodology is required, but one can start working on the example 
 * codes and learn these things along the way. <br>
 * The core of the program is the System class: it provides the functionality to define all conceivable MLS setups interacting with an arbitrary number of modes. It is intended that the user does not directly use the System class, 
 * but rather defines a child class called e.g. MySystem which just provides a setup function. <br>
 * PsiQuaSP uses a vectorized master equation, thus the density matrix and related objects are represented by vectors and the Liouville superoperators are represented by a matrices, which allows to compute the spectrum of the Liouvillian using standard 
 * linear algebra methods.<br>
 * PsiQuaSP is designed around the sketches introduced in Refs. [2,3]: Setting master equations usually involves drawing a sketch and then setting the bubbles and arrows of these sketches by single function calls. For two-level systems the functionality is
 * furhter encapsulated and single contributions in the master equation, like a Hamiltonian or a dissipator, can be set with a single function call.
 * The numbers n_{xy} as appearing in the sketches in Ref. [2,3] as well as the ket and bra side of the mode denstiy matrix element are called degrees of freedom in this documentation, i.e. a collection of identical two-level systems is described by three 
 * degrees of freedom and a single mode is represented by two degrees of freedom.
 *
 *
 * \section	two Specifying system and output
 *
 * The general setup of every application code can be divided into four steps:
 *
 *  1) First the user needs to specify the (physical) system that should be investigated, i.e. whether two-, three- or multi-level systems are considered, how many of those MLS and how many bosonic modes. This essentially defines the Liouville space of 
 * the system. The MLS degrees of freedom are added by calling the functions System::MLSAddDens() and System::MLSAddPol() for each density and polarization dof appearing in the sketches introduced in Ref. [2,3]. Adding one density implies two-level 
 * systems, adding two densities implies three-level systems etc since PsiQuaSP follows the convention of Ref. [2,3] and always recovers the n00 degree of freedom. Modes are added by calling System::ModeAdd(). <br>
 * Please note that all MLS dofs need to be specified before the mode dofs are specified. <br>
 * 
 * 2) The input about the Liouville space is then used to create the System::index object by calling System::PQSPSetup(). This object handles the indexing of the vectors and matrices, i.e. it associates quantum numbers to every element of a vector and also 
 * explicitly computes the size/dimensionality of the  Liouville space for the given input and thus how long the vectors have to be. All needed information about these dof has to be provided at function call. Internally each degree of freedom is 
 * is labeled with an index starting from 0 in the order of which the degree of freedom has been set. This is important e.g. when preparing the density matrix in an excited state, since then the user needs to provide an array containing the quantum numbers,
 * see e.g. the discussion in Ref. [3] and examples example/ex1a and example/ex1b.<br>
 * The System::PQSPSetup() function creates a vector and a user specified number of matrices at function call. These are intended to be used as density matrix and Liouvillian(s). More than one matrix for the Liouvillian can be used to treat explicit 
 * time-dependencies in the Liouvillian. The user may also later create more vectors and matrices (of the right sizes and parallel layouts) using the functions PQSPCreateMat() and PQSPCreateVec().
 *
 *  3) After Liouville space is determined and the vector for the density matrix and matrices for the Liouvillian(s) have been set, the user needs to specify the Liouvillian(s) and maybe needs to set start values for the density matrix (only important for time integration).
 * The Liouvillians can be specified using the AddXXX() functions. The start values for the density matrix can be set using the DMXXX() functions.<br>
 *
 *  4) The last step is to specify the oputput of the program. Please refer to Ref. [3] and the example codes about the details.
 *
 *
 *
 * @section	three Solving the equations. Picking the libraries.
 *
 * The equations are solved relying entirely on PETSc and SLEPc functionalities. PETSc provides a large variety of time-integration algorithms. For direct steady state computations we recommend SLEPc.
 * For techniques like spectral transformation (e.g. shift and invert, see SLEPc documentation) the PETSc LU factorization implementation is used, which only supports serial (!) operation (for PETSc v3.7.6)
 * Treating larger matrices like that is inefficient, for such purposes it is recommended to use SuperLU or MUMPS (see PETSc documentation). There are a variety of graph partitioning libraries available through
 * PETSc. These are intended to improve parallel performance/layout, but can also be used to detect additional symmetries in some master equations. See e.g. example/ex5.
 *
 *
 *
 * @section	four Parallel operation:
 *
 * PsiQuaSP supports serical as well as parallel operation. The support for parallel operation relies fully on the PETSc functionality. PETSc uses MPI and a distributed memory approach. <br>
 * For using multiple processors the PETSc MPI script must be used. Without using this script the program will operate in uniprocessor mode. <br>
 * PETSc provides interfaces to different partitioning packages such as METIS/PARMETIS or SCOTCH, which can be used to increase parallel performance.
 *
 *
 * @section	six Example codes
 *
 * Overview over the example codes:
 *
 *  + 1a) Damped, open Tavis-Cummings model with simple observables
 *        this example explains the overall working principle of PsiQuaSP and simple PETSc time integration
 *  + 1b) The same master equation but introducing more advanced PETSc related concepts that are important to
 *        understand PsiQuaSP
 *
 *  + 2a) two-level laser master equation, same as in 1a/b but introducing an incoherent pump
 *        this example explains how to define custom observables
 *  + 2b) two-level laser, same as in 2a but using the SLEPc krylov-schur algorithm for direct steady state computation
 *  + 2c) two-level laser -- strong coupling using nonrwa
 *
 *  + 3a) Lambda system master equation introduced in Ref [3] -- general multi-level system usage
 *  + 3b) Three-level laser from Ref. [2] -- another multi-level system setup with reduced scaling
 *
 *  + 4a) Phononlaser/Supercooling master equation: two-level systems coupled to a phonon mode with external optical driving
 *        This example explains how to build arbitrary master equations using the elementary arrows
 *
 *  + 5) Same master equation as in 3a but using the graph partitioning package ParMetis to find further symmetries
 *       that lead to a further reduction in degrees of freedom, approximately from N^8 to N^7
 *
 */


#ifndef _System
#define _System

#include<cstdlib>
#include<sstream>
#include<iostream>
#include<cstring>
#include<list>

#include<petscvec.h>
#include<petscmat.h>
#include<petscts.h>


#define MIN(a,b)	(((a)<(b))?(a):(b))			//!< minimum macro
#define MAX(a,b)	(((a)>(b))?(a):(b))			//!< maximum macro
#define MAX_PARAM	50			        		//!< maximum number of parameters that can be stored internally
#define MAX_D_MLS   10                          //!< maximum number of different types of mls

class Index;
class Dim;
class MLSDim;
class ModeDim;

/**
 * @brief	Specifies the system under consideration<br>
 *
 * 		Enables the system specification: how many levels, how many individual systems, which mls offdiagonals, how many modes.
 * 		Also is responsible for setting up all processes, Hamiltonian and Lindblad dissipators <br>
 * 		This class is used to set up the whole master equation, which can then be solved or further processed by PETSc and SLEPc routines.
 * 		It provides functions/functionalities to set up
 * 			- add dimensions, which is equivalent to determining the size of the Liouville space
 * 			- all sorts of vectors, density matrices, but also trace operations, which can be represented by vectors too.
 * 			- matrix setup functions for many different Liouville operators
 * 			- some utility functions
 *
 * 		The design idea is that this class does not
 * 			+ ...contain the density matrix
 * 			+ ...contain the matrices for the Liouvillians
 * 			+ ...know whether steady state or time evolution is desired
 * 			+ ...does not know the time dependencies of the different parts in the master equation!
 *
 * 		Because
 * 			+ it is easier to interface to Petsc solution methods if the density matrix and the Liouvillian are not contained in System
 * 			+ PETSc/SLEPc do not need to know all the information in System to solve the equations
 * 			+ Not encapsulating the System information form the PETSc/SLEPc solution stage would only allow for intended usage, which would reduce the possibilities provided by PETSc/SLEPc
 *
 */

class System
{
  protected:
  //-----------------------------------------------------------------
  // Parameters for system specification:
  // mainly needed for the constructor call of Index *index.
  //-----------------------------------------------------------------
    PetscInt		N_MLS[MAX_D_MLS] = {};		        //!< total number of mls
    PetscInt        N_D_MLS;                            //!< total number of different mls
    PetscInt        multiMLS_start[MAX_D_MLS] = {};     //!< the index of the frist dimension of the mls kind in the indices array
    PetscInt		num_dims;			                //!< how many dimensions in total
    PetscInt		num_mlsdims;			            //!< how many mls dimensions
    PetscInt		num_mlsdens;			            //!< how many mls density degrees of freedom
    PetscInt		num_modes;			                //!< how many bosonic modes
    PetscInt		loc_size;			                //!< how many local dm entries?
    
    PetscInt        useMulti;                           //!< used for sanity check so that MLSAdd and MLSAddMulti are not mixed!
    PetscInt        useSingle;                          //!< used for sanity check so that MLSAdd and MLSAddMulti are not mixed!

    std::list<Dim*>	dimensions;			                //!< name of the dim


    //sanity check flags
    PetscInt		parallel_layout;		            //!< checks whether local distribution has already been set or not, is 0 if not, 1 if yes
    PetscInt		modesetup;			                //!< checks whether the user has already set a mode dimension, produces error messages if the MLS and mode dimension setup functions are mixed


  //-----------------------------------------------------------------
  //PARAMETERS: system dependent physical constants
  //-----------------------------------------------------------------

    //output
    PetscBool		longout;			//!< 0 for short stdout messages, 1 (or anything else) for long stdout output
    PetscBool		propout;			//!< 0 for all property/vector stdout messages, 1 (or anything else) for not
    PetscBool		liouout;			//!< 0 for all liouvillian/matrix stdout messages, 1 (or anything else) for not
    PetscInt		numparams;			//!< the number of parameters
    PetscReal		params[MAX_PARAM]	= {};	//!< the parameters
    std::string		paramname[MAX_PARAM]	= {};	//!< name of the simulation parameters


    //convergence
    PetscReal		real_value_tolerance;		//!< internal parameter for real value tolerance checks, can be set from command line with the -realvaluetol flag
    PetscReal		hermitian_tolerance;		//!< internal parameter for hermitianity checks, can be set from command line with the -hermitiantol flag


  //-----------------------------------------------------------------
  //METHODS: setup dimensions and local boundaries
  //-----------------------------------------------------------------

    //normal dimension setup
    PetscErrorCode  MLSAdd(PetscInt nmls);
    PetscErrorCode  MLSAddMulti(PetscInt nmls);
    PetscErrorCode	MLSAddDens(PetscInt n, PetscInt lenght,PetscReal energy);
    PetscErrorCode	MLSAddPol(PetscInt ket, PetscInt bra, PetscInt lenght);
    PetscErrorCode	MLSAddDens(MLSDim * dim, PetscInt lenght,PetscReal energy);
    PetscErrorCode	MLSAddPol(MLSDim * dim, PetscInt lenght);
    PetscErrorCode	ModeAdd(PetscInt length, PetscInt offdiag, PetscReal energy);

    //symmetry based, advanced dimension setup
//    PetscErrorCode    AddMLSPolFake(MLSDim name, PetscInt lenght,PetscReal dipole, PetscInt *rule);
//    PetscErrorCode    AddModeOneFake(PetscInt length, PetscInt offdiag, PetscReal energy, PetscInt *rule);
//    PetscErrorCode    AddNonInteractingMode(PetscReal energy);

    //normal setup for matrices and vectors
    PetscErrorCode	PQSPSetup(Vec * dm, PetscInt matrices, Mat * AAs);				//!< Setup of the density matrix and all needed Liouvillians for time evolution and eigenvalue problems. Creates the index.
    PetscErrorCode	PQSPSetupKSP(Vec * dm, Vec * b, Mat * AA);					//!< Setup of the density matrix, right hand side vector and the needed Liouvillian for steady state problems. Creates the index.
    PetscErrorCode	PQSPSetupIndex();								//!< Create the index, needs to be called after all dimensions have been Added

    //internal interfacing
    PetscErrorCode	ExtractMLSDimLengths(PetscInt **ret);
    PetscErrorCode	ExtractModeDimLengths(PetscInt **ret);
    PetscErrorCode	ExtractMLSDimPol(PetscInt **ret);


  //-----------------------------------------------------------------
  //METHODS: initialize denstiy matrix with various distributions
  //-----------------------------------------------------------------
    PetscErrorCode	DMWriteDiagPoissonianMls(Vec dm, PetscReal lambda);
    PetscErrorCode	DMWriteDiagPoissonianPh(Vec dm, PetscReal lambda);
    PetscErrorCode	DMWriteDiagPoissonianAll(Vec dm, PetscReal lambda);
    PetscErrorCode	DMWriteGroundState(Vec dm);
    PetscErrorCode	DMWritePureState(Vec dm, PetscInt *indices);
    PetscErrorCode	DMWriteUniformDistribution(Vec dm);

    PetscErrorCode	DMWriteDiagThermal(Vec dm, PetscReal beta);
    PetscErrorCode	MLSPartitionFunction(PetscReal beta,PetscReal *ret);						//the partition function of the set of mls



  //-----------------------------------------------------------------
  //METHODS: set Lindbladians
  //-----------------------------------------------------------------

    //beginner Lindbladian setup
    PetscErrorCode	AddDiagZeros(Mat AA, PetscInt * d_nnz, PetscInt choose);											//write zeros into the diagonal
    PetscErrorCode	AddDiagOne(Mat AA, PetscInt * d_nnz, PetscInt choose);												//write ones into the diagonal
    PetscErrorCode	AddLastRowTrace(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose);									//write the trace operation into the last row of the matrix

    PetscErrorCode	AddModeH0(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt modenumber, PetscScalar couplingconst);				//H0 contributions
    PetscErrorCode	AddMLSH0(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, MLSDim * pol, PetscScalar couplingconst);					//

    PetscErrorCode	AddMLSModeInt(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, MLSDim * mlsdown, MLSDim * mlsup, ModeDim photon, PetscScalar matrixelem);	//basic interaction Hamiltonians
    PetscErrorCode	AddMLSCohDrive(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, MLSDim * mlsdown, MLSDim * mlsup, PetscScalar matrixelem);		//
    PetscErrorCode	AddModeCohDrive(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt modenumber, PetscScalar couplingconst);			//

    PetscErrorCode	AddLindbladRelaxMLS(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, MLSDim * start, MLSDim * goal, PetscReal matrixelem);		//dissipator matrix elements
    PetscErrorCode	AddLindbladDephMLS(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, MLSDim * n,PetscReal matrixelem);					//cannot be complex thats why the matrix element is PetscReal
    PetscErrorCode	AddLindbladMode(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt modenumber, PetscReal matrixelem);			// --> generates compiler error if this is complex
    PetscErrorCode	AddLindbladModeThermal(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt modenumber, PetscReal matrixelem, PetscReal beta);	//


    //advanced, modular Liouville space operators setup

    //multi-level systems
    PetscErrorCode	AddMLSSingleArrowNonconnecting(Mat AA, PetscInt *d_nnz, PetscInt *o_nnz, PetscInt choose, MLSDim * elem, PetscScalar matrixelem);
    PetscErrorCode	AddMLSSingleArrowConnecting(Mat AA, PetscInt *d_nnz, PetscInt *o_nnz, PetscInt choose, MLSDim * elem1, MLSDim * elem2, PetscScalar matrixelem);


    //modes
    PetscErrorCode	AddModeLeftB(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt modenumber, PetscScalar matrixelem);
    PetscErrorCode	AddModeLeftBd(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt modenumber, PetscScalar matrixelem);
    PetscErrorCode	AddModeRightB(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt modenumber, PetscScalar matrixelem);
    PetscErrorCode	AddModeRightBd(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt modenumber, PetscScalar matrixelem);

    PetscErrorCode	AddModeLeftBdB(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt modenumber, PetscScalar matrixelem);
    PetscErrorCode	AddModeRightBdB(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt modenumber, PetscScalar matrixelem);
    PetscErrorCode	AddModeLeftBBd(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt modenumber, PetscScalar matrixelem);
    PetscErrorCode	AddModeRightBBd(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt modenumber, PetscScalar matrixelem);

    PetscErrorCode	AddModeLeftBRightBd(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt modenumber, PetscScalar matrixelem);
    PetscErrorCode	AddModeLeftBdRightB(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt modenumber, PetscScalar matrixelem);


  public:
    System ();											//constructor
    virtual ~System();

    Index		*index;									//!< the index for the whole program/setup stage

    //some return functions
    PetscInt		NumDims()		{ return num_dims;		    }		//!< Return the number of dimensions
    PetscInt		NMls()			{ return N_MLS[0];			}		//!< Return the number of mls TODO: maybe change this to a more general function
    PetscInt        NDMLS()         { return N_D_MLS;           }       //!< Return the number of different mls types
    PetscInt		NumMlsdims()	{ return num_mlsdims;		}		//!< Return the number of mls dof
    PetscInt		NLevels()		{ return num_mlsdens+1;		}		//!< Return the number of mls levels
    PetscInt		NumModes()		{ return num_modes;		}		//!< Return the number of modes

    PetscInt		LocSize()		{ return loc_size;		}		//!< Return the local size of the parallel snippet

    PetscInt		NumParams()		{ return numparams;		}		//!< Return the number of parameters
    PetscScalar		PValue(PetscInt n)	{ return params[n];		}		//!< Return the value of parameter n
    std::string		PName(PetscInt n)	{ return paramname[n];		}		//!< Return the name of parameter n

    PetscBool		LongOut()		{ return longout;		}		//!< Return the longout flag
    PetscBool		PropOut()		{ return propout;		}		//!< Return the propout flag

    PetscReal		RealValueTolerance()	{ return real_value_tolerance;	}		//!< Return the real value tolerance cutoff
    PetscReal		HermitianTolerance()	{ return hermitian_tolerance;	}		//!< Return the hermitian tolerance cutoff


  //-----------------------------------------------------------------
  //usefull functions
  //-----------------------------------------------------------------

    //factorials
    PetscReal		Factorial(PetscInt n);							//factorial function
    PetscReal		FactorialTrunc(PetscInt n, PetscInt m);					//truncated factorial, i.e. n!/(n-m)! = n*(n-1)...(n-m+1)

    //params
    PetscErrorCode	AddParam(std::string pname, PetscReal pvalue);				//Add a parameter to the list
    PetscErrorCode	GetParam(std::string pname, PetscReal * pvalue);			//retrieve the parameter
    PetscErrorCode	UpdateParam(std::string pname, PetscReal pvalue);			//update the parameter

    //convergence
    PetscErrorCode	SetRealValueTolerance(PetscReal value);
    PetscErrorCode	SetHermitianTolerance(PetscReal value);

    //dimensions
    PetscErrorCode	FindMatch(Dim * name, PetscInt * dim);					//returns the number of the dimension into the dim pointer
    std::string		DimName(PetscInt n);						        	//returns the name of the nth dimension
    PetscErrorCode	MLSDimMaxVal(PetscInt n,PetscInt *ret);					//return the maxmimum value of the mls dimension
    PetscErrorCode	ModeDimLen(PetscInt n,PetscInt *ret);					//return the value of modedimlenghts
    PetscErrorCode	IsMLSDimPol(PetscInt n,PetscInt *ret);					//return whether the dimension n is a mls polarization or not (i.e. 1 or 0)
    PetscErrorCode	Energies(PetscInt n,PetscReal *ret);					//return the energy of dimension n
    PetscErrorCode  SameType(MLSDim * ptr1, MLSDim * ptr2, PetscInt * type);//checks wether the two pointers refer to the same mls type and returns error messages if not

    PetscErrorCode	PrintEnergies();							//print all energies into stdout
    PetscErrorCode	PrintNames();								//print all name into stdout
    PetscErrorCode	PrintDimlengths();							//print all dimlengths into stdout


  //-----------------------------------------------------------------
  //METHODS: general matrix and vector routines
  //-----------------------------------------------------------------
    PetscErrorCode	PQSPCreateVec(Vec * dm, PetscInt *start, PetscInt *end);		//!< Create a vector, e.g. the density matrix. The first call to this funtion determines the parallel layout of the whole program
    PetscErrorCode	PQSPCreateMat(Mat * AA);						//!< Setup a single square TotalDOF()xTotalDOF() matrix for use in the program
    PetscErrorCode	PQSPCreateVecPlus1(Vec * dm, PetscInt *start, PetscInt *end);		//!< Setup the density matrix plus an extra entry at the bottom. This step determines the parallel_layout of the whole program!
    PetscErrorCode	PQSPSetupMatPlus1Row(Mat * AA);						//!< Setup a single (TotalDOF()+1)xTotalDOF() matrix for use in the program, used for steady state ksp

    PetscErrorCode	VecMLSGroundStateModeTraceout(Vec a);
    PetscErrorCode	VecTrace(Vec a);
    PetscErrorCode	VecIsHermitian(Vec dm, PetscInt *flg);					//check whether the vector/density matrix is hermitian
    PetscErrorCode	VecContractReal(Vec a, PetscInt *num, PetscInt ** indices, PetscReal ** factors);
    PetscErrorCode	VecContractScalar(Vec a, PetscInt *num, PetscInt ** indices, PetscScalar ** factors);


    //pmodular utilities with modes
    PetscErrorCode	MatModeLeftB(Mat *AA, PetscInt modenumber, PetscScalar matrixelem);
    PetscErrorCode	MatModeLeftBd(Mat *AA, PetscInt modenumber, PetscScalar matrixelem);
    PetscErrorCode	MatModeLeftBdB(Mat *AA, PetscInt modenumber, PetscScalar matrixelem);
    PetscErrorCode	MatModeLeftBBd(Mat *AA, PetscInt modenumber, PetscScalar matrixelem);
};


/**
 * @brief	Can represent an element of the density matrix, needed for the recursive correlation function algorithms of GnFct
 *
 */

class Elem
{
  public:
    PetscInt	*indices;			//the multiindex of the dm entry
    PetscInt	dmindex;			//the index as in the Index class
    PetscInt	length;				//length of the multiindex array, i.e. sys->num_dims
    PetscInt	mlslength;			//the number of mls dofs
    PetscInt	NMLS;				//the number of mls
    PetscReal	factor;				//the prefactor for the dm entry for computation
    PetscInt	opactions;			//the number of operators that have been applied to the entry, should be twice the order in the end g^(2) ~ <JdJdJJ> i.e. four operators
    PetscInt	order;				//the order of the normal ordered expectation value: 2 for <JdJdJJ>
    PetscInt    mlsTypeNumber;      //the type of mls
    
    inline	PetscInt	EqualIndices(const Elem& comp);		//are the all indices the same?
    inline	PetscInt	n00();					//compute the n00 value

    PetscErrorCode		PrintIndices();
    static PetscErrorCode	ComputeIndex(System * sys, std::list< Elem* >* clean);
    static PetscErrorCode	CombineListElems(std::list< Elem* >* clean, std::list< Elem* >* raw);
    static bool			ElemComp(const Elem* first, const Elem* second);		//static member functions do not know a thing about the current element, needed for sorting algorithms

    Elem(Index * index,PetscInt inorder, PetscInt type);				//normal constructor
    Elem(const Elem& source);						                    //copy constructor

    ~Elem();								                            //default destructor
};


/**
 * @brief	Checks whether the indices of two Elems are identical, returns 1 if yes and zero if not.
 *
 * @param	comp	the compare Elem
 *
 */

inline PetscInt Elem::EqualIndices(const Elem& comp)
{
    PetscInt	i,ret = 0;

    for(i=0; i < length; i++)	if(indices[i] == comp.indices[i]) ret++;

    ret /= length;

    return ret;
}


/**
 * @brief	Computes the value of the n00 dof for the Elem class.
 *
 */

inline PetscInt Elem::n00()
{
    PetscInt	i,ret	= NMLS;

    for(i=0; i < mlslength; i++)	ret -= indices[i];

    return	ret;
}


#endif		// _System
