
/**
 * @file	ex1b.cpp
 * 
 * 		The physics of this example is the same as in ex1a but here we will introduce 
 * 		 - a lot of Petsc specific tools that are indispensable when the code becomes more complex.
 * 		 - the special two-level system distribution "dicke distribution", which plots the dicke state distribution into an output file.
 * 		 - an adaptive time stepping algorithm which normally provides a significant speedup compared to a fixed time step method
 * 
 * @author	Michael Gegg
 * 
 */
 
#include"ex1b.hpp"


/*
 * Setup for the open Tavis-Cummings OTC object. This function initializes the density matrix vector and the Liouvillian matrix that describes the master equation. 
 */

#undef __FUNCT__					//the macro __FUNCT__ is used by Petsc for internal error handling
#define __FUNCT__ "Setup"				//this macro should be set to the corresponding function name before each application function definition (not for prototypes), but petsc also complains if this is not the case

PetscErrorCode OTC::Setup(Vec* dm, Mat* AA)
{
    PetscFunctionBeginUser;				//also needed for petsc error handling
    PetscErrorCode	ierr;				//the return type of (almost) all petsc/slepc and psiquasp functions
    
    //parameters
    PetscInt		ntls = 2, dx = 2, m0=6, dm0=6;
    PetscReal		modeenergy = 2.0;
    PetscReal		domega_tls = 0.0;
    PetscReal		tlsenergy = modeenergy + domega_tls*hbar;
    PetscReal		gcouple = 0.001;
    PetscReal		gamma = 0.000001;
    PetscReal		kappa = 0.001;
    PetscReal		beta = 100.;
    
    /*
     * CHKERRQ() is the error checking function. this should be called each time the PetscErrorCode ierr has been reset.
     * according to petsc developers the error checking does not affect the performance of the code
     * 
     */
    
    //set simulation parameter to command line options
    ierr = PetscOptionsGetInt(NULL,NULL,"-ntls",&ntls,NULL);CHKERRQ(ierr);			//petsc relies heavily on command line options, which allows the user to try out different parameter sets, options, solvers etc.
    ierr = PetscOptionsGetInt(NULL,NULL,"-dx",&dx,NULL);CHKERRQ(ierr);				//whithout having to recompile the code, makes testing and exploring different solution methods much faster and more convenient.
    ierr = PetscOptionsGetInt(NULL,NULL,"-m0",&m0,NULL);CHKERRQ(ierr);				//therefore it is good practice to also use these features whenever possible
    ierr = PetscOptionsGetInt(NULL,NULL,"-dm0",&dm0,NULL);CHKERRQ(ierr);			//here we can reset the default values of the system size via command line
    
    ierr = PetscOptionsGetReal(NULL,NULL,"-domega_tls",&domega_tls,NULL);CHKERRQ(ierr);		//and here we can reset the simulation parameters
    ierr = PetscOptionsGetReal(NULL,NULL,"-gcouple",&gcouple,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-gamma",&gamma,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-kappa",&kappa,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-beta",&beta,NULL);CHKERRQ(ierr);
    
    ierr = AddParam("modeenergy",modeenergy); CHKERRQ(ierr);					//this is the internal storage for parameters in psiquasp, the modeenergy determines the rotating frame, which is needed for the polarization observables (see below)
    ierr = AddParam("domega_tls",domega_tls); CHKERRQ(ierr);					//for the other parameters the only effect is that they are written into the header of the output files
    ierr = AddParam("gcouple",gcouple); CHKERRQ(ierr);						//one can also change/update parameters, e.g. when external control parameters are varied
    ierr = AddParam("gamma",gamma); CHKERRQ(ierr);
    ierr = AddParam("kappa",kappa); CHKERRQ(ierr);
    ierr = AddParam("beta",beta); CHKERRQ(ierr);
   
    /*
     * dx and dm0 are the degree of the offdiagonals that are included, you can play with this parameter and see how low it can be in order to get converged results.
     * generally the larger the dephasing the fewer the number of necessary offdiagonals
     */
    
    //add the degrees of freedom
    ierr = TLSAdd(ntls,dx,dx,tlsenergy);CHKERRQ(ierr);					//set the internal structure to ntls two-level systems, no offdiagonal trunctation so far
    ierr = ModeAdd(m0+1,dm0,modeenergy);CHKERRQ(ierr);					//add a single bosonic mode, the modes always have to be set after all mls degrees of freedom have been specified.
    ierr = PrintEnergies();CHKERRQ(ierr);						//control output to see whether everything works fine
    
    
    //setup internal structure like the index and parallel layout and create vector for density matrix and matrix for Liouvillian
    ierr = PQSPSetup(dm,1,AA);CHKERRQ(ierr);
    
    
    //write start values into the density matrix
    PetscInt	qnumbers [5] = {1,0,0,0,0};
    ierr = DMWritePureState(*dm,qnumbers);CHKERRQ(ierr);
    
    
    //preallocation stage, this is crucial when there are many different contributions in the master equation since otherwise the matrix setup becomes super slow
    PetscInt	*d_nnz	= new PetscInt [loc_size] ();						//estimate for the number of local elements per row, 
    PetscInt	*o_nnz	= new PetscInt [loc_size] ();						//estimate for the number of nonlocal elements per row, only exists if multiprocessor mode is used
    
    ierr = AddTLSH0(*AA,d_nnz,o_nnz,0,1.0);CHKERRQ(ierr);					//possible detuning between mode and TLS
    ierr = AddTavisCummingsHamiltonianRWA(*AA,d_nnz,o_nnz,0,0,1.0);CHKERRQ(ierr);		//Tavis-Cummings Hamiltonian with RWA
    ierr = AddTLSSpontaneousEmission(*AA,d_nnz,o_nnz,0,1.0);CHKERRQ(ierr);			//individual spontaneous emission of the TLS
    ierr = AddLindbladModeThermal(*AA,d_nnz,o_nnz,0,0,1.0,1.0);CHKERRQ(ierr);			//coupling the mode to a thermal bath
    
    
    /*
     * preallocation, based on d_nnz and o_nnz petsc preallocates storage for the matrix. the whole preallocation is only needed if the petsc default value of preallocated
     * storage per row is too small for the current application. if this happens then petsc reallocates the whole matrix for each element that is too much and then copies the entire matrix and repeats this
     * until all excess matrix elements have been set. this is incredibly inefficient. therefore the preallocation mode of the AddXXX() functions (activated with the 0 flag, fourth argument) gives a rough upper bound for the number
     * of elements per row. then the matrix can be filled with values and all preallocated slots that did not get filled with an entry are freed in the MAT_FINAL_ASSEMBLY step below, so that no storage overhead is possible
     * this is a necessity for large sparse matrices, but ultimately reduces storage requirements and computing time.
     */
    
    ierr = MatMPIAIJSetPreallocation(*AA,0,d_nnz,0,o_nnz); CHKERRQ(ierr);			//parallel
    ierr = MatSeqAIJSetPreallocation(*AA,0,d_nnz); CHKERRQ(ierr);				//and serial, the parallel (serial) function does nothing if single (multi) processor is used
    
    
    //specify the Liouvillian
    ierr = AddTLSH0(*AA,NULL,NULL,1,domega_tls*PETSC_i);CHKERRQ(ierr);				//possible detuning between mode and TLS
    ierr = AddTavisCummingsHamiltonianRWA(*AA,NULL,NULL,1,0,gcouple*PETSC_i);CHKERRQ(ierr);	//Tavis-Cummings Hamiltonian with RWA
    ierr = AddTLSSpontaneousEmission(*AA,NULL,NULL,1,gamma/2.0);CHKERRQ(ierr);			//individual spontaneous emission of the TLS
    ierr = AddLindbladModeThermal(*AA,NULL,NULL,1,0,kappa/2.0,beta);CHKERRQ(ierr);		//coupling the mode to a thermal bath
    
    ierr = MatAssemblyBegin(*AA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);				//assemble the matrix
    ierr = MatAssemblyEnd(*AA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);				//PETSc requires this step, this is basically due to the sparse matrix format.
    
    PetscFunctionReturn(0);
}


/*
 * Setup for MyOut object. Initializes the output files and adds them to the list.
 * There is no limit to the number of output files
 */

#undef __FUNCT__
#define __FUNCT__ "SetupMyOut"

PetscErrorCode MyOut::SetupMyOut(OTC * system)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
  
    //allocate all files that should be in the output
    ObservablesFile	*obsfile	= new ObservablesFile;
    DistFile		*n11file	= new DistFile;
    DistFile		*m0file		= new DistFile;
    CorrelationsFile	*gnfile		= new CorrelationsFile;
    DickeDistFile	*ddistfile	= new DickeDistFile;
    
    
    //dof identifiers
    MLSDim	n11 (1,1);
    
    
    //initialization
    ierr = obsfile->SetupMyObsFile(system,"observables.dat");CHKERRQ(ierr);			//user specified (see below)
    ierr = AddOFile(obsfile);CHKERRQ(ierr);
    
    ierr = n11file->SetupMLSDistFile(system,"n11.dat",n11);CHKERRQ(ierr);			//distribution in the n11 number states P[n11,0,0]
    ierr = AddOFile(n11file);CHKERRQ(ierr);
    
    ierr = m0file->SetupModeDistFile(system,"m0.dat",0);CHKERRQ(ierr);				//mode distribution, i.e. mode diagonal elements
    ierr = AddOFile(m0file);CHKERRQ(ierr);
    
    ierr = gnfile->SetupMyGnFile(system,"correlation.dat");CHKERRQ(ierr);			//user specified (see below)
    ierr = AddOFile(gnfile);CHKERRQ(ierr);

    /*
     * This is the dicke distribution file, which internally handles the DickeDistribution class. In this file the occpations in the Dicke states are plotted.
     * The first entry is the time, then the superradiant subspace is written with ascending excitation number starting with the ground state occupation (P[0,0,0])
     * after that the l_{max}-1 subspace is written with ascending excitation number, etc.
     * ground state here refers to the "normal", trivial ground state, i.e. no ultra strong coupling etc assumed, the distribution will still be valid in the ultra strong coupling regime, but the energy ordering will then likely be off
     */
    
    ierr = ddistfile->SetupDickeDistFile(system,"dicke.dat"); CHKERRQ(ierr);
    ierr = AddOFile(ddistfile); CHKERRQ(ierr);
    
    
    PetscFunctionReturn(0);
}


/*
 * Setup for the Observables file. Initializes the Observables and adds them to the list. There is no limit for the number of Observables.
 */

#undef __FUNCT__
#define __FUNCT__ "SetupMyObsFile"

PetscErrorCode ObservablesFile::SetupMyObsFile(OTC * system, std::string name)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    
    //open file, make gen. header
    ierr = SetOFile(system,name);CHKERRQ(ierr);		//Open File, set file name
    ierr = WriteSystemParameters(system);CHKERRQ(ierr);	//write system parameters into the output file
    
    
    //allocate all files that should be in the output
    Observable	*ptrace		= new Observable();
    Observable	*pdens11	= new Observable();
    Observable	*pdens00	= new Observable();
    Observable	*ppol10		= new Observable();
    Observable	*ppol01		= new Observable();
    Observable	*pmodeocc	= new Observable();
    Observable	*pmodepol	= new Observable();
    
    
    //dof identifiers
    MLSDim	n11 (1,1);					//these objects allow to identify different degrees of freedom 
    MLSDim	n00 (0,0);					//and thus facilitate the setup of observables, distributions but also more general Liouvillians
    MLSDim	n10 (1,0);
    MLSDim	n01 (0,1);
    
    
    //get the value of the prefactor in the rotating frame Hamiltonian, which is equal to modeenergy in this example
    PetscReal	rotenergy;
    ierr = system->GetParam("modeenergy",&rotenergy); CHKERRQ(ierr);
    
    
    //initialize them and add them to the list
    ierr = ptrace->SetupTrMinus1(system);CHKERRQ(ierr);				//this computes tr(rho)-1, which is a nice convergence check
    ierr = AddElem(ptrace,"tr[]-1\t");CHKERRQ(ierr);				//add it to the list and give it a name that is printed into the file header
    
    ierr = pdens11->SetupMlsOccupation(system,n11);CHKERRQ(ierr);		//computes the mean occupation in the upper TLS level, which is the expectation value of the n11 distribution, the identifier n11 allows to setup this observable for any level
    ierr = AddElem(pdens11,"<J_11>\t");CHKERRQ(ierr);
    
    ierr = pdens00->SetupMlsOccupation(system,n00);CHKERRQ(ierr);		//computes the mean occupation in the lower TLS level
    ierr = AddElem(pdens00,"<J_00>\t");CHKERRQ(ierr);
    
    ierr = ppol10->SetupMlsPolarization(system,n10,rotenergy/hbar);CHKERRQ(ierr);	//computes the <J_{10}> expectation value, this is not hermitian, so PsiQuaSP prints real and imaginary part by default 
    ierr = AddElem(ppol10,"Re<J_10>\t\tIm<J_10>");CHKERRQ(ierr);
    
    ierr = ppol01->SetupMlsPolarization(system,n01,-rotenergy/hbar);CHKERRQ(ierr);	//computes the <J_{01}> expectation value, 
    ierr = AddElem(ppol01,"Re<J_01>\t\tIm<J_01>");CHKERRQ(ierr);
    
    ierr = pmodeocc->SetupModeOccupation(system,0);CHKERRQ(ierr);		//computes the <b^\dagger b> mode occupation expectation value, there is an internal error check for the real valuedness of expectation values of hermitian operators
    ierr = AddElem(pmodeocc,"<bdb>\t");CHKERRQ(ierr);
    
    ierr = pmodepol->SetupModePolarization(system,0,-rotenergy/hbar);CHKERRQ(ierr);	//computes <b>
    ierr = AddElem(pmodepol,"Re<b>\t\tIm<b>");CHKERRQ(ierr);
    
    ierr = MakeHeaderTEV();CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


/*
 * Setup for the Gnfcts file.
 * 
 */

#undef __FUNCT__
#define __FUNCT__ "SetupMyGnFile"

PetscErrorCode CorrelationsFile::SetupMyGnFile(OTC * sys, std::string name)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    
    //open file, make gen. header
    ierr = SetOFile(sys,name);CHKERRQ(ierr);		//Open File, set file name
    ierr = WriteSystemParameters(sys);CHKERRQ(ierr);	//write system parameters into the output file
    
    
    //allocate new Observable objects
    Gnfct	*mode0secorder		= new Gnfct();
    Gnfct	*mode0thirdorder	= new Gnfct();
    Gnfct	*n11secorder		= new Gnfct();
    Gnfct	*n11thirdorder		= new Gnfct();
    
    
    //dof identifiers
    MLSDim	n01 (0,1);
    

    //initialize them and add them to the list
    ierr = mode0secorder->SetupModeGnfct(sys,0,2);CHKERRQ(ierr);			//computes g^(2) = <b^\dagger b^\dagger b b>/<b^\dagger b>^2 
    ierr = AddElem(mode0secorder,"g(2)(m0)");CHKERRQ(ierr); 
   
    ierr = mode0thirdorder->SetupModeGnfct(sys,0,3);CHKERRQ(ierr);			//computes g^(2) = <b^\dagger b^\dagger b^\dagger b b b>/<b^\dagger b>^3 
    ierr = AddElem(mode0thirdorder,"g(3)(m0)");CHKERRQ(ierr);
    
    ierr = n11secorder->SetupMLSGnfct(sys,n01,2);CHKERRQ(ierr);				//computes g^(2) = <J_{10} J_{10} J_{01} J_{01}>/<J_{10} J_{01}>^2 
    ierr = AddElem(n11secorder,"g(2)(n11)");CHKERRQ(ierr);
    
    ierr = n11thirdorder->SetupMLSGnfct(sys,n01,3);CHKERRQ(ierr);			//computes g^(2) = <J_{10} J_{10} J_{10} J_{01} J_{01} J_{01}>/<J_{10} J_{01}>^3  
    ierr = AddElem(n11thirdorder,"g(3)(n11)");CHKERRQ(ierr);
    
    ierr = MakeHeaderTEV();CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


/*
 * User provided main function. Calls the setup routines, the computation stage then solely relies on Petsc.
 * 
 */

static char help[] = "Simple open Tavis-Cummings model example.\n\n";


int main(int argc, char **args)
{
    PetscInitialize(&argc,&args,(char*)0,help);
 
    PetscErrorCode	ierr;
    Vec			dm;
    Mat			AA;
    

    
    
    //setup stage: using PsiQuaSP to initialize everything
    OTC	twolevel;							//system specification
    ierr = twolevel.Setup(&dm,&AA);CHKERRQ(ierr);

    MyOut	*out = new MyOut;					//output specification
    ierr = out->SetupMyOut(&twolevel);CHKERRQ(ierr);
    
    
    
    
    //computing stage: use Petsc to solve the problem using explicit Runge-Kutta time integration
    TS		ts;							//time stepper
    TSAdapt	adapt;							//adaptive time step context
    
    //time step solver
    ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);				//create time stepper context
    ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);				//tell petsc that we solve a linear diff. eq.
    ierr = TSSetType(ts,TSRK);CHKERRQ(ierr);						//set the time stepper to runge kutta
    ierr = TSRKSetType(ts,TSRK3BS);CHKERRQ(ierr);					//set it to 3rd order RK scheme of Bogacki-Shampine with 2nd order embedded method, this is an adaptive step width Runge-Kutta
    ierr = TSSetDuration(ts,100000,1.e+6);CHKERRQ(ierr);				//set the maximum integration cycles and time
    ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);		//what to do if the final time is not exactly reached with the time stepper, in this case nothing
    
    //adaptivity context for time stepper
    ierr = TSSetTolerances(ts,1.e-10,NULL,1.e-10,NULL);CHKERRQ(ierr);			//set the tolerances for adaptive time stepping
    ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);						//create the adaptive time stepping context
    ierr = TSAdaptSetType(adapt,TSADAPTBASIC);CHKERRQ(ierr);				//set the type of adaptivity

    /*
     * with the TSSetFromOptions() command every specification of the time stepper integration algorithm can be set from command line, which is incredibly useful
     * there are a vast number of different settings and solvers that may speed up convergence significantly, see petsc documentation for the available options
     * for instance you can compare the runtime of an adaptive vs a nonadaptive time integration algorithm, you can try other methods such as pseudo time stepping, see the petsc documentation
     */
    
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);						//magic set everything from command line function
    
    
    //set the master equation
    ierr = TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,NULL);CHKERRQ(ierr);	//use the time independent, matrix based
    ierr = TSSetRHSJacobian(ts,AA,AA,TSComputeRHSJacobianConstant,NULL);CHKERRQ(ierr);	//jacobian method
    
    //set the output
    ierr = TSMonitorSet(ts,MyOut::TEVMonitor,out,NULL);CHKERRQ(ierr);
    
    //security measure
    ierr = TSSetTimeStep(ts,0.01);CHKERRQ(ierr);					//for oscillatory stuff, otherwise it crashes, because initial time step may be too large
    
    
    //solve it and write into the output at every step
    ierr = TSSolve(ts,dm);CHKERRQ(ierr);						//actual solution


    
    //clean up stage: free everything that is not needed anymore
    MatDestroy(&AA); 
    VecDestroy(&dm); 
    TSDestroy(&ts); 
    delete out;
    
    PetscFinalize(); 
    return 0;
}

