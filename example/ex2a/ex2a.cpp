
/**
 * @file	ex2a.cpp
 *
 *
 *
 * @author	Michael Gegg
 *
 */

#include"ex2a.hpp"


/*
 * Setup for the Laser object. This function initializes the density matrix vector and the Liouvillian matrix that describes the master equation.
 */

#undef __FUNCT__					//the macro __FUNCT__ is used by Petsc for internal error handling
#define __FUNCT__ "Setup"				//this macro should be set to the corresponding function name before each application function definition (not for prototypes), but petsc also complains if this is not the case

PetscErrorCode Laser::Setup(Vec* dm, Mat* AA)
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
    PetscReal		pump = 0.01;


    //set simulation parameter to command line options
    ierr = PetscOptionsGetInt(NULL,NULL,"-ntls",&ntls,NULL);CHKERRQ(ierr);			//petsc relies heavily on command line options, which allows the user to try out different parameter sets, options, solvers etc.
    ierr = PetscOptionsGetInt(NULL,NULL,"-dx",&dx,NULL);CHKERRQ(ierr);				//whithout having to recompile the code, makes testing and exploring different solution methods much faster and more convenient.
    ierr = PetscOptionsGetInt(NULL,NULL,"-m0",&m0,NULL);CHKERRQ(ierr);				//therefore it is good practice to also use these features whenever possible
    ierr = PetscOptionsGetInt(NULL,NULL,"-dm0",&dm0,NULL);CHKERRQ(ierr);			//here we can reset the default values of the system size via command line

    ierr = PetscOptionsGetReal(NULL,NULL,"-domega_tls",&domega_tls,NULL);CHKERRQ(ierr);		//and here we can reset the simulation parameters
    ierr = PetscOptionsGetReal(NULL,NULL,"-gcouple",&gcouple,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-gamma",&gamma,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-pump",&pump,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-kappa",&kappa,NULL);CHKERRQ(ierr);

    ierr = AddParam("modeenergy",modeenergy); CHKERRQ(ierr);					//internal storage for parameters
    ierr = AddParam("domega_tls",domega_tls); CHKERRQ(ierr);
    ierr = AddParam("gcouple",gcouple); CHKERRQ(ierr);
    ierr = AddParam("gamma",gamma); CHKERRQ(ierr);
    ierr = AddParam("pump",pump); CHKERRQ(ierr);
    ierr = AddParam("kappa",kappa); CHKERRQ(ierr);


    //add the degrees of freedom
    ierr = TLSAdd(ntls,dx,dx,tlsenergy);CHKERRQ(ierr);					//set the internal structure to ntls two-level systems, no offdiagonal trunctation so far
    ierr = ModeAdd(m0+1,dm0,modeenergy);CHKERRQ(ierr);					//add a single bosonic mode, the modes always have to be set after all mls degrees of freedom have been specified.


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
    ierr = AddTLSIncoherentPump(*AA,d_nnz,o_nnz,0,1.0);CHKERRQ(ierr);				//incoherent pump
    ierr = AddLindbladMode(*AA,d_nnz,o_nnz,0,0,1.0);CHKERRQ(ierr);				//finite mode lifetime

    ierr = MatMPIAIJSetPreallocation(*AA,0,d_nnz,0,o_nnz); CHKERRQ(ierr);			//parallel
    ierr = MatSeqAIJSetPreallocation(*AA,0,d_nnz); CHKERRQ(ierr);				//and serial, the parallel (serial) function does nothing if single (multi) processor is used


    //specify the Liouvillian
    ierr = AddTLSH0(*AA,NULL,NULL,1,domega_tls*PETSC_i);CHKERRQ(ierr);					//possible detuning between mode and TLS
    ierr = AddTavisCummingsHamiltonianRWA(*AA,NULL,NULL,1,0,gcouple*PETSC_i);CHKERRQ(ierr);	//Tavis-Cummings Hamiltonian with RWA
    ierr = AddTLSSpontaneousEmission(*AA,NULL,NULL,1,gamma/2.0);CHKERRQ(ierr);			//individual spontaneous emission of the TLS
    ierr = AddTLSIncoherentPump(*AA,d_nnz,o_nnz,1,pump/2.0);CHKERRQ(ierr);			//incoherent pump
    ierr = AddLindbladMode(*AA,NULL,NULL,1,0,kappa/2.0);CHKERRQ(ierr);				//finite mode lifetime

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

PetscErrorCode MyOut::SetupMyOut(Laser * system)
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

    ierr = ddistfile->SetupDickeDistFile(system,"dicke.dat"); CHKERRQ(ierr);
    ierr = AddOFile(ddistfile); CHKERRQ(ierr);


    PetscFunctionReturn(0);
}


/*
 * Setup for the Observables file. Initializes the Observables and adds them to the list. There is no limit for the number of Observables.
 */

#undef __FUNCT__
#define __FUNCT__ "SetupMyObsFile"

PetscErrorCode ObservablesFile::SetupMyObsFile(Laser * system, std::string name)
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
    J10b	*j10b		= new J10b();
    J11bdb	*j11bdb		= new J11bdb();

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

    ierr = j10b->Setup(system);CHKERRQ(ierr);					//computes <J_{10} b>
    ierr = AddElem(j10b,"Re<J10b>\t\tIm<J10b>");CHKERRQ(ierr);

    ierr = j11bdb->Setup(system);CHKERRQ(ierr);					//computes <J_{11} b^\dagger b>
    ierr = AddElem(j11bdb,"<J10b>\t\t");CHKERRQ(ierr);

    ierr = MakeHeaderTEV();CHKERRQ(ierr);

    PetscFunctionReturn(0);
}


/*
 * Setup for the Gnfcts file.
 *
 */

#undef __FUNCT__
#define __FUNCT__ "SetupMyGnFile"

PetscErrorCode CorrelationsFile::SetupMyGnFile(Laser * sys, std::string name)
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
 * Setup for the <J_{10} b> observable.
 * the idea behind this is as follows:
 * the density matrix is a vector, lets write it like |rho>, then we can also express the trace as a vector |tr> and define the trace operation as a scalar product <tr|rho> = 1
 * but its also possible to express all conceivable observables using this: let O be the operator then <tr|O|rho> does the trick
 * the PModular class provides the function GenerateLeft(System*sys,Mat O) which essentially computes |left> = O^\dagger |tr>, then the observable is just <left|rho> which is computed by the Compute() function (called be the GenMonitor() function)
 * luckily there is no phase factor arising from the rotating frame for this observable
 */
#undef __FUNCT__
#define __FUNCT__ "Setup"

PetscErrorCode J10b::Setup(Laser* sys)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;

    Mat		J10,b,mult;

    ierr = sys->MatTLSJ10Left(&J10,1.0); CHKERRQ(ierr);		//corresponds to J_{10} \rho
    ierr = sys->MatModeLeftB(&b,0,1.0); CHKERRQ(ierr);		//corresponds to b \rho

    ierr = MatMatMult(J10,b,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&mult); CHKERRQ(ierr);	//this results in J_{10} b \rho

    ierr = GenerateLeft(sys,mult);CHKERRQ(ierr);		//this creates a vector corresponding to mult^\dagger |tr>

    isherm = 0;							//flag that tells psiquasp that the matrix is not hermitian, i.e. the observable not necessarily real valued and that we want real and imaginary value plotted into the ouput file

    ierr = MatDestroy(&J10);CHKERRQ(ierr);			//these are not needed anymore
    ierr = MatDestroy(&b);CHKERRQ(ierr);
    ierr = MatDestroy(&mult);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}


/*
 * Setup for the <J_{11} b^\dagger b> observable. Also no phase factor for this observable.
 */

#undef __FUNCT__
#define __FUNCT__ "Setup"

PetscErrorCode J11bdb::Setup(Laser* sys)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;

    Mat		J11,bdb,mult;

    ierr = sys->MatTLSJ11Left(&J11,1.0); CHKERRQ(ierr);		//corresponds to J_{11} \rho
    ierr = sys->MatModeLeftBdB(&bdb,0,1.0); CHKERRQ(ierr);	//corresponds to bdb \rho

    ierr = MatMatMult(J11,bdb,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&mult); CHKERRQ(ierr);	//this results in J_{11} bdb \rho

    ierr = GenerateLeft(sys,mult);CHKERRQ(ierr);		//this creates a vector corresponding to mult^\dagger |tr>

    isherm = 1;							//flag that tells psiquasp that the matrix is hermitian, i.e. the observable should be real valued and that we want only the real value plotted and the magnitude of the imaginary value checked

    ierr = MatDestroy(&J11);CHKERRQ(ierr);			//these are not needed anymore
    ierr = MatDestroy(&bdb);CHKERRQ(ierr);
    ierr = MatDestroy(&mult);CHKERRQ(ierr);

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
    Laser	twolevel;							//system specification
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
