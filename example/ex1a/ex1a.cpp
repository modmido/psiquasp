
/**
 * @file	ex1a.cpp
 * 
 * 		The system that is solved in this example is the open Tavis-Cummings model. There is a set of identical two-level systems coupled to a bosonic cavity mode with the usual dipole-dipole, rotating wave approximation interaction Hamiltonian. <br>
 * 		The two-level systems are subject to individual, spontaneous emission and the cavity photons also have a finite lifetime. Both these processes are described with a Lindbald dissipator. <br>
 * 		After the setup stage the code uses the PETSc time stepper functionalities to perform a direct integration of the differential equation using a Runge-Kutta algorithm. PETSc uses adaptive stepwidth Runge-Kutta. <br>
 * 		Some simple output file are generated, which monitor the solution.
 * 
 * @author	Michael Gegg
 * 
 */
 
#include"ex1a.hpp"


/*
 * Setup for the open Tavis-Cummings OTC object. This function initializes the density matrix vector and the Liouvillian matrix that describes the master equation. 
 */

void OTC::Setup(Vec* dm, Mat* AA)
{
    //parameters
    PetscInt		ntls = 2, m0=6, dm0=6;
    PetscReal		modeenergy = 2.0;
    PetscReal		domega_tls = 0.0;
    PetscReal		tlsenergy = modeenergy + domega_tls*hbar;	//rotating frame relation
    PetscReal		gcouple = 0.001;
    PetscReal		gamma = 0.00001;
    PetscReal		kappa = 0.001;
    
    
    //add the degrees of freedom
    TLSAdd(ntls,ntls,ntls,tlsenergy);					//set the internal structure to ntls two-level systems, no offdiagonal trunctation so far
    ModeAdd(m0+1,dm0,modeenergy);					//add a single bosonic mode, the modes always have to be set after all mls degrees of freedom have been specified. dm0 is the order of the offdiagonals that are included, more about this in ex1b
    
    //setup internal structure like the index and parallel layout and create vector for density matrix and matrix for Liouvillian
    PQSPSetup(dm,1,AA);
    
    index->PrintIndices();
    index->PrintElements();
    
    //write start values into the density matrix
    PetscInt	qnumbers [5] = {1,0,0,0,0};
    DMWritePureState(*dm,qnumbers);
    
    
    //specify the Liouvillian
    AddTLSH0(*AA,NULL,NULL,1,domega_tls*PETSC_i);			//possible detuning between mode and TLS
    AddTavisCummingsHamiltonianRWA(*AA,NULL,NULL,1,0,gcouple*PETSC_i);	//Tavis-Cummings Hamiltonian with RWA
    AddTLSSpontaneousEmission(*AA,NULL,NULL,1,gamma/2.0);		//individual spontaneous emission of the TLS
    AddLindbladMode(*AA,NULL,NULL,1,0,kappa/2.0);			//finite mode lifetime
    
    MatAssemblyBegin(*AA,MAT_FINAL_ASSEMBLY);				//assemble the matrix
    MatAssemblyEnd(*AA,MAT_FINAL_ASSEMBLY);				//PETSc requires this step, this is basically due to the sparse matrix format.
}


/*
 * Setup for MyOut object. Initializes the output files and adds them to the list.
 * There is no limit to the number of output files
 */

void MyOut::SetupMyOut(OTC * system)
{
    //allocate all files that should be in the output
    ObservablesFile	*obsfile	= new ObservablesFile;
    DistFile		*n11file	= new DistFile;
    DistFile		*m0file		= new DistFile;
    CorrelationsFile	*gnfile		= new CorrelationsFile;
    
    
    //dof identifiers
    MLSDim	n11 (1,1);
    
    
    //initialization
    obsfile->SetupMyObsFile(system,"observables.dat");			//user specified (see below)
    AddOFile(obsfile);
    
    n11file->SetupMLSDistFile(system,"n11.dat",n11);			//distribution in the n11 number states P[n11,0,0]
    AddOFile(n11file);
    
    m0file->SetupModeDistFile(system,"m0.dat",0);			//mode distribution, i.e. mode diagonal elements
    AddOFile(m0file);
    
    gnfile->SetupMyGnFile(system,"correlation.dat");			//user specified (see below)
    AddOFile(gnfile);
}


/*
 * Setup for the Observables file. Initializes the Observables and adds them to the list. There is no limit for the number of Observables.
 */

void ObservablesFile::SetupMyObsFile(OTC * system, std::string name)
{
    //open file, make gen. header
    SetOFile(system,name);		//Open File, set file name
    WriteSystemParameters(system);	//write system parameters into the output file
    
    
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
    
    
    //initialize them and add them to the list
    ptrace->SetupTrMinus1(system);				//this computes tr(rho)-1, which is a nice convergence check
    AddElem(ptrace,"tr[]-1\t");					//add it to the list and give it a name that is printed into the file header
    
    pdens11->SetupMlsOccupation(system,n11);			//computes the mean occupation in the upper TLS level, which is the expectation value of the n11 distribution, the identifier n11 allows to setup this observable for any level
    AddElem(pdens11,"<J_11>\t");
    
    pdens00->SetupMlsOccupation(system,n00);			//computes the mean occupation in the lower TLS level
    AddElem(pdens00,"<J_00>\t");
    
    ppol10->SetupMlsPolarization(system,n10,2.0/hbar);		//computes the <J_{10}> expectation value, this is not hermitian, so PsiQuaSP prints real and imaginary part by default 
    AddElem(ppol10,"Re<J_10>\t\tIm<J_10>");
    
    ppol01->SetupMlsPolarization(system,n01,-2.0/hbar);		//computes the <J_{01}> expectation value, 
    AddElem(ppol01,"Re<J_01>\t\tIm<J_01>");
    
    pmodeocc->SetupModeOccupation(system,0);			//computes the <b^\dagger b> mode occupation expectation value, there is an internal error check for the real valuedness of expectation values of hermitian operators
    AddElem(pmodeocc,"<bdb>\t");
    
    pmodepol->SetupModePolarization(system,0,-2.0/hbar);	//computes <b>
    AddElem(pmodepol,"Re<b>\t\tIm<b>");
    
    MakeHeaderTEV();
}


/*
 * Setup for the Gnfcts file.
 * 
 */

void CorrelationsFile::SetupMyGnFile(OTC * sys, std::string name)
{
    //open file, make gen. header
    SetOFile(sys,name); 		//Open File, set file name
    WriteSystemParameters(sys); 	//write system parameters into the output file
    
    
    //allocate new Observable objects
    Gnfct	*mode0secorder		= new Gnfct();
    Gnfct	*mode0thirdorder	= new Gnfct();
    Gnfct	*n11secorder		= new Gnfct();
    Gnfct	*n11thirdorder		= new Gnfct();
    
    
    //dof identifiers
    MLSDim	n01 (0,1);
    

    //initialize them and add them to the list
    mode0secorder->SetupModeGnfct(sys,0,2);			//computes g^(2) = <b^\dagger b^\dagger b b>/<b^\dagger b>^2 
    AddElem(mode0secorder,"g(2)(m0)"); 
   
    mode0thirdorder->SetupModeGnfct(sys,0,3);			//computes g^(2) = <b^\dagger b^\dagger b^\dagger b b b>/<b^\dagger b>^3 
    AddElem(mode0thirdorder,"g(3)(m0)"); 
    
    n11secorder->SetupMLSGnfct(sys,n01,2); 			//computes g^(2) = <J_{10} J_{10} J_{01} J_{01}>/<J_{10} J_{01}>^2 
    AddElem(n11secorder,"g(2)(n11)"); 
    
    n11thirdorder->SetupMLSGnfct(sys,n01,3); 			//computes g^(2) = <J_{10} J_{10} J_{10} J_{01} J_{01} J_{01}>/<J_{10} J_{01}>^3  
    AddElem(n11thirdorder,"g(3)(n11)"); 
    
    MakeHeaderTEV();
}


/*
 * User provided main function. Calls the setup routines, the computation stage then solely relies on Petsc.
 * 
 */

static char help[] = "Simple open Tavis-Cummings model example.\n\n";


int main(int argc, char **args)
{
    PetscInitialize(&argc,&args,(char*)0,help);
 
    Vec			dm;
    Mat			AA;
    

    
    
    //setup stage: using PsiQuaSP to initialize everything
    OTC	twolevel;							//system specification
    twolevel.Setup(&dm,&AA);

    MyOut	*out = new MyOut;					//output specification
    out->SetupMyOut(&twolevel);
    
    
    
    /*
    //computing stage: use the Petsc ODE utilities to solve the problem using explicit Runge-Kutta time integration
    TS		ts;							//time stepper
    TSAdapt	adapt;							//adaptive time step context
    
    //time step solver
    TSCreate(PETSC_COMM_WORLD,&ts);					//create time stepper context
    TSSetProblemType(ts,TS_LINEAR);					//tell petsc that we solve a linear diff. eq.
    TSSetType(ts,TSRK);							//set the time stepper to runge kutta
    TSRKSetType(ts,TSRK4);						//set it to 4 step runge kutta
    TSSetDuration(ts,100000,1.e+6);					//set the maximum integration cycles and time
    TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);			//what to do if the final time is not exactly reached with the time stepper, in this case nothing
    
    //adaptivity context for time stepper
    TSSetTolerances(ts,1.e-10,NULL,1.e-10,NULL);			//set the tolerances for adaptive time stepping
    TSGetAdapt(ts,&adapt);						//create the adaptive time stepping context
    TSAdaptSetType(adapt,TSADAPTNONE);					//set the type of adaptivity, none in this case

    //set the master equation
    TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,NULL); 		//use the time independent, matrix based
    TSSetRHSJacobian(ts,AA,AA,TSComputeRHSJacobianConstant,NULL); 	//jacobian method
    
    //set the output
    TSMonitorSet(ts,MyOut::TEVMonitor,out,NULL);
    
    //security measure
    TSSetTimeStep(ts,1); 						//for oscillatory stuff, otherwise it crashes, probably because initial time step too large
    
    //solve it and write into the output files at every 30th time step
    TSSolve(ts,dm);							//seems like initial conditions and solution/time steps get the same Vec, convenient...
*/

    
    //clean up stage: free everything that is not needed anymore
    MatDestroy(&AA); 
    VecDestroy(&dm); 
    //TSDestroy(&ts);
    delete out;
    
    PetscFinalize(); 
    return 0;
}

