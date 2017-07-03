
/**
 * @file	ex3a.cpp
 * 
 * 		
 * 
 * @author	Michael Gegg
 * 
 */
 
#include"ex3a.hpp"

#undef __FUNCT__
#define __FUNCT__ "Setup"

PetscErrorCode Lambda::Setup(Vec* dm, Mat* AA)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    
    //parameters
    PetscInt		nmls = 2, dx21 = 2, dx20 = 2, dx10 = 2, m0=4, dm0=4;
    PetscReal		energy1 = 2.0;
    PetscReal		laserfreq = 1.9/hbar;
    PetscReal		delta0 = 0.0;
    PetscReal		delta1 = 0.0;
    PetscReal		energy2 = delta1 + energy1 - hbar*laserfreq;		//rotating frame energy and detuning relations
    PetscReal		modeenergy = delta0 + energy1;				//...
    PetscReal		gcouple = 0.001;
    PetscReal		edrive = 0.001;
    PetscReal		gamma = 0.000001;
    PetscReal		gammap = 0.000001;
    PetscReal		kappa = 0.01;

    
    //set simulation parameter to command line options
    ierr = PetscOptionsGetInt(NULL,NULL,"-nmls",&nmls,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-dx21",&dx21,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-dx20",&dx20,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-dx10",&dx10,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-m0",&m0,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-dm0",&dm0,NULL);CHKERRQ(ierr);
    
    ierr = PetscOptionsGetReal(NULL,NULL,"-delta0",&delta0,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-delta1",&delta1,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-gcouple",&gcouple,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-edrive",&edrive,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-gamma",&gamma,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-gammap",&gammap,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-kappa",&kappa,NULL);CHKERRQ(ierr);
    
    ierr = AddParam("energy1",energy1); CHKERRQ(ierr);
    ierr = AddParam("laserfreq",laserfreq); CHKERRQ(ierr);
    ierr = AddParam("delta0",delta0); CHKERRQ(ierr);
    ierr = AddParam("delta1",delta1); CHKERRQ(ierr);
    ierr = AddParam("gcouple",gcouple); CHKERRQ(ierr);
    ierr = AddParam("edrive",edrive); CHKERRQ(ierr);
    ierr = AddParam("gamma",gamma); CHKERRQ(ierr);
    ierr = AddParam("gammap",gammap); CHKERRQ(ierr);
    ierr = AddParam("kappa",kappa); CHKERRQ(ierr);
    
    
    //add the degrees of freedom
    MLSDim	n22 (2,2), n21 (2,1), n20 (2,0), n12 (1,2), n11 (1,1), n10 (1,0), n02 (0,2), n01 (0,1), n00 (0,0);		//first we create all MLS and Mode dimension identifiers
    ModeDim	mket (0,0), mbra (1,0);
    
    N_MLS = nmls;							//N_MLS is the internal parameter that defines the number of multi-level systems
        
    ierr = MLSAddDens(n22,nmls+1,energy2); CHKERRQ(ierr);		//adding the mls degrees of freedom
    ierr = MLSAddPol(n21,dx21+1); CHKERRQ(ierr);			//here we use offdiagonal truncation if dx21 < N_MLS
    ierr = MLSAddPol(n12,dx21+1); CHKERRQ(ierr);			//
    ierr = MLSAddPol(n20,dx20+1); CHKERRQ(ierr);
    ierr = MLSAddPol(n02,dx20+1); CHKERRQ(ierr);
    ierr = MLSAddDens(n11,nmls+1,energy1); CHKERRQ(ierr);
    ierr = MLSAddPol(n10,dx10+1); CHKERRQ(ierr);
    ierr = MLSAddPol(n01,dx10+1); CHKERRQ(ierr);
    ierr = ModeAdd(m0+1,dm0,modeenergy);CHKERRQ(ierr);			//add a single bosonic mode, the modes always have to be set after all mls degrees of freedom have been specified.
    
    
    //setup internal structure like the index and parallel layout and create vector for density matrix and matrix for Liouvillian
    ierr = PQSPSetup(dm,1,AA);CHKERRQ(ierr);
    
    
    //write start values into the density matrix
    PetscInt	qnumbers[10] = {};
    qnumbers[5] = 1;
    ierr = DMWritePureState(*dm,qnumbers);CHKERRQ(ierr);
    
    
    //preallocation stage, this is crucial when there are many different contributions in the master equation since otherwise the matrix setup becomes super slow
    PetscInt	*d_nnz	= new PetscInt [loc_size] ();						//estimate for the number of local elements per row, 
    PetscInt	*o_nnz	= new PetscInt [loc_size] ();						//estimate for the number of nonlocal elements per row, only exists if multiprocessor mode is used
    
    ierr = AddMLSH0(*AA,d_nnz,o_nnz,0,n21,1.0);CHKERRQ(ierr);					//detuning contribution for possible detuning between laser frequency and 1 <-> 2 transition frequency
    ierr = AddMLSH0(*AA,d_nnz,o_nnz,0,n20,1.0);CHKERRQ(ierr);					//....
    ierr = AddModeH0(*AA,d_nnz,o_nnz,0,0,1.0);CHKERRQ(ierr);					//detuning between mode and 1 <-> 0 transition
    
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,0,n02,n12,mket,1.0);CHKERRQ(ierr);			//mode - mls interaction, mode ket and left index change
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,0,n01,n11,mket,1.0);CHKERRQ(ierr);			//each function call represents one twoheaded arrow in the sketches, connecting two bubbles
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,0,n00,n10,mket,1.0);CHKERRQ(ierr);
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,0,n20,n21,mbra,1.0);CHKERRQ(ierr);			//mode bra and right index change
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,0,n10,n11,mbra,1.0);CHKERRQ(ierr);
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,0,n00,n01,mbra,1.0);CHKERRQ(ierr);
    
    ierr = AddMLSCohDrive(*AA,d_nnz,o_nnz,0,n22,n12,1.0);CHKERRQ(ierr);				//coherent drive of the 1 <-> transition
    ierr = AddMLSCohDrive(*AA,d_nnz,o_nnz,0,n21,n11,1.0);CHKERRQ(ierr);
    ierr = AddMLSCohDrive(*AA,d_nnz,o_nnz,0,n20,n10,1.0);CHKERRQ(ierr);
    ierr = AddMLSCohDrive(*AA,d_nnz,o_nnz,0,n22,n21,1.0);CHKERRQ(ierr);
    ierr = AddMLSCohDrive(*AA,d_nnz,o_nnz,0,n12,n11,1.0);CHKERRQ(ierr);
    ierr = AddMLSCohDrive(*AA,d_nnz,o_nnz,0,n02,n01,1.0);CHKERRQ(ierr);
    
    ierr = AddLindbladRelaxMLS(*AA,d_nnz,o_nnz,0,n11,n00,1.0);CHKERRQ(ierr);			//two spontaneous emission contributions
    ierr = AddLindbladRelaxMLS(*AA,d_nnz,o_nnz,0,n11,n22,1.0);CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(*AA,d_nnz,o_nnz,0,n10,1.0);CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(*AA,d_nnz,o_nnz,0,n01,1.0);CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(*AA,d_nnz,o_nnz,0,n12,1.0);CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(*AA,d_nnz,o_nnz,0,n21,1.0);CHKERRQ(ierr);
    
    ierr = AddLindbladMode(*AA,d_nnz,o_nnz,0,0,1.0);CHKERRQ(ierr);				//finite mode lifetime
    
    ierr = MatMPIAIJSetPreallocation(*AA,0,d_nnz,0,o_nnz); CHKERRQ(ierr);			//parallel
    ierr = MatSeqAIJSetPreallocation(*AA,0,d_nnz); CHKERRQ(ierr);				//and serial, the parallel (serial) function does nothing if single (multi) processor is used
    
    
    //specify the Liouvillian
    ierr = AddMLSH0(*AA,d_nnz,o_nnz,1,n21,delta1*PETSC_i);CHKERRQ(ierr);				//detuning contribution for possible detuning between laser frequency and 1 <-> 2 transition frequency
    ierr = AddMLSH0(*AA,d_nnz,o_nnz,1,n20,delta1*PETSC_i);CHKERRQ(ierr);				//....
    ierr = AddModeH0(*AA,d_nnz,o_nnz,1,0,delta0*PETSC_i);CHKERRQ(ierr);					//detuning between mode and 1 <-> 0 transition
    
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,1,n02,n12,mket,gcouple*PETSC_i);CHKERRQ(ierr);			//mode - mls interaction, ket and left index change
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,1,n01,n11,mket,gcouple*PETSC_i);CHKERRQ(ierr);
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,1,n00,n10,mket,gcouple*PETSC_i);CHKERRQ(ierr);
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,1,n20,n21,mbra,-gcouple*PETSC_i);CHKERRQ(ierr);			//minus sign due to the commutator
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,1,n10,n11,mbra,-gcouple*PETSC_i);CHKERRQ(ierr);
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,1,n00,n01,mbra,-gcouple*PETSC_i);CHKERRQ(ierr);
    
    ierr = AddMLSCohDrive(*AA,d_nnz,o_nnz,1,n22,n12,edrive*PETSC_i);CHKERRQ(ierr);			//coherent drive of the 1 <-> transition
    ierr = AddMLSCohDrive(*AA,d_nnz,o_nnz,1,n21,n11,edrive*PETSC_i);CHKERRQ(ierr);
    ierr = AddMLSCohDrive(*AA,d_nnz,o_nnz,1,n20,n10,edrive*PETSC_i);CHKERRQ(ierr);
    ierr = AddMLSCohDrive(*AA,d_nnz,o_nnz,1,n22,n21,-edrive*PETSC_i);CHKERRQ(ierr);
    ierr = AddMLSCohDrive(*AA,d_nnz,o_nnz,1,n12,n11,-edrive*PETSC_i);CHKERRQ(ierr);
    ierr = AddMLSCohDrive(*AA,d_nnz,o_nnz,1,n02,n01,-edrive*PETSC_i);CHKERRQ(ierr);
    
    ierr = AddLindbladRelaxMLS(*AA,d_nnz,o_nnz,1,n11,n00,gamma/2.0);CHKERRQ(ierr);			//two spontaneous emission contributions
    ierr = AddLindbladRelaxMLS(*AA,d_nnz,o_nnz,1,n11,n22,gammap/2.0);CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(*AA,d_nnz,o_nnz,1,n10,(gamma+gammap)/2.0);CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(*AA,d_nnz,o_nnz,1,n01,(gamma+gammap)/2.0);CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(*AA,d_nnz,o_nnz,1,n12,(gamma+gammap)/2.0);CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(*AA,d_nnz,o_nnz,1,n21,(gamma+gammap)/2.0);CHKERRQ(ierr);
    
    ierr = AddLindbladMode(*AA,d_nnz,o_nnz,1,0,kappa/2.0);CHKERRQ(ierr);				//finite mode lifetime
    
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

PetscErrorCode MyOut::SetupMyOut(Lambda * system)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
  
    //allocate all files that should be in the output
    ObservablesFile	*obsfile	= new ObservablesFile;
    DistFile		*n11file	= new DistFile;
    DistFile		*n22file	= new DistFile;
    DistFile		*m0file		= new DistFile;
    CorrelationsFile	*gnfile		= new CorrelationsFile;
    
    
    //dof identifiers
    MLSDim	n11 (1,1);
    MLSDim	n22 (1,1);
    
    
    //initialization
    ierr = obsfile->SetupMyObsFile(system,"observables.dat");CHKERRQ(ierr);			//user specified (see below)
    ierr = AddOFile(obsfile);CHKERRQ(ierr);
    
    ierr = n11file->SetupMLSDistFile(system,"n11.dat",n11);CHKERRQ(ierr);			//distribution in the n11 number states P[n11,0,0]
    ierr = AddOFile(n11file);CHKERRQ(ierr);
    
    ierr = n22file->SetupMLSDistFile(system,"n22.dat",n22);CHKERRQ(ierr);			//distribution in the n11 number states P[n11,0,0]
    ierr = AddOFile(n11file);CHKERRQ(ierr);
    
    ierr = m0file->SetupModeDistFile(system,"m0.dat",0);CHKERRQ(ierr);				//mode distribution, i.e. mode diagonal elements
    ierr = AddOFile(m0file);CHKERRQ(ierr);
    
    ierr = gnfile->SetupMyGnFile(system,"correlation.dat");CHKERRQ(ierr);			//user specified (see below)
    ierr = AddOFile(gnfile);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


/*
 * Setup for the Observables file. Initializes the Observables and adds them to the list. There is no limit for the number of Observables.
 */

#undef __FUNCT__
#define __FUNCT__ "SetupMyObsFile"

PetscErrorCode ObservablesFile::SetupMyObsFile(Lambda * system, std::string name)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    
    //open file, make gen. header
    ierr = SetOFile(system,name);CHKERRQ(ierr);		//Open File, set file name
    ierr = WriteSystemParameters(system);CHKERRQ(ierr);	//write system parameters into the output file
    
    
    //allocate all files that should be in the output
    Observable	*ptrace		= new Observable();
    Observable	*pdens11	= new Observable();
    Observable	*pdens22	= new Observable();
    Observable	*ppol10		= new Observable();
    Observable	*ppol20		= new Observable();
    Observable	*ppol21		= new Observable();
    Observable	*pmodeocc	= new Observable();
    Observable	*pmodepol	= new Observable();
    
    //dof identifiers
    MLSDim	n11 (1,1), n22 (2,2), n10 (1,0), n20 (2,0), n21 (2,1);
    
    
    //get the values necessary for the rotating frame frequencies
    PetscReal	energy1, laserfreq;
    ierr = system->GetParam("energy1",&energy1); CHKERRQ(ierr);
    ierr = system->GetParam("laserfreq",&laserfreq); CHKERRQ(ierr);
    
    //initialize them and add them to the list
    ierr = ptrace->SetupTrMinus1(system);CHKERRQ(ierr);				//this computes tr(rho)-1, which is a nice convergence check
    ierr = AddElem(ptrace,"tr[]-1\t");CHKERRQ(ierr);				//add it to the list and give it a name that is printed into the file header
    
    ierr = pdens11->SetupMlsOccupation(system,n11);CHKERRQ(ierr);		//computes the mean occupation in the upper TLS level, which is the expectation value of the n11 distribution, the identifier n11 allows to setup this observable for any level
    ierr = AddElem(pdens11,"<J_11>\t");CHKERRQ(ierr);
    
    ierr = pdens22->SetupMlsOccupation(system,n22);CHKERRQ(ierr);		//computes the mean occupation in the lower TLS level
    ierr = AddElem(pdens22,"<J_22>\t");CHKERRQ(ierr);
    
    ierr = ppol10->SetupMlsPolarization(system,n10,energy1/hbar);CHKERRQ(ierr);	//computes the <J_{10}> expectation value, this is not hermitian, so PsiQuaSP prints real and imaginary part by default 
    ierr = AddElem(ppol10,"Re<J_10>\t\tIm<J_10>");CHKERRQ(ierr);
    
    ierr = ppol20->SetupMlsPolarization(system,n20,energy1/hbar-laserfreq);CHKERRQ(ierr);
    ierr = AddElem(ppol20,"Re<J_20>\t\tIm<J_20>");CHKERRQ(ierr);
    
    ierr = ppol21->SetupMlsPolarization(system,n21,-laserfreq);CHKERRQ(ierr);
    ierr = AddElem(ppol21,"Re<J_21>\t\tIm<J_21>");CHKERRQ(ierr);
    
    ierr = pmodeocc->SetupModeOccupation(system,0);CHKERRQ(ierr);		//computes the <b^\dagger b> mode occupation expectation value, there is an internal error check for the real valuedness of expectation values of hermitian operators
    ierr = AddElem(pmodeocc,"<bdb>\t");CHKERRQ(ierr);
    
    ierr = pmodepol->SetupModePolarization(system,0,-energy1/hbar);CHKERRQ(ierr);	//computes <b>
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

PetscErrorCode CorrelationsFile::SetupMyGnFile(Lambda * sys, std::string name)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    
    //open file, make gen. header
    ierr = SetOFile(sys,name);CHKERRQ(ierr);		//Open File, set file name
    ierr = WriteSystemParameters(sys);CHKERRQ(ierr);	//write system parameters into the output file
    
    
    //allocate new Observable objects
    Gnfct	*mode0secorder		= new Gnfct();
    Gnfct	*mode0thirdorder	= new Gnfct();
    Gnfct	*n10secorder		= new Gnfct();
    Gnfct	*n10thirdorder		= new Gnfct();
    Gnfct	*n12secorder		= new Gnfct();
    Gnfct	*n12thirdorder		= new Gnfct();
    
    
    //dof identifiers
    MLSDim	n01 (0,1), n21 (2,1);
    

    //initialize them and add them to the list
    ierr = mode0secorder->SetupModeGnfct(sys,0,2);CHKERRQ(ierr);			//computes g^(2) = <b^\dagger b^\dagger b b>/<b^\dagger b>^2 
    ierr = AddElem(mode0secorder,"g(2)(m0)");CHKERRQ(ierr); 
   
    ierr = mode0thirdorder->SetupModeGnfct(sys,0,3);CHKERRQ(ierr);			//computes g^(2) = <b^\dagger b^\dagger b^\dagger b b b>/<b^\dagger b>^3 
    ierr = AddElem(mode0thirdorder,"g(3)(m0)");CHKERRQ(ierr);
    
    ierr = n10secorder->SetupMLSGnfct(sys,n01,2);CHKERRQ(ierr);				//computes g^(2) = <J_{10} J_{10} J_{01} J_{01}>/<J_{10} J_{01}>^2 
    ierr = AddElem(n10secorder,"g(2)(n11)");CHKERRQ(ierr);
    
    ierr = n10thirdorder->SetupMLSGnfct(sys,n01,3);CHKERRQ(ierr);			//computes g^(2) = <J_{10} J_{10} J_{10} J_{01} J_{01} J_{01}>/<J_{10} J_{01}>^3  
    ierr = AddElem(n10thirdorder,"g(3)(n11)");CHKERRQ(ierr);
    
    ierr = n12secorder->SetupMLSGnfct(sys,n21,2);CHKERRQ(ierr);				//computes g^(2) = <J_{12} J_{12} J_{21} J_{21}>/<J_{12} J_{21}>^2 
    ierr = AddElem(n12secorder,"g(2)(n11)");CHKERRQ(ierr);
    
    ierr = n12thirdorder->SetupMLSGnfct(sys,n21,3);CHKERRQ(ierr);
    ierr = AddElem(n12thirdorder,"g(3)(n11)");CHKERRQ(ierr);
    
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
    Lambda	lambda;							//system specification
    ierr = lambda.Setup(&dm,&AA);CHKERRQ(ierr);

    MyOut	*out = new MyOut;					//output specification
    ierr = out->SetupMyOut(&lambda);CHKERRQ(ierr);
    
    
    
    
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

