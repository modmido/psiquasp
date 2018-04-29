
/**
 * @file	ex6a.cpp
 *
 * @author	Michael Gegg
 * 
 */
 
#include"ex6a.hpp"


/*
 * Setup for the open Tavis-Cummings TwoTLS object. This function initializes the density matrix vector and the Liouvillian matrix that describes the master equation. 
 */

#undef __FUNCT__
#define __FUNCT__ "Setup"

PetscErrorCode TwoTLS::Setup(Vec* dm, Mat* AA)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    
    //parameters
    PetscInt		ntls0 = 2, dx0 = 1, ntls1 = 3, dx1 = 2, m0=1, dm0=1;
    PetscReal		modeenergy = 2.0;
    PetscReal		domega_tls0 = 0.0;
    PetscReal       domega_tls1 = 0.0;
    PetscReal		energy0 = modeenergy + domega_tls0*hbar;
    PetscReal       energy1 = modeenergy + domega_tls1*hbar;
    PetscReal		gcouple0 = 0.001;
    PetscReal       gcouple1 = 0.001;
    PetscReal		gamma0 = 0.000001;
    PetscReal       gamma1 = 0.000001;
    PetscReal		kappa = 0.001;
    PetscReal		beta = 100.;
    
    //set simulation parameter from command line options
    ierr = PetscOptionsGetInt(NULL,NULL,"-ntls0",&ntls0,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-dx0",&dx0,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-ntls1",&ntls1,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-dx1",&dx1,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-m0",&m0,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-dm0",&dm0,NULL);CHKERRQ(ierr);
    
    ierr = PetscOptionsGetReal(NULL,NULL,"-domega_tls0",&domega_tls0,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-domega_tls1",&domega_tls1,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-gcouple0",&gcouple0,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-gcouple1",&gcouple1,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-gamma0",&gamma0,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-gamma1",&gamma1,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-kappa",&kappa,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-beta",&beta,NULL);CHKERRQ(ierr);
    
    //put important parameters into globally accessible storage
    ierr = AddParam("modeenergy",modeenergy); CHKERRQ(ierr);
    ierr = AddParam("domega_tls0",domega_tls0); CHKERRQ(ierr);
    ierr = AddParam("domega_tls1",domega_tls1); CHKERRQ(ierr);
    ierr = AddParam("gcouple0",gcouple0); CHKERRQ(ierr);
    ierr = AddParam("gcouple1",gcouple1); CHKERRQ(ierr);
    ierr = AddParam("gamma0",gamma0); CHKERRQ(ierr);
    ierr = AddParam("gamma1",gamma1); CHKERRQ(ierr);
    ierr = AddParam("kappa",kappa); CHKERRQ(ierr);
    ierr = AddParam("beta",beta); CHKERRQ(ierr);
   
    //specifying the dimension identifiers
    MultiMLSDim    n11_0 (1,1,0), n10_0 (1,0,0), n01_0 (0,1,0), n00_0 (0,0,0);        //create the four dimension identifiers for the first type of tls
    MultiMLSDim    n11_1 (1,1,1), n10_1 (1,0,1), n01_1 (0,1,1), n00_1 (0,0,1);        //create the four dimension identifiers for the second type of tls
    ModeDim        mket (0,0), mbra (1,0);
    
    //creating the neccessary dimensions
    ierr = MLSAddMulti(ntls0); CHKERRQ(ierr);                       //first type of tls
    ierr = MLSAddDens(n11_0,ntls0+1,energy0); CHKERRQ(ierr);       //
    ierr = MLSAddPol(n10_0,dx0+1); CHKERRQ(ierr);                  //
    ierr = MLSAddPol(n01_0,dx0+1); CHKERRQ(ierr);                  //
    
    ierr = MLSAddMulti(ntls1); CHKERRQ(ierr);                       //second type of tls
    ierr = MLSAddDens(n11_1,ntls1+1,energy1); CHKERRQ(ierr);       //
    ierr = MLSAddPol(n10_1,dx1+1); CHKERRQ(ierr);                  //
    ierr = MLSAddPol(n01_1,dx1+1); CHKERRQ(ierr);                  //
    
    ierr = ModeAdd(m0+1,dm0,modeenergy);CHKERRQ(ierr);              //single bosonic mode
    
    //setup internal structure like the index and parallel layout and create vector for density matrix and matrix for Liouvillian
    ierr = PQSPSetup(dm,1,AA);CHKERRQ(ierr);
    
    index->PrintBlockSizes();
    
    //write start values into the density matrix
    PetscInt	qnumbers [8] = {2,0,0,0,0,0,0,0};                  //first tls type has a single excitation
    ierr = DMWritePureState(*dm,qnumbers);CHKERRQ(ierr);
    
    
    //preallocation stage, this is crucial when there are many different contributions in the master equation since otherwise the matrix setup becomes super slow
    PetscInt    *d_nnz    = new PetscInt [loc_size] ();                        //estimate for the number of local elements per row,
    PetscInt    *o_nnz    = new PetscInt [loc_size] ();                        //estimate for the number of nonlocal elements per row, only exists if multiprocessor mode is used
    
    ierr = AddMLSH0(*AA,d_nnz,o_nnz,0,n10_0,1.0);CHKERRQ(ierr);
    ierr = AddMLSH0(*AA,d_nnz,o_nnz,0,n10_1,1.0);CHKERRQ(ierr);
    
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,0,n00_0,n10_0,mket,1.0);CHKERRQ(ierr);           //the interaction of the frist kind of tls with the radiation mode
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,0,n01_0,n11_0,mket,1.0);CHKERRQ(ierr);
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,0,n00_0,n01_0,mbra,1.0);CHKERRQ(ierr);
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,0,n10_0,n11_0,mbra,1.0);CHKERRQ(ierr);
    
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,0,n00_1,n10_1,mket,1.0);CHKERRQ(ierr);           //the interaction of the second kind of tls with the radiation mode
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,0,n01_1,n11_1,mket,1.0);CHKERRQ(ierr);
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,0,n00_1,n01_1,mbra,1.0);CHKERRQ(ierr);
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,0,n10_1,n11_1,mbra,1.0);CHKERRQ(ierr);
    
    ierr = AddLindbladRelaxMLS(*AA,d_nnz,o_nnz,0,n11_0,n00_0,1.0);CHKERRQ(ierr);          //spontaneous emission of the first type of tls
    ierr = AddLindbladDephMLS(*AA,d_nnz,o_nnz,0,n10_0,1.0);CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(*AA,d_nnz,o_nnz,0,n01_0,1.0);CHKERRQ(ierr);
    
    ierr = AddLindbladRelaxMLS(*AA,d_nnz,o_nnz,0,n11_1,n00_1,1.0);CHKERRQ(ierr);          //spontaneous emission of the second type of tls
    ierr = AddLindbladDephMLS(*AA,d_nnz,o_nnz,0,n10_1,1.0);CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(*AA,d_nnz,o_nnz,0,n01_1,1.0);CHKERRQ(ierr);
    
    ierr = AddLindbladMode(*AA,d_nnz,o_nnz,0,0,1.0);CHKERRQ(ierr);                          //finite mode lifetime
    
    ierr = MatMPIAIJSetPreallocation(*AA,0,d_nnz,0,o_nnz); CHKERRQ(ierr);                   //parallel
    ierr = MatSeqAIJSetPreallocation(*AA,0,d_nnz); CHKERRQ(ierr);                           //and serial, the parallel (serial) function does nothing if single (multi) processor is used

    
    //specify the Liouvillian
    ierr = AddMLSH0(*AA,d_nnz,o_nnz,1,n10_0,domega_tls0*PETSC_i);CHKERRQ(ierr);
    ierr = AddMLSH0(*AA,d_nnz,o_nnz,1,n10_1,domega_tls1*PETSC_i);CHKERRQ(ierr);
    
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,1,n00_0,n10_0,mket,gcouple0*PETSC_i);CHKERRQ(ierr);           //the interaction of the frist kind of tls with the radiation mode
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,1,n01_0,n11_0,mket,gcouple0*PETSC_i);CHKERRQ(ierr);
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,1,n00_0,n01_0,mbra,-gcouple0*PETSC_i);CHKERRQ(ierr);
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,1,n10_0,n11_0,mbra,-gcouple0*PETSC_i);CHKERRQ(ierr);
    
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,1,n00_1,n10_1,mket,gcouple1*PETSC_i);CHKERRQ(ierr);           //the interaction of the second kind of tls with the radiation mode
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,1,n01_1,n11_1,mket,gcouple1*PETSC_i);CHKERRQ(ierr);
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,1,n00_1,n01_1,mbra,-gcouple1*PETSC_i);CHKERRQ(ierr);
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,1,n10_1,n11_1,mbra,-gcouple1*PETSC_i);CHKERRQ(ierr);
    
    ierr = AddLindbladRelaxMLS(*AA,d_nnz,o_nnz,1,n11_0,n00_0,gamma0/2.0);CHKERRQ(ierr);                //spontaneous emission of the first type of tls
    ierr = AddLindbladDephMLS(*AA,d_nnz,o_nnz,1,n10_0,gamma0/2.0);CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(*AA,d_nnz,o_nnz,1,n01_0,gamma0/2.0);CHKERRQ(ierr);
    
    ierr = AddLindbladRelaxMLS(*AA,d_nnz,o_nnz,1,n11_1,n00_1,gamma1/2.0);CHKERRQ(ierr);                //spontaneous emission of the second type of tls
    ierr = AddLindbladDephMLS(*AA,d_nnz,o_nnz,1,n10_1,gamma1/2.0);CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(*AA,d_nnz,o_nnz,1,n01_1,gamma1/2.0);CHKERRQ(ierr);
    
    ierr = AddLindbladMode(*AA,d_nnz,o_nnz,1,0,kappa/2.0);CHKERRQ(ierr);                                //finite mode lifetime

    ierr = MatAssemblyBegin(*AA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);                  //assemble the matrix
    ierr = MatAssemblyEnd(*AA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);                    //PETSc requires this step, this is basically due to the sparse matrix format.
    
    PetscFunctionReturn(0);
}


/*
 * Setup for MyOut object. Initializes the output files and adds them to the list.
 * There is no limit to the number of output files
 */

#undef __FUNCT__
#define __FUNCT__ "SetupMyOut"

PetscErrorCode MyOut::SetupMyOut(TwoTLS * system)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
  
    //allocate all files that should be in the output
    ObservablesFile	    *obsfile	= new ObservablesFile;
    DistFile		    *n110file	= new DistFile;
    DistFile            *n111file   = new DistFile;
    DistFile		    *m0file		= new DistFile;
    CorrelationsFile	*gnfile		= new CorrelationsFile;
    
    
    //dof identifiers
    MultiMLSDim         n11_0 (1,1,0), n11_1 (1,1,1);;
    
    
    //initialization
    ierr = obsfile->SetupMyObsFile(system,"observables.dat");CHKERRQ(ierr);			//user specified (see below)
    ierr = AddOFile(obsfile);CHKERRQ(ierr);
    
    ierr = n110file->SetupMLSDistFile(system,"n11_0.dat",n11_0);CHKERRQ(ierr);		//distribution in the n11 number states P[n11,0,0]
    ierr = AddOFile(n110file);CHKERRQ(ierr);
    
    ierr = n111file->SetupMLSDistFile(system,"n11_1.dat",n11_1);CHKERRQ(ierr);     //distribution in the n11 number states P[n11,0,0]
    ierr = AddOFile(n111file);CHKERRQ(ierr);
    
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

PetscErrorCode ObservablesFile::SetupMyObsFile(TwoTLS * system, std::string name)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    
    //open file, make gen. header
    ierr = SetOFile(system,name);CHKERRQ(ierr);		//Open File, set file name
    ierr = WriteSystemParameters(system);CHKERRQ(ierr);	//write system parameters into the output file
    
    
    //allocate all files that should be in the output
    Observable	*ptrace		= new Observable();
    Observable	*pdens11_0	= new Observable();
    Observable	*pdens00_0	= new Observable();
    Observable	*ppol10_0	= new Observable();
    Observable  *pdens11_1  = new Observable();
    Observable  *pdens00_1  = new Observable();
    Observable  *ppol10_1   = new Observable();
    Observable	*pmodeocc	= new Observable();
    Observable	*pmodepol	= new Observable();
    
    
    //dof identifiers
    MultiMLSDim     n11_0 (1,1,0), n00_0 (0,0,0), n10_0 (1,0,0), n01_0 (0,1,0);
    MultiMLSDim     n11_1 (1,1,1), n00_1 (0,0,1), n10_1 (1,0,1), n01_1 (0,1,1);
    
    
    //get the value of the prefactor in the rotating frame Hamiltonian, which is equal to modeenergy in this example
    PetscReal	rotenergy;
    ierr = system->GetParam("modeenergy",&rotenergy); CHKERRQ(ierr);
    
    
    //initialize them and add them to the list
    ierr = ptrace->SetupTrMinus1(system);CHKERRQ(ierr);				//this computes tr(rho)-1, which is a nice convergence check
    ierr = AddElem(ptrace,"tr[]-1\t");CHKERRQ(ierr);				//add it to the list and give it a name that is printed into the file header
    
    ierr = pdens11_0->SetupMlsOccupation(system,n11_0);CHKERRQ(ierr);
    ierr = AddElem(pdens11_0,"<J_11_0>");CHKERRQ(ierr);
    
    ierr = pdens00_0->SetupMlsOccupation(system,n00_0);CHKERRQ(ierr);		//computes the mean occupation in the lower TLS level
    ierr = AddElem(pdens00_0,"<J_00_0>");CHKERRQ(ierr);
    
    ierr = ppol10_0->SetupMlsPolarization(system,n10_0,rotenergy/hbar);CHKERRQ(ierr);	//computes the <J_{10}> expectation value, this is not hermitian, so PsiQuaSP prints real and imaginary part by default
    ierr = AddElem(ppol10_0,"Re<J_10_0>\t\tIm<J_10_0>");CHKERRQ(ierr);
    
    ierr = pdens11_1->SetupMlsOccupation(system,n11_1);CHKERRQ(ierr);
    ierr = AddElem(pdens11_1,"<J_11_1>");CHKERRQ(ierr);
    
    ierr = pdens00_1->SetupMlsOccupation(system,n00_1);CHKERRQ(ierr);        //computes the mean occupation in the lower TLS level
    ierr = AddElem(pdens00_1,"<J_00_1>");CHKERRQ(ierr);
    
    ierr = ppol10_1->SetupMlsPolarization(system,n10_1,rotenergy/hbar);CHKERRQ(ierr);    //computes the <J_{10}> expectation value, this is not hermitian, so PsiQuaSP prints real and imaginary part by default
    ierr = AddElem(ppol10_1,"Re<J_10_1>\t\tIm<J_10_1>");CHKERRQ(ierr);
    
    ierr = pmodeocc->SetupModeOccupation(system,0);CHKERRQ(ierr);		//computes the <b^\dagger b> mode occupation expectation value, there is an internal error check for the real valuedness of expectation values of hermitian operators
    ierr = AddElem(pmodeocc,"<bdb>\t");CHKERRQ(ierr);
    
    ierr = pmodepol->SetupModePolarization(system,0,-rotenergy/hbar);CHKERRQ(ierr);	//computes <b>
    ierr = AddElem(pmodepol,"Re<b>\t\tIm<b>");CHKERRQ(ierr);
    
    ierr = MakeHeaderTEV();CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


/*
 * Setup for the Gnfcts file.
 * The Gnfct class does not work with the multimls setup, how the mls gnfcts are set up in this case is explained in ex6b.
 *
 */

#undef __FUNCT__
#define __FUNCT__ "SetupMyGnFile"

PetscErrorCode CorrelationsFile::SetupMyGnFile(TwoTLS * sys, std::string name)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    
    //open file, make gen. header
    ierr = SetOFile(sys,name);CHKERRQ(ierr);		//Open File, set file name
    ierr = WriteSystemParameters(sys);CHKERRQ(ierr);	//write system parameters into the output file
    
    
    //allocate new Observable objects
    Gnfct	        *mode0secorder		= new Gnfct();
    Gnfct	        *mode0thirdorder	= new Gnfct();
    
    
    //initialize them and add them to the list
    ierr = mode0secorder->SetupModeGnfct(sys,0,2);CHKERRQ(ierr);			//computes g^(2) = <b^\dagger b^\dagger b b>/<b^\dagger b>^2 
    ierr = AddElem(mode0secorder,"g(2)(m0)");CHKERRQ(ierr); 
   
    ierr = mode0thirdorder->SetupModeGnfct(sys,0,3);CHKERRQ(ierr);			//computes g^(2) = <b^\dagger b^\dagger b^\dagger b b b>/<b^\dagger b>^3 
    ierr = AddElem(mode0thirdorder,"g(3)(m0)");CHKERRQ(ierr);
    
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
    TwoTLS	twolevel;							//system specification
    ierr = twolevel.Setup(&dm,&AA);CHKERRQ(ierr);

    MyOut    *out = new MyOut;                    //output specification
    ierr = out->SetupMyOut(&twolevel);CHKERRQ(ierr);
    
    
    
    
    //computing stage: use Petsc to solve the problem using explicit Runge-Kutta time integration
    TS        ts;                            //time stepper
    TSAdapt    adapt;                            //adaptive time step context

    //time step solver
    ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);                            //create time stepper context
    ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);                            //tell petsc that we solve a linear diff. eq.
    ierr = TSSetType(ts,TSRK);CHKERRQ(ierr);                                        //set the time stepper to runge kutta
    ierr = TSRKSetType(ts,TSRK3BS);CHKERRQ(ierr);                                    //set it to 3rd order RK scheme of Bogacki-Shampine with 2nd order embedded method, this is an adaptive step width Runge-Kutta
    ierr = TSSetMaxTime(ts,1.e+6);CHKERRQ(ierr);                                     //set the maximum integration time
    ierr = TSSetMaxSteps(ts,100000);CHKERRQ(ierr);                                   //set the maximum integration steps
    ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);        //what to do if the final time is not exactly reached with the time stepper, in this case nothing

    //adaptivity context for time stepper
    ierr = TSSetTolerances(ts,1.e-10,NULL,1.e-10,NULL);CHKERRQ(ierr);            //set the tolerances for adaptive time stepping
    ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);                        //create the adaptive time stepping context
    ierr = TSAdaptSetType(adapt,TSADAPTBASIC);CHKERRQ(ierr);                //set the type of adaptivity

    /*
     * with the TSSetFromOptions() command every specification of the time stepper integration algorithm can be set from command line, which is incredibly useful
     * there are a vast number of different settings and solvers that may speed up convergence significantly, see petsc documentation for the available options
     * for instance you can compare the runtime of an adaptive vs a nonadaptive time integration algorithm, you can try other methods such as pseudo time stepping, see the petsc documentation
     */

    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);                        //magic set everything from command line function


    //set the master equation
    ierr = TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,NULL);CHKERRQ(ierr);    //use the time independent, matrix based
    ierr = TSSetRHSJacobian(ts,AA,AA,TSComputeRHSJacobianConstant,NULL);CHKERRQ(ierr);    //jacobian method

    //set the output
    ierr = TSMonitorSet(ts,MyOut::TEVMonitor,out,NULL);CHKERRQ(ierr);

    //security measure
    ierr = TSSetTimeStep(ts,0.01);CHKERRQ(ierr);                    //for oscillatory stuff, otherwise it crashes, because initial time step may be too large


    //solve it and write into the output
    ierr = TSSolve(ts,dm);CHKERRQ(ierr);                        //actual solution


    
    //clean up stage: free everything that is not needed anymore
    MatDestroy(&AA); 
    VecDestroy(&dm); 
    TSDestroy(&ts);
    delete out;
    
    PetscFinalize(); 
    return 0;
}

