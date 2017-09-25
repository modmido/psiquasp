
/**
 * @file	ex5.cpp
 * 
 * 		
 * 
 * @author	Michael Gegg
 * 
 */
 
#include"ex5.hpp"


/*
 * Setup for the Laser object. This function initializes the density matrix vector and the Liouvillian matrix that describes the master equation. 
 */

#undef __FUNCT__
#define __FUNCT__ "Setup"

PetscErrorCode Lambda::Setup(Vec* dm, Mat* LL, Mat* PP, PetscInt *dim)
{
    PetscFunctionBeginUser;				//also needed for petsc error handling
    PetscErrorCode	ierr;				//the return type of (almost) all petsc/slepc and psiquasp functions
    
    //parameters
    PetscInt		nmls = 2, dx21 = 2, dx20 = 2, dx10 = 2, m0=4, dm0=4;
    PetscReal		energy1 = 2.0;
    PetscReal		laserfreq = 1.9/hbar;
    PetscReal		delta0 = 0.0;
    PetscReal		delta1 = 0.0;

    
    //set simulation parameter to command line options
    ierr = PetscOptionsGetInt(NULL,NULL,"-nmls",&nmls,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-dx21",&dx21,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-dx20",&dx20,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-dx10",&dx10,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-m0",&m0,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-dm0",&dm0,NULL);CHKERRQ(ierr);
    
    ierr = PetscOptionsGetReal(NULL,NULL,"-laserfreq",&laserfreq,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-delta0",&delta0,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-delta1",&delta1,NULL);CHKERRQ(ierr);
    
    PetscReal		energy2 = delta1 + energy1 - hbar*laserfreq;		//rotating frame energy and detuning relations
    PetscReal		modeenergy = delta0 + energy1;				//...

    ierr = AddParam("energy1",energy1); CHKERRQ(ierr);
    ierr = AddParam("laserfreq",laserfreq); CHKERRQ(ierr);
    
    
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
    Mat		ADJ;
    Vec		dmfull;
    
    ierr = PQSPSetup(&dmfull,1,&ADJ);CHKERRQ(ierr);
    
    
    //write start values into the density matrix
    PetscInt	qnumbers[10] = {};
    qnumbers[5] = 1;
    ierr = DMWritePureState(dmfull,qnumbers);CHKERRQ(ierr);
    
    
    //Parmetis reuction of degrees of freedom
    ierr = SetLiouvillianOne(ADJ);CHKERRQ(ierr);
    ierr = ParmetisReduce(ADJ,PP,dim);CHKERRQ(ierr);
    
    
    //create downprojected Liouvillian
    Mat		LFULL,PPLFULL,PPT;
    
    ierr = PQSPCreateMat(&LFULL);CHKERRQ(ierr);
    ierr = SetLiouvillian(LFULL);CHKERRQ(ierr);
    ierr = MatMatMult(*PP,LFULL,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&PPLFULL);CHKERRQ(ierr);
    ierr = MatTranspose(*PP,MAT_INITIAL_MATRIX,&PPT);CHKERRQ(ierr);
    ierr = MatMatMult(PPLFULL,PPT,MAT_INITIAL_MATRIX,PETSC_DEFAULT,LL);CHKERRQ(ierr);
    
    
    //create downprojected density matrix vector
    ierr = VecCreate(PETSC_COMM_WORLD,dm); CHKERRQ(ierr);
    ierr = VecSetSizes(*dm,PETSC_DECIDE,*dim); CHKERRQ(ierr);
    ierr = VecSetUp(*dm); CHKERRQ(ierr);
    
    
    //print old and new Liouville space dimension into stdout
    PetscInt	size1,size2;
    ierr = VecGetSize(dmfull,&size1);CHKERRQ(ierr);
    ierr = VecGetSize(*dm,&size2);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Vec sizes %d \t %d\n",(int) size1, (int) size2);CHKERRQ(ierr);
    
    ierr = MatMult(*PP,dmfull,*dm);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SetLiouvillian"

PetscErrorCode Lambda::SetLiouvillian(Mat LFULL)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    
    PetscReal		delta0 = 0.0;
    PetscReal		delta1 = 0.0;
    PetscReal		gcouple = 0.001;
    PetscReal		edrive = 0.001;
    PetscReal		gamma = 0.000001;
    PetscReal		gammap = 0.000001;
    PetscReal		kappa = 0.01;

    
    ierr = PetscOptionsGetReal(NULL,NULL,"-delta0",&delta0,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-delta1",&delta1,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-gcouple",&gcouple,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-edrive",&edrive,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-gamma",&gamma,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-gammap",&gammap,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-kappa",&kappa,NULL);CHKERRQ(ierr);
    
    ierr = AddParam("delta0",delta0); CHKERRQ(ierr);
    ierr = AddParam("delta1",delta1); CHKERRQ(ierr);
    ierr = AddParam("gcouple",gcouple); CHKERRQ(ierr);
    ierr = AddParam("edrive",edrive); CHKERRQ(ierr);
    ierr = AddParam("gamma",gamma); CHKERRQ(ierr);
    ierr = AddParam("gammap",gammap); CHKERRQ(ierr);
    ierr = AddParam("kappa",kappa); CHKERRQ(ierr);
    
    
    MLSDim	n22 (2,2), n21 (2,1), n20 (2,0), n12 (1,2), n11 (1,1), n10 (1,0), n02 (0,2), n01 (0,1), n00 (0,0);		//first we create all MLS and Mode dimension identifiers
    ModeDim	mket (0,0), mbra (1,0);
    
    
    //preallocation stage, this is crucial when there are many different contributions in the master equation since otherwise the matrix setup becomes super slow
    PetscInt	*d_nnz	= new PetscInt [loc_size] ();						//estimate for the number of local elements per row, 
    PetscInt	*o_nnz	= new PetscInt [loc_size] ();						//estimate for the number of nonlocal elements per row, only exists if multiprocessor mode is used
    
    ierr = AddMLSH0(LFULL,d_nnz,o_nnz,0,n21,1.0);CHKERRQ(ierr);					//detuning contribution for possible detuning between laser frequency and 1 <-> 2 transition frequency
    ierr = AddMLSH0(LFULL,d_nnz,o_nnz,0,n20,1.0);CHKERRQ(ierr);					//....
    ierr = AddModeH0(LFULL,d_nnz,o_nnz,0,0,1.0);CHKERRQ(ierr);					//detuning between mode and 1 <-> 0 transition
    
    ierr = AddMLSModeInt(LFULL,d_nnz,o_nnz,0,n02,n12,mket,1.0);CHKERRQ(ierr);			//mode - mls interaction, mode ket and left index change
    ierr = AddMLSModeInt(LFULL,d_nnz,o_nnz,0,n01,n11,mket,1.0);CHKERRQ(ierr);			//each function call represents one twoheaded arrow in the sketches, connecting two bubbles
    ierr = AddMLSModeInt(LFULL,d_nnz,o_nnz,0,n00,n10,mket,1.0);CHKERRQ(ierr);
    ierr = AddMLSModeInt(LFULL,d_nnz,o_nnz,0,n20,n21,mbra,1.0);CHKERRQ(ierr);			//mode bra and right index change
    ierr = AddMLSModeInt(LFULL,d_nnz,o_nnz,0,n10,n11,mbra,1.0);CHKERRQ(ierr);
    ierr = AddMLSModeInt(LFULL,d_nnz,o_nnz,0,n00,n01,mbra,1.0);CHKERRQ(ierr);
    
    ierr = AddMLSCohDrive(LFULL,d_nnz,o_nnz,0,n22,n12,1.0);CHKERRQ(ierr);				//coherent drive of the 1 <-> transition
    ierr = AddMLSCohDrive(LFULL,d_nnz,o_nnz,0,n21,n11,1.0);CHKERRQ(ierr);
    ierr = AddMLSCohDrive(LFULL,d_nnz,o_nnz,0,n20,n10,1.0);CHKERRQ(ierr);
    ierr = AddMLSCohDrive(LFULL,d_nnz,o_nnz,0,n22,n21,1.0);CHKERRQ(ierr);
    ierr = AddMLSCohDrive(LFULL,d_nnz,o_nnz,0,n12,n11,1.0);CHKERRQ(ierr);
    ierr = AddMLSCohDrive(LFULL,d_nnz,o_nnz,0,n02,n01,1.0);CHKERRQ(ierr);
    
    ierr = AddLindbladRelaxMLS(LFULL,d_nnz,o_nnz,0,n11,n00,1.0);CHKERRQ(ierr);			//two spontaneous emission contributions
    ierr = AddLindbladRelaxMLS(LFULL,d_nnz,o_nnz,0,n11,n22,1.0);CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(LFULL,d_nnz,o_nnz,0,n10,1.0);CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(LFULL,d_nnz,o_nnz,0,n01,1.0);CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(LFULL,d_nnz,o_nnz,0,n12,1.0);CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(LFULL,d_nnz,o_nnz,0,n21,1.0);CHKERRQ(ierr);
    
    ierr = AddLindbladMode(LFULL,d_nnz,o_nnz,0,0,1.0);CHKERRQ(ierr);				//finite mode lifetime
    
    ierr = MatMPIAIJSetPreallocation(LFULL,0,d_nnz,0,o_nnz); CHKERRQ(ierr);			//parallel
    ierr = MatSeqAIJSetPreallocation(LFULL,0,d_nnz); CHKERRQ(ierr);
    
    
    ierr = AddMLSH0(LFULL,d_nnz,o_nnz,1,n21,delta1*PETSC_i);CHKERRQ(ierr);				//detuning contribution for possible detuning between laser frequency and 1 <-> 2 transition frequency
    ierr = AddMLSH0(LFULL,d_nnz,o_nnz,1,n20,delta1*PETSC_i);CHKERRQ(ierr);				//....
    ierr = AddModeH0(LFULL,d_nnz,o_nnz,1,0,delta0*PETSC_i);CHKERRQ(ierr);					//detuning between mode and 1 <-> 0 transition
    
    ierr = AddMLSModeInt(LFULL,d_nnz,o_nnz,1,n02,n12,mket,gcouple*PETSC_i);CHKERRQ(ierr);			//mode - mls interaction, ket and left index change
    ierr = AddMLSModeInt(LFULL,d_nnz,o_nnz,1,n01,n11,mket,gcouple*PETSC_i);CHKERRQ(ierr);
    ierr = AddMLSModeInt(LFULL,d_nnz,o_nnz,1,n00,n10,mket,gcouple*PETSC_i);CHKERRQ(ierr);
    ierr = AddMLSModeInt(LFULL,d_nnz,o_nnz,1,n20,n21,mbra,-gcouple*PETSC_i);CHKERRQ(ierr);			//minus sign due to the commutator
    ierr = AddMLSModeInt(LFULL,d_nnz,o_nnz,1,n10,n11,mbra,-gcouple*PETSC_i);CHKERRQ(ierr);
    ierr = AddMLSModeInt(LFULL,d_nnz,o_nnz,1,n00,n01,mbra,-gcouple*PETSC_i);CHKERRQ(ierr);
    
    ierr = AddMLSCohDrive(LFULL,d_nnz,o_nnz,1,n22,n12,edrive*PETSC_i);CHKERRQ(ierr);			//coherent drive of the 1 <-> transition
    ierr = AddMLSCohDrive(LFULL,d_nnz,o_nnz,1,n21,n11,edrive*PETSC_i);CHKERRQ(ierr);
    ierr = AddMLSCohDrive(LFULL,d_nnz,o_nnz,1,n20,n10,edrive*PETSC_i);CHKERRQ(ierr);
    ierr = AddMLSCohDrive(LFULL,d_nnz,o_nnz,1,n22,n21,-edrive*PETSC_i);CHKERRQ(ierr);
    ierr = AddMLSCohDrive(LFULL,d_nnz,o_nnz,1,n12,n11,-edrive*PETSC_i);CHKERRQ(ierr);
    ierr = AddMLSCohDrive(LFULL,d_nnz,o_nnz,1,n02,n01,-edrive*PETSC_i);CHKERRQ(ierr);
    
    ierr = AddLindbladRelaxMLS(LFULL,d_nnz,o_nnz,1,n11,n00,gamma/2.0);CHKERRQ(ierr);			//two spontaneous emission contributions
    ierr = AddLindbladRelaxMLS(LFULL,d_nnz,o_nnz,1,n11,n22,gammap/2.0);CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(LFULL,d_nnz,o_nnz,1,n10,(gamma+gammap)/2.0);CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(LFULL,d_nnz,o_nnz,1,n01,(gamma+gammap)/2.0);CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(LFULL,d_nnz,o_nnz,1,n12,(gamma+gammap)/2.0);CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(LFULL,d_nnz,o_nnz,1,n21,(gamma+gammap)/2.0);CHKERRQ(ierr);
    
    ierr = AddLindbladMode(LFULL,d_nnz,o_nnz,1,0,kappa/2.0);CHKERRQ(ierr);				//finite mode lifetime
    
    ierr = MatAssemblyBegin(LFULL,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);				//assemble the matrix
    ierr = MatAssemblyEnd(LFULL,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);				//PETSc requires this step, this is basically due to the sparse matrix format.
    
    delete[]	d_nnz;
    delete[]	o_nnz;
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SetLiouvillianOne"

PetscErrorCode Lambda::SetLiouvillianOne(Mat LFULL)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    
    PetscReal		delta0 = 0.0;
    PetscReal		delta1 = 0.0;
    PetscReal		gcouple = 0.001;
    PetscReal		edrive = 0.001;
    PetscReal		gamma = 0.000001;
    PetscReal		gammap = 0.000001;
    PetscReal		kappa = 0.01;

    
    ierr = PetscOptionsGetReal(NULL,NULL,"-delta0",&delta0,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-delta1",&delta1,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-gcouple",&gcouple,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-edrive",&edrive,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-gamma",&gamma,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-gammap",&gammap,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-kappa",&kappa,NULL);CHKERRQ(ierr);
    
    ierr = AddParam("delta0",delta0); CHKERRQ(ierr);
    ierr = AddParam("delta1",delta1); CHKERRQ(ierr);
    ierr = AddParam("gcouple",gcouple); CHKERRQ(ierr);
    ierr = AddParam("edrive",edrive); CHKERRQ(ierr);
    ierr = AddParam("gamma",gamma); CHKERRQ(ierr);
    ierr = AddParam("gammap",gammap); CHKERRQ(ierr);
    ierr = AddParam("kappa",kappa); CHKERRQ(ierr);
    
    
    MLSDim	n22 (2,2), n21 (2,1), n20 (2,0), n12 (1,2), n11 (1,1), n10 (1,0), n02 (0,2), n01 (0,1), n00 (0,0);		//first we create all MLS and Mode dimension identifiers
    ModeDim	mket (0,0), mbra (1,0);
    
    
    //preallocation stage, this is crucial when there are many different contributions in the master equation since otherwise the matrix setup becomes super slow
    PetscInt	*d_nnz	= new PetscInt [loc_size] ();						//estimate for the number of local elements per row, 
    PetscInt	*o_nnz	= new PetscInt [loc_size] ();						//estimate for the number of nonlocal elements per row, only exists if multiprocessor mode is used
    
    ierr = AddMLSH0(LFULL,d_nnz,o_nnz,0,n21,1.0);CHKERRQ(ierr);					//detuning contribution for possible detuning between laser frequency and 1 <-> 2 transition frequency
    ierr = AddMLSH0(LFULL,d_nnz,o_nnz,0,n20,1.0);CHKERRQ(ierr);					//....
    ierr = AddModeH0(LFULL,d_nnz,o_nnz,0,0,1.0);CHKERRQ(ierr);					//detuning between mode and 1 <-> 0 transition
    
    ierr = AddMLSModeInt(LFULL,d_nnz,o_nnz,0,n02,n12,mket,1.0);CHKERRQ(ierr);			//mode - mls interaction, mode ket and left index change
    ierr = AddMLSModeInt(LFULL,d_nnz,o_nnz,0,n01,n11,mket,1.0);CHKERRQ(ierr);			//each function call represents one twoheaded arrow in the sketches, connecting two bubbles
    ierr = AddMLSModeInt(LFULL,d_nnz,o_nnz,0,n00,n10,mket,1.0);CHKERRQ(ierr);
    ierr = AddMLSModeInt(LFULL,d_nnz,o_nnz,0,n20,n21,mbra,1.0);CHKERRQ(ierr);			//mode bra and right index change
    ierr = AddMLSModeInt(LFULL,d_nnz,o_nnz,0,n10,n11,mbra,1.0);CHKERRQ(ierr);
    ierr = AddMLSModeInt(LFULL,d_nnz,o_nnz,0,n00,n01,mbra,1.0);CHKERRQ(ierr);
    
    ierr = AddMLSCohDrive(LFULL,d_nnz,o_nnz,0,n22,n12,1.0);CHKERRQ(ierr);				//coherent drive of the 1 <-> transition
    ierr = AddMLSCohDrive(LFULL,d_nnz,o_nnz,0,n21,n11,1.0);CHKERRQ(ierr);
    ierr = AddMLSCohDrive(LFULL,d_nnz,o_nnz,0,n20,n10,1.0);CHKERRQ(ierr);
    ierr = AddMLSCohDrive(LFULL,d_nnz,o_nnz,0,n22,n21,1.0);CHKERRQ(ierr);
    ierr = AddMLSCohDrive(LFULL,d_nnz,o_nnz,0,n12,n11,1.0);CHKERRQ(ierr);
    ierr = AddMLSCohDrive(LFULL,d_nnz,o_nnz,0,n02,n01,1.0);CHKERRQ(ierr);
    
    ierr = AddLindbladRelaxMLS(LFULL,d_nnz,o_nnz,0,n11,n00,1.0);CHKERRQ(ierr);			//two spontaneous emission contributions
    ierr = AddLindbladRelaxMLS(LFULL,d_nnz,o_nnz,0,n11,n22,1.0);CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(LFULL,d_nnz,o_nnz,0,n10,1.0);CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(LFULL,d_nnz,o_nnz,0,n01,1.0);CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(LFULL,d_nnz,o_nnz,0,n12,1.0);CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(LFULL,d_nnz,o_nnz,0,n21,1.0);CHKERRQ(ierr);
    
    ierr = AddLindbladMode(LFULL,d_nnz,o_nnz,0,0,1.0);CHKERRQ(ierr);				//finite mode lifetime
    
    ierr = MatMPIAIJSetPreallocation(LFULL,0,d_nnz,0,o_nnz); CHKERRQ(ierr);			//parallel
    ierr = MatSeqAIJSetPreallocation(LFULL,0,d_nnz); CHKERRQ(ierr);
    
    ierr = AddMLSH0(LFULL,d_nnz,o_nnz,1,n21,1.0);CHKERRQ(ierr);				//detuning contribution for possible detuning between laser frequency and 1 <-> 2 transition frequency
    ierr = AddMLSH0(LFULL,d_nnz,o_nnz,1,n20,1.0);CHKERRQ(ierr);				//....
    ierr = AddModeH0(LFULL,d_nnz,o_nnz,1,0,1.0);CHKERRQ(ierr);					//detuning between mode and 1 <-> 0 transition
    
    ierr = AddMLSModeInt(LFULL,d_nnz,o_nnz,1,n02,n12,mket,1.0);CHKERRQ(ierr);			//mode - mls interaction, ket and left index change
    ierr = AddMLSModeInt(LFULL,d_nnz,o_nnz,1,n01,n11,mket,1.0);CHKERRQ(ierr);
    ierr = AddMLSModeInt(LFULL,d_nnz,o_nnz,1,n00,n10,mket,1.0);CHKERRQ(ierr);
    ierr = AddMLSModeInt(LFULL,d_nnz,o_nnz,1,n20,n21,mbra,1.0);CHKERRQ(ierr);			//minus sign due to the commutator
    ierr = AddMLSModeInt(LFULL,d_nnz,o_nnz,1,n10,n11,mbra,1.0);CHKERRQ(ierr);
    ierr = AddMLSModeInt(LFULL,d_nnz,o_nnz,1,n00,n01,mbra,1.0);CHKERRQ(ierr);
    
    ierr = AddMLSCohDrive(LFULL,d_nnz,o_nnz,1,n22,n12,1.0);CHKERRQ(ierr);			//coherent drive of the 1 <-> transition
    ierr = AddMLSCohDrive(LFULL,d_nnz,o_nnz,1,n21,n11,1.0);CHKERRQ(ierr);
    ierr = AddMLSCohDrive(LFULL,d_nnz,o_nnz,1,n20,n10,1.0);CHKERRQ(ierr);
    ierr = AddMLSCohDrive(LFULL,d_nnz,o_nnz,1,n22,n21,1.0);CHKERRQ(ierr);
    ierr = AddMLSCohDrive(LFULL,d_nnz,o_nnz,1,n12,n11,1.0);CHKERRQ(ierr);
    ierr = AddMLSCohDrive(LFULL,d_nnz,o_nnz,1,n02,n01,1.0);CHKERRQ(ierr);
    
    ierr = AddLindbladRelaxMLS(LFULL,d_nnz,o_nnz,1,n11,n00,1.0);CHKERRQ(ierr);			//two spontaneous emission contributions
    ierr = AddLindbladRelaxMLS(LFULL,d_nnz,o_nnz,1,n11,n22,1.0);CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(LFULL,d_nnz,o_nnz,1,n10,1.0);CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(LFULL,d_nnz,o_nnz,1,n01,1.0);CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(LFULL,d_nnz,o_nnz,1,n12,1.0);CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(LFULL,d_nnz,o_nnz,1,n21,1.0);CHKERRQ(ierr);
    
    ierr = AddLindbladMode(LFULL,d_nnz,o_nnz,1,0,1.0);CHKERRQ(ierr);				//finite mode lifetime
    
    ierr = MatAssemblyBegin(LFULL,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);				//assemble the matrix
    ierr = MatAssemblyEnd(LFULL,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);				//PETSc requires this step, this is basically due to the sparse matrix format.
    
    delete[]	d_nnz;
    delete[]	o_nnz;
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ParmetisReduce"

PetscErrorCode Lambda::ParmetisReduce(Mat ADJ, Mat* PP, PetscInt* dim)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    
    
    //create a symmetric connectivity matrix
    Mat			TT,SYM;
    
    ierr = MatTranspose(ADJ,MAT_INITIAL_MATRIX,&TT); CHKERRQ(ierr);
    ierr = MatDuplicate(ADJ,MAT_COPY_VALUES,&SYM); CHKERRQ(ierr);
    ierr = MatAXPY(SYM,1.0,TT,DIFFERENT_NONZERO_PATTERN);
    
//     ierr = MatView(SYM,PETSC_VIEWER_STDOUT_SELF);
    
    //start the repartitioning
    MatPartitioning	part;
    IS			is;
    PetscInt		cut = 0, npart = 2;
    
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nStarting repartitioning:\n"); CHKERRQ(ierr);
    
    while(!cut)
    {
      //create partitioner
      ierr = MatPartitioningCreate(PETSC_COMM_WORLD,&part); CHKERRQ(ierr);
      ierr = MatPartitioningSetAdjacency(part,SYM); CHKERRQ(ierr);
      ierr = MatPartitioningSetNParts(part,npart); CHKERRQ(ierr);
      ierr = MatPartitioningSetFromOptions(part); CHKERRQ(ierr);
      
      //do the partitioning
      ierr = MatPartitioningApply(part,&is); CHKERRQ(ierr);
    
      //get edge cuts
      ierr = MatPartitioningParmetisGetEdgeCut(part,&cut); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of cuts %d\n",(int) cut);CHKERRQ(ierr);
      
      //destroy
      ierr = MatPartitioningDestroy(&part);CHKERRQ(ierr);
      npart++;
    }
    
    npart -= 2;
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of partitions used %d\n",(int) npart);CHKERRQ(ierr);
    
    ierr = MatPartitioningCreate(PETSC_COMM_WORLD,&part);CHKERRQ(ierr);
    ierr = MatPartitioningSetAdjacency(part,SYM);CHKERRQ(ierr);
    ierr = MatPartitioningSetNParts(part,npart);CHKERRQ(ierr);
    ierr = MatPartitioningSetFromOptions(part);CHKERRQ(ierr);
    ierr = MatPartitioningApply(part,&is);CHKERRQ(ierr);
    
    ierr = MatPartitioningParmetisGetEdgeCut(part,&cut);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of cuts %d\n",(int) cut);CHKERRQ(ierr);
    if(cut)
    {
      (*PetscErrorPrintf)("Partitioning messed up!\n");
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
    }
    
    ierr = MatPartitioningDestroy(&part);CHKERRQ(ierr);
    
    ierr = MatProject(is,PP,dim);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"New dimension %d\n",(int) *dim);CHKERRQ(ierr);
    
    
    //clean up
    ierr = MatDestroy(&TT);CHKERRQ(ierr);
    ierr = MatDestroy(&SYM);CHKERRQ(ierr);
    ierr = MatPartitioningDestroy(&part);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ParmetisReduce2"

PetscErrorCode Lambda::ParmetisReduce2(Vec dmfull, Mat LFULL, Vec* dm, Mat* LL, Mat* PP, PetscInt* dim)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    
    
    //create a symmetric connectivity matrix
    Mat			TT,SYM;
    
    ierr = MatTranspose(LFULL,MAT_INITIAL_MATRIX,&TT); CHKERRQ(ierr);
    ierr = MatDuplicate(LFULL,MAT_COPY_VALUES,&SYM); CHKERRQ(ierr);
    ierr = MatAXPY(SYM,1.0,TT,DIFFERENT_NONZERO_PATTERN);
    
    
    //start the repartitioning
    MatPartitioning	part;
    IS			is;
    PetscInt		i,cut = 0, npart = 2;
    
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nStarting repartitioning:\n"); CHKERRQ(ierr);
    
    for(i=2; i < 40; i++)
    {
      //create partitioner
      ierr = MatPartitioningCreate(PETSC_COMM_WORLD,&part); CHKERRQ(ierr);
      ierr = MatPartitioningSetAdjacency(part,SYM); CHKERRQ(ierr);
      ierr = MatPartitioningSetNParts(part,i); CHKERRQ(ierr);
      ierr = MatPartitioningSetFromOptions(part); CHKERRQ(ierr);
      
      //do the partitioning
      ierr = MatPartitioningApply(part,&is); CHKERRQ(ierr);
    
      //get edge cuts
      ierr = MatPartitioningParmetisGetEdgeCut(part,&cut); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of cuts %d\n",(int) cut);CHKERRQ(ierr);
      if( !cut )		npart = i;
      
      //destroy
      ierr = MatPartitioningDestroy(&part);CHKERRQ(ierr);
    }
    
//     npart -= 2;
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of partitions used %d\n",(int) npart);CHKERRQ(ierr);
    
    ierr = MatPartitioningCreate(PETSC_COMM_WORLD,&part);CHKERRQ(ierr);
    ierr = MatPartitioningSetAdjacency(part,SYM);CHKERRQ(ierr);
    ierr = MatPartitioningSetNParts(part,npart);CHKERRQ(ierr);
    ierr = MatPartitioningSetFromOptions(part);CHKERRQ(ierr);
    ierr = MatPartitioningApply(part,&is);CHKERRQ(ierr);
    
    ierr = MatPartitioningParmetisGetEdgeCut(part,&cut);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of cuts %d\n",(int) cut);CHKERRQ(ierr);
    if(cut)
    {
      (*PetscErrorPrintf)("Partitioning messed up!\n");
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
    }
    
    ierr = MatPartitioningDestroy(&part);CHKERRQ(ierr);
    
    ierr = MatProject(is,PP,dim);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"New dimension %d\n",(int) *dim);CHKERRQ(ierr);
    
    
    //create downprojected Liouvillian
    Mat		PPLFULL,PPT;
    ierr = MatMatMult(*PP,LFULL,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&PPLFULL);CHKERRQ(ierr);
    ierr = MatTranspose(*PP,MAT_INITIAL_MATRIX,&PPT);CHKERRQ(ierr);
    ierr = MatMatMult(PPLFULL,PPT,MAT_INITIAL_MATRIX,PETSC_DEFAULT,LL);CHKERRQ(ierr);
    
    
    //create downprojected density matrix vector
    ierr = VecCreate(PETSC_COMM_WORLD,dm); CHKERRQ(ierr);
    ierr = VecSetSizes(*dm,PETSC_DECIDE,*dim); CHKERRQ(ierr);
    ierr = VecSetUp(*dm); CHKERRQ(ierr);
    
    PetscInt	size1,size2;
    ierr = VecGetSize(dmfull,&size1);CHKERRQ(ierr);
    ierr = VecGetSize(*dm,&size2);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Vec sizes %d \t %d\n",(int) size1, (int) size2);CHKERRQ(ierr);
    
    ierr = MatMult(*PP,dmfull,*dm);CHKERRQ(ierr);
    
    
    //clean up
    ierr = MatDestroy(&TT);CHKERRQ(ierr);
    ierr = MatDestroy(&SYM);CHKERRQ(ierr);
    ierr = MatDestroy(&PPLFULL);CHKERRQ(ierr);
    ierr = MatDestroy(&PPT);CHKERRQ(ierr);
    ierr = MatPartitioningDestroy(&part);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatProject"

PetscErrorCode Lambda::MatProject(IS is, Mat* AA, PetscInt *dim)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    
    //create downprojection matrix
    const PetscInt	*ptr;
    ierr = ISGetIndices(is,&ptr); CHKERRQ(ierr);
    PetscInt		partnumber = ptr[0];			//the partition containing the ground state should be used
    
    PetscInt		i,count = 1;				//how many entries in this partition?
    for(i=1; i < loc_size; i++)
    {
      if( ptr[i] == partnumber )	count++;
    }
    
    ierr = MatCreate(PETSC_COMM_WORLD,AA); CHKERRQ(ierr);
    ierr = MatSetSizes(*AA,PETSC_DECIDE,PETSC_DECIDE,count,loc_size); CHKERRQ(ierr);
    ierr = MatSetUp(*AA); CHKERRQ(ierr);
    
    PetscInt		count2 = 0;
    for(i=0; i < loc_size; i++)
    {
      if( ptr[i] == partnumber )
      {
	ierr = MatSetValue(*AA,count2,i,1.0,INSERT_VALUES);CHKERRQ(ierr);
	count2++;
      }
    }
    
    ierr = MatAssemblyBegin(*AA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*AA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    
    ierr = ISRestoreIndices(is,&ptr); CHKERRQ(ierr);    
    
    if( count == count2 )	*dim = count;
    else
    {
      (*PetscErrorPrintf)("Partitioning messed up!\n");
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_MIN_VALUE,"");
    }
    
    PetscFunctionReturn(0);
}


/*
 * Setup for MyOut object. Initializes the output files and adds them to the list.
 * There is no limit to the number of output files
 */

#undef __FUNCT__
#define __FUNCT__ "SetupMyOut"

PetscErrorCode MyOut::SetupMyOut(Lambda * system,Mat PP,PetscInt dim)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
  
    //allocate all files that should be in the output
    ObservablesFile	*obsfile	= new ObservablesFile;
    
    
    //initialization
    ierr = obsfile->SetupMyObsFile(system,"observables.dat",PP,dim);CHKERRQ(ierr);			//user specified (see below)
    ierr = AddOFile(obsfile);CHKERRQ(ierr);
    
    
    PetscFunctionReturn(0);
}


/*
 * Setup for the Observables file.
 */

#undef __FUNCT__
#define __FUNCT__ "SetupMyObsFile"

PetscErrorCode ObservablesFile::SetupMyObsFile(Lambda * system, std::string name, Mat PP,PetscInt dim)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    
    //open file, make gen. header
    ierr = SetOFile(system,name);CHKERRQ(ierr);		//Open File, set file name
    ierr = WriteSystemParameters(system);CHKERRQ(ierr);	//write system parameters into the output file
    
    
    //allocate all files that should be in the output
    MyObs	*trm1		= new MyObs();
    MyObs	*j11		= new MyObs();
    MyObs	*j22		= new MyObs();
    MyObs	*j10		= new MyObs();
    MyObs	*j21		= new MyObs();
    MyObs	*j20		= new MyObs();
    MyObs	*bdb		= new MyObs();
    MyObs	*b		= new MyObs();

    //create matrices
    Mat		II,J11,J22,J10,J21,J20,BdB,B;
    
    ierr = system->MatId(&II,PP,1.0);CHKERRQ(ierr);
    ierr = system->MatJ11Left(&J11,PP,1.0);CHKERRQ(ierr);
    ierr = system->MatJ22Left(&J22,PP,1.0);CHKERRQ(ierr);
    ierr = system->MatJ10Left(&J10,PP,1.0);CHKERRQ(ierr);
    ierr = system->MatJ21Left(&J21,PP,1.0);CHKERRQ(ierr);
    ierr = system->MatJ20Left(&J20,PP,1.0);CHKERRQ(ierr);
    ierr = system->MatBdBLeft(&BdB,PP,1.0);CHKERRQ(ierr);
    ierr = system->MatBLeft(&B,PP,1.0);CHKERRQ(ierr);
    
    
    //get the values necessary for the rotating frame frequencies
    PetscReal	energy1, laserfreq;
    ierr = system->GetParam("energy1",&energy1); CHKERRQ(ierr);
    ierr = system->GetParam("laserfreq",&laserfreq); CHKERRQ(ierr);
    
    
    //setup the observalbes and add them to the list
    ierr = trm1->Setup(system,II,PP,dim,1,1.0,.0);CHKERRQ(ierr);		//this computes tr(rho)-1
    ierr = AddElem(trm1,"tr-1\t");CHKERRQ(ierr);				//add it to the list and give it a name that is printed into the file header
    
    ierr = j11->Setup(system,J11,PP,dim,1,0.0,0.0);CHKERRQ(ierr);		//this computes <J11>
    ierr = AddElem(j11,"<J11>\t");CHKERRQ(ierr);				//add it to the list and give it a name that is printed into the file header
    
    ierr = j22->Setup(system,J22,PP,dim,1,0.0,0.0);CHKERRQ(ierr);		//this computes <J22>
    ierr = AddElem(j22,"<J22>\t");CHKERRQ(ierr);				//add it to the list and give it a name that is printed into the file header
    
    ierr = j10->Setup(system,J10,PP,dim,0,0.0,energy1/hbar);CHKERRQ(ierr);	//this computes <J10>
    ierr = AddElem(j10,"Re[<J10>]\tIm[<J10>]\t");CHKERRQ(ierr);			//add it to the list and give it a name that is printed into the file header
    
    ierr = j21->Setup(system,J21,PP,dim,0,0.0,-laserfreq);CHKERRQ(ierr);	//this computes <J21>
    ierr = AddElem(j21,"Re[<J21>]\tIm[<J21>]\t");CHKERRQ(ierr);			//add it to the list and give it a name that is printed into the file header
    
    ierr = j20->Setup(system,J20,PP,dim,0,0.0,energy1/hbar-laserfreq);CHKERRQ(ierr);//this computes <J20>
    ierr = AddElem(j20,"Re[<J20>]\tIm[<J20>]\t");CHKERRQ(ierr);			//add it to the list and give it a name that is printed into the file header
    
    ierr = bdb->Setup(system,BdB,PP,dim,1,0.0,0.0);CHKERRQ(ierr);		//this computes <BdB>
    ierr = AddElem(bdb,"<BdB>\t");CHKERRQ(ierr);				//add it to the list and give it a name that is printed into the file header
    
    ierr = b->Setup(system,B,PP,dim,0,0.0,-energy1/hbar);CHKERRQ(ierr);		//this computes <B>
    ierr = AddElem(b,"Re[<B>]\tIm[<B>]\t");CHKERRQ(ierr);			//add it to the list and give it a name that is printed into the file header
    
    ierr = MakeHeaderTEV();CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

/*
 * Setup for the MyObs class
 */

#undef __FUNCT__
#define __FUNCT__ "Setup"

PetscErrorCode MyObs::Setup(Lambda* system, Mat AA, Mat PP, PetscInt dim, PetscInt herm,PetscReal inshift,PetscReal inomega)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    Vec			tr,trproj;
    
    ierr = VecCreate(PETSC_COMM_WORLD,&trproj);CHKERRQ(ierr);
    ierr = VecSetSizes(trproj,PETSC_DECIDE,dim);CHKERRQ(ierr);
    ierr = VecSetUp(trproj);CHKERRQ(ierr);
    
    ierr = VecCreate(PETSC_COMM_WORLD,&left);CHKERRQ(ierr);
    ierr = VecSetSizes(left,PETSC_DECIDE,dim);CHKERRQ(ierr);
    ierr = VecSetUp(left);CHKERRQ(ierr);
    
    ierr = system->PQSPCreateVec(&tr,NULL,NULL);CHKERRQ(ierr);		//create vector for trace
    ierr = system->VecTrace(tr);CHKERRQ(ierr);				//write the trace into the vector
    ierr = MatMult(PP,tr,trproj);CHKERRQ(ierr);				//downprojection
    ierr = MatMultHermitianTranspose(AA,trproj,left); CHKERRQ(ierr);	//creat the left hand side vector for obs = <left|rho>

    isherm	= herm;							//sets the hermitian observable flag
    shift	= inshift;						//sets the shift value, this normally only needed for the tr(rho)-1 observable, which is a convergence check
    omega	= inomega;
    
    PetscFunctionReturn(0);
}

/*
 * Setup functions for the downprojected liouvillian matrices for the observables. Using the normal observalbes utilities does not work when using a downprojected density matrix
 * this would also work by just using JXY*PPd because the custom property is setu up with the command AAd |tr> and multiplying the matrix with the projector form the right, would result in the right format, however it would probably be more confusing
 * and using the full projection can also be used for constructing correaltion functions and everything else.
 */

#undef __FUNCT__
#define __FUNCT__ "MatId"

PetscErrorCode Lambda::MatId(Mat* AA, Mat PP, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    Mat			Full,Halffull,PPT;
    
    ierr = PQSPCreateMat(&Full);CHKERRQ(ierr);
    ierr = AddDiagOne(Full,NULL,1);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(Full,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);				//assemble the matrix
    ierr = MatAssemblyEnd(Full,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    
    ierr = MatMatMult(PP,Full,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Halffull);CHKERRQ(ierr);
    ierr = MatTranspose(PP,MAT_INITIAL_MATRIX,&PPT);CHKERRQ(ierr);
    ierr = MatMatMult(Halffull,PPT,MAT_INITIAL_MATRIX,PETSC_DEFAULT,AA);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatJ11Left"

PetscErrorCode Lambda::MatJ11Left(Mat* AA, Mat PP, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    
    MLSDim	n21 (2,1), n11 (1,1), n01 (0,1);
    Mat		Full,Halffull,PPT;
    
    ierr = PQSPCreateMat(&Full);CHKERRQ(ierr);
    ierr = AddMLSSingleArrowNonconnecting(Full,NULL,NULL,1,n21,matrixelem);CHKERRQ(ierr);
    ierr = AddMLSSingleArrowNonconnecting(Full,NULL,NULL,1,n11,matrixelem);CHKERRQ(ierr);
    ierr = AddMLSSingleArrowNonconnecting(Full,NULL,NULL,1,n01,matrixelem);CHKERRQ(ierr);
    
    ierr = MatAssemblyBegin(Full,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);				//assemble the matrix
    ierr = MatAssemblyEnd(Full,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    
    ierr = MatMatMult(PP,Full,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Halffull);CHKERRQ(ierr);
    ierr = MatTranspose(PP,MAT_INITIAL_MATRIX,&PPT);CHKERRQ(ierr);
    ierr = MatMatMult(Halffull,PPT,MAT_INITIAL_MATRIX,PETSC_DEFAULT,AA);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatJ22Left"

PetscErrorCode Lambda::MatJ22Left(Mat* AA, Mat PP, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    
    MLSDim	n22 (2,2), n12 (1,2), n02 (0,2);
    Mat		Full,Halffull,PPT;
    
    ierr = PQSPCreateMat(&Full);CHKERRQ(ierr);
    ierr = AddMLSSingleArrowNonconnecting(Full,NULL,NULL,1,n22,matrixelem);CHKERRQ(ierr);
    ierr = AddMLSSingleArrowNonconnecting(Full,NULL,NULL,1,n12,matrixelem);CHKERRQ(ierr);
    ierr = AddMLSSingleArrowNonconnecting(Full,NULL,NULL,1,n02,matrixelem);CHKERRQ(ierr);
    
    ierr = MatAssemblyBegin(Full,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);				//assemble the matrix
    ierr = MatAssemblyEnd(Full,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    
    ierr = MatMatMult(PP,Full,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Halffull);CHKERRQ(ierr);
    ierr = MatTranspose(PP,MAT_INITIAL_MATRIX,&PPT);CHKERRQ(ierr);
    ierr = MatMatMult(Halffull,PPT,MAT_INITIAL_MATRIX,PETSC_DEFAULT,AA);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatJ10Left"

PetscErrorCode Lambda::MatJ10Left(Mat* AA, Mat PP, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    
    MLSDim	n21 (2,1), n20 (2,0), n11 (1,1), n10 (1,0), n01 (0,1), n00 (0,0);
    Mat		Full,Halffull,PPT;
    
    ierr = PQSPCreateMat(&Full);CHKERRQ(ierr);

    ierr = AddMLSSingleArrowConnecting(Full,NULL,NULL,1,n00,n01,matrixelem); CHKERRQ(ierr);
    ierr = AddMLSSingleArrowConnecting(Full,NULL,NULL,1,n10,n11,matrixelem); CHKERRQ(ierr);
    ierr = AddMLSSingleArrowConnecting(Full,NULL,NULL,1,n20,n21,matrixelem); CHKERRQ(ierr);
    
    ierr = MatAssemblyBegin(Full,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);				//assemble the matrix
    ierr = MatAssemblyEnd(Full,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    
    ierr = MatMatMult(PP,Full,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Halffull);CHKERRQ(ierr);
    ierr = MatTranspose(PP,MAT_INITIAL_MATRIX,&PPT);CHKERRQ(ierr);
    ierr = MatMatMult(Halffull,PPT,MAT_INITIAL_MATRIX,PETSC_DEFAULT,AA);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatJ21Left"

PetscErrorCode Lambda::MatJ21Left(Mat* AA, Mat PP, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    
    MLSDim	n22 (2,2), n21 (2,1), n12 (1,2), n11 (1,1), n02 (0,2), n01 (0,1);
    Mat		Full,Halffull,PPT;
    
    ierr = PQSPCreateMat(&Full);CHKERRQ(ierr);

    ierr = AddMLSSingleArrowConnecting(Full,NULL,NULL,1,n01,n02,matrixelem); CHKERRQ(ierr);
    ierr = AddMLSSingleArrowConnecting(Full,NULL,NULL,1,n11,n12,matrixelem); CHKERRQ(ierr);
    ierr = AddMLSSingleArrowConnecting(Full,NULL,NULL,1,n21,n22,matrixelem); CHKERRQ(ierr);
    
    ierr = MatAssemblyBegin(Full,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);				//assemble the matrix
    ierr = MatAssemblyEnd(Full,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    
    ierr = MatMatMult(PP,Full,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Halffull);CHKERRQ(ierr);
    ierr = MatTranspose(PP,MAT_INITIAL_MATRIX,&PPT);CHKERRQ(ierr);
    ierr = MatMatMult(Halffull,PPT,MAT_INITIAL_MATRIX,PETSC_DEFAULT,AA);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatJ20Left"

PetscErrorCode Lambda::MatJ20Left(Mat* AA, Mat PP, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    
    MLSDim	n22 (2,2), n20 (2,0), n12 (1,2), n10 (1,0), n02 (0,2), n00 (0,0);
    Mat		Full,Halffull,PPT;
    
    ierr = PQSPCreateMat(&Full);CHKERRQ(ierr);

    ierr = AddMLSSingleArrowConnecting(Full,NULL,NULL,1,n00,n02,matrixelem); CHKERRQ(ierr);
    ierr = AddMLSSingleArrowConnecting(Full,NULL,NULL,1,n10,n12,matrixelem); CHKERRQ(ierr);
    ierr = AddMLSSingleArrowConnecting(Full,NULL,NULL,1,n20,n22,matrixelem); CHKERRQ(ierr);
    
    ierr = MatAssemblyBegin(Full,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);				//assemble the matrix
    ierr = MatAssemblyEnd(Full,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    
    ierr = MatMatMult(PP,Full,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Halffull);CHKERRQ(ierr);
    ierr = MatTranspose(PP,MAT_INITIAL_MATRIX,&PPT);CHKERRQ(ierr);
    ierr = MatMatMult(Halffull,PPT,MAT_INITIAL_MATRIX,PETSC_DEFAULT,AA);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatBdBLeft"

PetscErrorCode Lambda::MatBdBLeft(Mat* AA, Mat PP, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;

    Mat		Full,Halffull,PPT;
    
    ierr = PQSPCreateMat(&Full);CHKERRQ(ierr);

    ierr = AddModeLeftBdB(Full,NULL,NULL,1,0,matrixelem); CHKERRQ(ierr);
    
    ierr = MatAssemblyBegin(Full,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);				//assemble the matrix
    ierr = MatAssemblyEnd(Full,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    
    ierr = MatMatMult(PP,Full,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Halffull);CHKERRQ(ierr);
    ierr = MatTranspose(PP,MAT_INITIAL_MATRIX,&PPT);CHKERRQ(ierr);
    ierr = MatMatMult(Halffull,PPT,MAT_INITIAL_MATRIX,PETSC_DEFAULT,AA);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatBLeft"

PetscErrorCode Lambda::MatBLeft(Mat* AA, Mat PP, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;

    Mat		Full,Halffull,PPT;
    
    ierr = PQSPCreateMat(&Full);CHKERRQ(ierr);

    ierr = AddModeLeftB(Full,NULL,NULL,1,0,matrixelem); CHKERRQ(ierr);
    
    ierr = MatAssemblyBegin(Full,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);				//assemble the matrix
    ierr = MatAssemblyEnd(Full,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    
    ierr = MatMatMult(PP,Full,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Halffull);CHKERRQ(ierr);
    ierr = MatTranspose(PP,MAT_INITIAL_MATRIX,&PPT);CHKERRQ(ierr);
    ierr = MatMatMult(Halffull,PPT,MAT_INITIAL_MATRIX,PETSC_DEFAULT,AA);CHKERRQ(ierr);
    
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
    Mat			AA,PP;
    PetscInt		dim;
    
    
    //setup stage: using PsiQuaSP to initialize everything
    Lambda	lambda;							//system specification
    ierr = lambda.Setup(&dm,&AA,&PP,&dim);CHKERRQ(ierr);

    MyOut	*out = new MyOut;					//output specification
    ierr = out->SetupMyOut(&lambda,PP,dim);CHKERRQ(ierr);
    
    PetscPrintf(PETSC_COMM_WORLD,"Computing stage\n\n");
    
    
    //computing stage: use the Petsc stuff to solve the problem using explicit Runge-Kutta time integration
    TS		ts;							//time stepper
    TSAdapt	adapt;								//adaptive time step context
    
    //time step solver
    ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);			//create time stepper context
    ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);			//tell petsc that we solve a linear diff. eq.
    ierr = TSSetType(ts,TSRK);CHKERRQ(ierr);					//set the time stepper to runge kutta
    ierr = TSRKSetType(ts,TSRK3BS);CHKERRQ(ierr);				//set it to third order RK scheme of Bogacki-Shampine with 2nd order embedded method, this is an adaptive step width Runge-Kutta
    ierr = TSSetDuration(ts,100000,25000);CHKERRQ(ierr);			//set the maximum integration cycles and time
    ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);	//what to do if the final time is not exactly reached with the time stepper, in this case nothing
    
    //adaptivity context for time stepper
    ierr = TSSetTolerances(ts,1.e-10,NULL,1.e-10,NULL);CHKERRQ(ierr);	//set the tolerances for adaptive time stepping
    ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);				//create the adaptive time stepping context
    ierr = TSAdaptSetType(adapt,TSADAPTBASIC);CHKERRQ(ierr);			//set the type of adaptivity

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
    ierr = TSSetTimeStep(ts,2.);CHKERRQ(ierr);						//for oscillatory stuff, otherwise it crashes, probably because initial time step too large
    
    //solve it and write into the output files at every 30th time step
    ierr = TSSolve(ts,dm);CHKERRQ(ierr);						//seems like initial conditions and solution/time steps get the same Vec, convenient...


    
    //clean up stage: free everything that is not needed anymore
    MatDestroy(&AA); 
    MatDestroy(&PP); 
    VecDestroy(&dm); 
//     TSDestroy(&ts); 
    delete out;
    
    PetscFinalize(); 
    return 0;
}

