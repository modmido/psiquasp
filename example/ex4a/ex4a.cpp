
/**
 * @file	phononlaser.cpp
 *
 * @author	Michael Gegg
 *
 */

#include"ex4a.hpp"

#undef __FUNCT__
#define __FUNCT__ "Setup"

PetscErrorCode Phononlaser::Setup(Vec * dm, Mat * AA)
{
    PetscFunctionBeginUser;

    PetscErrorCode	ierr;


    //set simulation parameter default values
    PetscInt	nmls=2,dx=2, m0=1, dm0=1;


    //set simulation parameter to command line options
    ierr = PetscOptionsGetInt(NULL,NULL,"-NMLS",&nmls,NULL);CHKERRQ(ierr);			//this is the total number of two level systems
    ierr = PetscOptionsGetInt(NULL,NULL,"-dx",&dx,NULL);CHKERRQ(ierr);				//this is the order of allowed tls system offdiagonals, dx = N_MLS means no truncation, dx > N_MLS will likely crash
    ierr = PetscOptionsGetInt(NULL,NULL,"-m0max",&m0,NULL);CHKERRQ(ierr);			//the maximum phonon number state
    ierr = PetscOptionsGetInt(NULL,NULL,"-dm0max",&dm0,NULL);CHKERRQ(ierr);			//the maximum phonon density matrix offdiagonal order, for large m0 this should be in general smaller than m0


    //specify dimensions
    ierr = MLSAdd(nmls); CHKERRQ(ierr);
    ierr = MLSAddDens(1,nmls+1,2.0*hbar); CHKERRQ(ierr);					//this is the TLS density dimension
    ierr = MLSAddPol(1,0,dx+1); CHKERRQ(ierr);							//two types of TLS offdiagonals
    ierr = MLSAddPol(0,1,dx+1); CHKERRQ(ierr);							//... if you want more levels, just add more dimensions
    ierr = ModeAdd(m0+1,dm0,2.0*hbar); CHKERRQ(ierr);						//and this is the phononmode, if you want more phononmodes call this function multiple times...


    //initialize density matrix and Lindbladian matrix, create index, this is important, since it specifies all the parallel and sets up everything needed for the rest of the program
    ierr = PQSPSetup(dm,1,AA);CHKERRQ(ierr);


    //fill the dm with start values
    PetscInt	qnumbers [5] = {1,0,0,0,0};
    ierr = DMWritePureState(*dm,qnumbers);CHKERRQ(ierr);


    //set Lindbladian
    Mat		Coh, EP, Diss;

    ierr = PQSPCreateMat(&Coh); CHKERRQ(ierr);							//first we create the matrices needed for storage
    ierr = PQSPCreateMat(&EP); CHKERRQ(ierr);
    ierr = PQSPCreateMat(&Diss); CHKERRQ(ierr);

    ierr = H0Part(*AA); CHKERRQ(ierr);								//setup the matrices for the different parts in the master equation
    ierr = CoherentDrivePart(Coh); CHKERRQ(ierr);
    ierr = ElectronPhononPart(EP); CHKERRQ(ierr);						//this is the only part where the modular methods are really needed but we also use it for the other contributions in this example
    ierr = DissipationPart(Diss); CHKERRQ(ierr);

    ierr = MatAXPY(*AA,1.0,Coh,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);			//add them all together *AA += 1.0*Coh ...
    ierr = MatAXPY(*AA,1.0,EP,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
    ierr = MatAXPY(*AA,1.0,Diss,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);

    ierr = MatSetOption(*AA,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE); CHKERRQ(ierr);		//tell Petsc that we will not add any elements at new positions, maybe unnecessary
    
    
    //clean up
    ierr = MatDestroy(&Coh); CHKERRQ(ierr);
    ierr = MatDestroy(&EP); CHKERRQ(ierr);
    ierr = MatDestroy(&Diss); CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\n\n\nSetup stage completed.\n\n\n\n"); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "H0Part"

PetscErrorCode	Phononlaser::H0Part(Mat AA)
{
    PetscFunctionBeginUser;

    PetscErrorCode	ierr;


    //parameters
    PetscReal	domega_tls	= -0.011;									//default
    PetscReal	omega_ph	= 0.011;

    ierr = PetscOptionsGetReal(NULL,NULL,"-domega_tls",&domega_tls,NULL);CHKERRQ(ierr);			//command line
    ierr = PetscOptionsGetReal(NULL,NULL,"-omega_ph",&omega_ph,NULL);CHKERRQ(ierr);
    ierr = AddParam("domega_tls",domega_tls);CHKERRQ(ierr);

    //set Lindbladian
    PetscInt	*d_nnz	= new PetscInt [loc_size] ();
    PetscInt	*o_nnz	= new PetscInt [loc_size] ();

    ierr = AddDiagZeros(AA,d_nnz,0); CHKERRQ(ierr);					//this is necessary since the Petsc solvers need the diagonal of the sparse matrix to be set, even if the elements are zero, otherwise it crashes...
    ierr = TLS_J11right(AA,d_nnz,o_nnz,0,1.0); CHKERRQ(ierr);				//preassembly, count the diagonal and offdiagonal (Petsc style) elements per row
    ierr = TLS_J11left(AA,d_nnz,o_nnz,0,1.0); CHKERRQ(ierr);
    ierr = AddModeRightBdB(AA,d_nnz,o_nnz,0,0,1.0); CHKERRQ(ierr);
    ierr = AddModeLeftBdB(AA,d_nnz,o_nnz,0,0,1.0); CHKERRQ(ierr);

    ierr = MatMPIAIJSetPreallocation(AA,0,d_nnz,0,o_nnz); CHKERRQ(ierr);		//actual preassembly, tells Petsc approx. how many entries are in the matrix, this is a speedup step
    ierr = MatSeqAIJSetPreallocation(AA,0,d_nnz); CHKERRQ(ierr);

    ierr = AddDiagZeros(AA,d_nnz,1); CHKERRQ(ierr);
    ierr = TLS_J11right(AA,d_nnz,o_nnz,1,PETSC_i*domega_tls); CHKERRQ(ierr);		//set the actual entries into the matrix
    ierr = TLS_J11left(AA,d_nnz,o_nnz,1,-PETSC_i*domega_tls); CHKERRQ(ierr);
    ierr = AddModeRightBdB(AA,d_nnz,o_nnz,1,0,PETSC_i*omega_ph); CHKERRQ(ierr);
    ierr = AddModeLeftBdB(AA,d_nnz,o_nnz,1,0,-PETSC_i*omega_ph); CHKERRQ(ierr);

    ierr = MatAssemblyBegin(AA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);			//Assemble the matrix, this is a step required by Petsc
    ierr = MatAssemblyEnd(AA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);


    //clean up
    delete[] d_nnz;
    delete[] o_nnz;

    PetscFunctionReturn(0);

}


#undef __FUNCT__
#define __FUNCT__ "CoherentDrivePart"

PetscErrorCode	Phononlaser::CoherentDrivePart(Mat AA)
{
    PetscFunctionBeginUser;

    PetscErrorCode	ierr;


    //parameters
    PetscReal	gdrive		= 0.001;							//default
    ierr = PetscOptionsGetReal(NULL,NULL,"-gdrive",&gdrive,NULL);CHKERRQ(ierr);			//command line


    //set the frist half of the Lindbladian, i.e. gdrive*(rho J_10 - J_10 rho)
    PetscInt	*d_nnz	= new PetscInt [loc_size] ();
    PetscInt	*o_nnz	= new PetscInt [loc_size] ();

    ierr = TLS_J10right(AA,d_nnz,o_nnz,0,1.0); CHKERRQ(ierr);					//preassembly, tells Petsc approx. how many entries are in the matrix, this is a speedup step
    ierr = TLS_J10left(AA,d_nnz,o_nnz,0,1.0); CHKERRQ(ierr);
    ierr = TLS_J01right(AA,d_nnz,o_nnz,0,1.0); CHKERRQ(ierr);
    ierr = TLS_J01left(AA,d_nnz,o_nnz,0,1.0); CHKERRQ(ierr);

    ierr = MatMPIAIJSetPreallocation(AA,0,d_nnz,0,o_nnz); CHKERRQ(ierr);			//actual preassembly, tells Petsc approx. how many entries are in the matrix, this is a speedup step
    ierr = MatSeqAIJSetPreallocation(AA,0,d_nnz); CHKERRQ(ierr);

    ierr = TLS_J10right(AA,d_nnz,o_nnz,1,PETSC_i*gdrive); CHKERRQ(ierr);
    ierr = TLS_J10left(AA,d_nnz,o_nnz,1,-PETSC_i*gdrive); CHKERRQ(ierr);
    ierr = TLS_J01right(AA,d_nnz,o_nnz,1,PETSC_i*gdrive); CHKERRQ(ierr);
    ierr = TLS_J01left(AA,d_nnz,o_nnz,1,-PETSC_i*gdrive); CHKERRQ(ierr);

    ierr = MatAssemblyBegin(AA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);				//Assemble the matrix, this is a step required by Petsc
    ierr = MatAssemblyEnd(AA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);


    //clean up
    delete[] d_nnz;
    delete[] o_nnz;

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "NoRWA1"

PetscErrorCode Phononlaser::NoRWA1(Mat AA)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;

    Mat		JLeft, JRight, bdbL, bdbR;
    
    ierr =  PQSPCreateMat(&JLeft);CHKERRQ(ierr);
    ierr =  PQSPCreateMat(&JRight);CHKERRQ(ierr);
    ierr =  PQSPCreateMat(&bdbL);CHKERRQ(ierr);
    ierr =  PQSPCreateMat(&bdbR);CHKERRQ(ierr);

    ierr = TLS_J10right(JRight,NULL,NULL,1,1.0); CHKERRQ(ierr);					//preassembly, tells Petsc approx. how many entries are in the matrix, this is a speedup step
    ierr = TLS_J01right(JRight,NULL,NULL,1,1.0); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(JRight,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);				//Assemble the matrix, this is a step required by Petsc
    ierr = MatAssemblyEnd(JRight,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    ierr = TLS_J10left(JLeft,NULL,NULL,1,1.0); CHKERRQ(ierr);
    ierr = TLS_J01left(JLeft,NULL,NULL,1,1.0); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(JLeft,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);				//Assemble the matrix, this is a step required by Petsc
    ierr = MatAssemblyEnd(JLeft,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    ierr = AddModeLeftB(bdbL,NULL,NULL,1,0,1.0); CHKERRQ(ierr);
    ierr = AddModeLeftBd(bdbL,NULL,NULL,1,0,1.0); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(bdbL,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);			//Assemble the matrix, this is a step required by Petsc
    ierr = MatAssemblyEnd(bdbL,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    
    ierr = AddModeRightB(bdbR,NULL,NULL,1,0,1.0); CHKERRQ(ierr);
    ierr = AddModeRightBd(bdbR,NULL,NULL,1,0,1.0); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(bdbR,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);			//Assemble the matrix, this is a step required by Petsc
    ierr = MatAssemblyEnd(bdbR,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    //set them together
    Mat		R,L;
    
    ierr = MatMatMult(JRight,bdbR,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&R); CHKERRQ(ierr);	//R = i*gcouple*rho*J_11*(b+bd), multiply them to get the coupling
    ierr = MatMatMult(JLeft,bdbL,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&L); CHKERRQ(ierr);	//L = -i*gcouple*J_11*(b+bd)*rho
    ierr = MatCopy(R,AA,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);				//AA = B
    ierr = MatAXPY(AA,-1.0,L,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
    
    
    //clean up

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NoRWA2"

PetscErrorCode Phononlaser::NoRWA2(Mat AA)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;
    
    MLSDim		n11 (1,1);
    MLSDim		n10 (1,0);
    MLSDim		n01 (0,1);
    MLSDim		n00 (0,0);
    ModeDim		mket (0,0);
    ModeDim		mbra (1,0);
    
    ierr = AddMLSModeInt(AA,NULL,NULL,1,n10,n11,mbra,-1);CHKERRQ(ierr);
    ierr = AddMLSModeInt(AA,NULL,NULL,1,n11,n10,mbra,-1);CHKERRQ(ierr);

    ierr = AddMLSModeInt(AA,NULL,NULL,1,n01,n11,mket,1);CHKERRQ(ierr);
    ierr = AddMLSModeInt(AA,NULL,NULL,1,n11,n01,mket,1);CHKERRQ(ierr);

    ierr = AddMLSModeInt(AA,NULL,NULL,1,n00,n10,mket,1);CHKERRQ(ierr);
    ierr = AddMLSModeInt(AA,NULL,NULL,1,n10,n00,mket,1);CHKERRQ(ierr);

    ierr = AddMLSModeInt(AA,NULL,NULL,1,n00,n01,mbra,-1);CHKERRQ(ierr);
    ierr = AddMLSModeInt(AA,NULL,NULL,1,n01,n00,mbra,-1);CHKERRQ(ierr);
    
    ierr = MatAssemblyBegin(AA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);			//Assemble the matrix, this is a step required by Petsc
    ierr = MatAssemblyEnd(AA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);   
}

#undef __FUNCT__
#define __FUNCT__ "ElectronPhononPart"

PetscErrorCode	Phononlaser::ElectronPhononPart(Mat AA)
{
    PetscFunctionBeginUser;

    PetscErrorCode	ierr;


    //parameters
    PetscReal	gcouple		= 0.005;							//default
    ierr = PetscOptionsGetReal(NULL,NULL,"-gcouple",&gcouple,NULL);CHKERRQ(ierr);		//command line


    //set the basic required matrices
    Mat		J11R,J11L,BplusBdR,BplusBdL;

    ierr = PQSPCreateMat(&J11R); CHKERRQ(ierr);
    ierr = PQSPCreateMat(&J11L); CHKERRQ(ierr);
    ierr = PQSPCreateMat(&BplusBdR); CHKERRQ(ierr);
    ierr = PQSPCreateMat(&BplusBdL); CHKERRQ(ierr);

    ierr = TLS_J11right(J11R,NULL,NULL,1,PETSC_i*gcouple); CHKERRQ(ierr);			//i*gcouple*rho*J_11, preassembly is not neccessary for matrices with very few entries
    ierr = MatAssemblyBegin(J11R,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);				//Assemble the matrix
    ierr = MatAssemblyEnd(J11R,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    ierr = TLS_J11left(J11L,NULL,NULL,1,PETSC_i*gcouple); CHKERRQ(ierr);   			//i*gcouple*J_11*rho
    ierr = MatAssemblyBegin(J11L,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J11L,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    ierr = AddModeRightB(BplusBdR,NULL,NULL,1,0,1.0); CHKERRQ(ierr);				//rho*b
    ierr = AddModeRightBd(BplusBdR,NULL,NULL,1,0,1.0); CHKERRQ(ierr);				//rho*b + rho*bd
    ierr = MatAssemblyBegin(BplusBdR,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(BplusBdR,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    ierr = AddModeLeftB(BplusBdL,NULL,NULL,1,0,1.0); CHKERRQ(ierr);				//b*rho
    ierr = AddModeLeftBd(BplusBdL,NULL,NULL,1,0,1.0); CHKERRQ(ierr);				//b*rho + bd*rho
    ierr = MatAssemblyBegin(BplusBdL,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(BplusBdL,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);


    //set them together
    Mat		R,L;

    ierr = MatMatMult(J11R,BplusBdR,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&R); CHKERRQ(ierr);	//R = i*gcouple*rho*J_11*(b+bd), multiply them to get the coupling
    ierr = MatMatMult(J11L,BplusBdL,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&L); CHKERRQ(ierr);	//L = i*gcouple*J_11*(b+bd)*rho
    ierr = MatCopy(R,AA,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);				//AA = B
    ierr = MatAXPY(AA,-1.0,L,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);				//AA -= L


    //clean up
    ierr = MatDestroy(&J11R); CHKERRQ(ierr);
    ierr = MatDestroy(&J11L); CHKERRQ(ierr);
    ierr = MatDestroy(&BplusBdR); CHKERRQ(ierr);
    ierr = MatDestroy(&BplusBdL); CHKERRQ(ierr);
    ierr = MatDestroy(&R); CHKERRQ(ierr);
    ierr = MatDestroy(&L); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DissipationPart"

PetscErrorCode	Phononlaser::DissipationPart(Mat AA)
{
    PetscFunctionBeginUser;

    PetscErrorCode	ierr;


    //parameters
    PetscReal	gamma		= 0.0002;							//default
    PetscReal	delta		= 0.00000;
    PetscReal	kappa		= 0.5;

    ierr = PetscOptionsGetReal(NULL,NULL,"-gamma",&gamma,NULL);CHKERRQ(ierr);			//command line
    ierr = PetscOptionsGetReal(NULL,NULL,"-delta",&delta,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-kappa",&kappa,NULL);CHKERRQ(ierr);


    //set the Lindbladian
    MLSDim	n11 (1,1), n10 (1,0), n01 (0,1), n00 (0,0);

    PetscInt	*d_nnz	= new PetscInt [loc_size] ();
    PetscInt	*o_nnz	= new PetscInt [loc_size] ();

    ierr = AddModeLeftBRightBd(AA,d_nnz,o_nnz,0,0,1.0);CHKERRQ(ierr);
    ierr = AddModeLeftBdB(AA,d_nnz,o_nnz,0,0,1.0);CHKERRQ(ierr);
    ierr = AddModeRightBdB(AA,d_nnz,o_nnz,0,0,1.0);CHKERRQ(ierr);

    ierr = AddMLSSingleArrowNonconnecting(AA,d_nnz,o_nnz,0,n11,1.0); CHKERRQ(ierr);
    ierr = AddMLSSingleArrowNonconnecting(AA,d_nnz,o_nnz,0,n10,1.0); CHKERRQ(ierr);
    ierr = AddMLSSingleArrowNonconnecting(AA,d_nnz,o_nnz,0,n01,1.0); CHKERRQ(ierr);
    ierr = AddMLSSingleArrowConnecting(AA,d_nnz,o_nnz,0,n11,n00,1.0); CHKERRQ(ierr);

    ierr = MatMPIAIJSetPreallocation(AA,0,d_nnz,0,o_nnz); CHKERRQ(ierr);		//actual preassembly, tells Petsc approx. how many entries are in the matrix, this is a speedup step
    ierr = MatSeqAIJSetPreallocation(AA,0,d_nnz); CHKERRQ(ierr);

    ierr = AddModeLeftBRightBd(AA,d_nnz,o_nnz,1,0,kappa);CHKERRQ(ierr);
    ierr = AddModeLeftBdB(AA,d_nnz,o_nnz,1,0,-kappa/2.0);CHKERRQ(ierr);
    ierr = AddModeRightBdB(AA,d_nnz,o_nnz,1,0,-kappa/2.0);CHKERRQ(ierr);

    ierr = AddMLSSingleArrowNonconnecting(AA,d_nnz,o_nnz,1,n11,-gamma); CHKERRQ(ierr);
    ierr = AddMLSSingleArrowNonconnecting(AA,d_nnz,o_nnz,1,n10,-(delta+gamma)/2.0); CHKERRQ(ierr);
    ierr = AddMLSSingleArrowNonconnecting(AA,d_nnz,o_nnz,1,n01,-(delta+gamma)/2.0); CHKERRQ(ierr);
    ierr = AddMLSSingleArrowConnecting(AA,d_nnz,o_nnz,1,n11,n00,gamma); CHKERRQ(ierr);

    ierr = MatAssemblyBegin(AA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(AA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    //clean up
    delete[] d_nnz;
    delete[] o_nnz;

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TLS_J10left"

PetscErrorCode	Phononlaser::TLS_J10left(Mat AA,PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;

    MLSDim	n11 (1,1), n10 (1,0), n01 (0,1), n00 (0,0);

    ierr = AddMLSSingleArrowConnecting(AA,d_nnz,o_nnz,choose,n00,n01,matrixelem); CHKERRQ(ierr);
    ierr = AddMLSSingleArrowConnecting(AA,d_nnz,o_nnz,choose,n10,n11,matrixelem); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TLS_J01left"

/**
 * @brief	Creates the matrix corresponding to the unitary TAvis-Cummings interaction time evolution in the master equation.
 *
 * @param	AA	the Lindbladian.
 *
 */

PetscErrorCode	Phononlaser::TLS_J01left(Mat AA,PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;

    MLSDim	n11 (1,1), n10 (1,0), n01 (0,1), n00 (0,0);

    ierr = AddMLSSingleArrowConnecting(AA,d_nnz,o_nnz,choose,n01,n00,matrixelem); CHKERRQ(ierr);
    ierr = AddMLSSingleArrowConnecting(AA,d_nnz,o_nnz,choose,n11,n10,matrixelem); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TLS_J10right"

PetscErrorCode	Phononlaser::TLS_J10right(Mat AA,PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;

    MLSDim	n11 (1,1), n10 (1,0), n01 (0,1), n00 (0,0);

    ierr = AddMLSSingleArrowConnecting(AA,d_nnz,o_nnz,choose,n11,n01,matrixelem); CHKERRQ(ierr);
    ierr = AddMLSSingleArrowConnecting(AA,d_nnz,o_nnz,choose,n10,n00,matrixelem); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TLS_J01right"

/**
 * @brief	Creates the matrix corresponding to the unitary TAvis-Cummings interaction time evolution in the master equation.
 *
 * @param	AA	the Lindbladian.
 *
 */

PetscErrorCode	Phononlaser::TLS_J01right(Mat AA,PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;

    MLSDim	n11 (1,1), n10 (1,0), n01 (0,1), n00 (0,0);

    ierr = AddMLSSingleArrowConnecting(AA,d_nnz,o_nnz,choose,n01,n11,matrixelem); CHKERRQ(ierr);
    ierr = AddMLSSingleArrowConnecting(AA,d_nnz,o_nnz,choose,n00,n10,matrixelem); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TLS_J11left"

PetscErrorCode	Phononlaser::TLS_J11left(Mat AA,PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;

    MLSDim	n11 (1,1), n01 (0,1);

    ierr = AddMLSSingleArrowNonconnecting(AA,d_nnz,o_nnz,choose,n11,matrixelem); CHKERRQ(ierr);
    ierr = AddMLSSingleArrowNonconnecting(AA,d_nnz,o_nnz,choose,n01,matrixelem); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TLS_J11right"

PetscErrorCode	Phononlaser::TLS_J11right(Mat AA,PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;

    MLSDim	n11 (1,1), n10 (1,0);

    ierr = AddMLSSingleArrowNonconnecting(AA,d_nnz,o_nnz,choose,n11,matrixelem); CHKERRQ(ierr);
    ierr = AddMLSSingleArrowNonconnecting(AA,d_nnz,o_nnz,choose,n10,matrixelem); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  output
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

#undef __FUNCT__
#define __FUNCT__ "SetupMyOut"

PetscErrorCode MyOut::SetupMyOut(Phononlaser * system, std::string param)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;

    MyObsFile		*obsfile	= new MyObsFile;		//the file for the observables
    DistFile		*n11file	= new DistFile;			//the tls number state distribution file
    DistFile		*n11firstofile	= new DistFile;			//the first offdiagonal distribution file
    DistFile		*n11secofile	= new DistFile;			//second offdiagonal distribution file
    DistFile		*m0file		= new DistFile;			//phonon number state distribution file
    MyGnFile		*gnfile		= new MyGnFile;

    MLSDim	n11 (1,1), n01 (0,1);

    ierr = obsfile->SetupMyObsFile(system,"observables.dat",2.0,param); CHKERRQ(ierr);
    ierr = AddOFile(obsfile); CHKERRQ(ierr);

    ierr = n11file->SetupMLSDistFile(system,"n11.dat",n11,param); CHKERRQ(ierr);
    ierr = AddOFile(n11file); CHKERRQ(ierr);

    ierr = n11firstofile->SetupMLSOffdiagDistFile(system,"j01_1.dat",n01,1,param); CHKERRQ(ierr);
    ierr = AddOFile(n11firstofile); CHKERRQ(ierr);

    ierr = n11secofile->SetupMLSOffdiagDistFile(system,"j01_2.dat",n01,2,param); CHKERRQ(ierr);
    ierr = AddOFile(n11secofile); CHKERRQ(ierr);

    ierr = m0file->SetupModeDistFile(system,"m0.dat",0,param); CHKERRQ(ierr);
    ierr = AddOFile(m0file); CHKERRQ(ierr);

    ierr = gnfile->SetupMyGnFile(system,"correlation.dat",param); CHKERRQ(ierr);
    ierr = AddOFile(gnfile); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SetupMyObsFile"

PetscErrorCode MyObsFile::SetupMyObsFile(Phononlaser * system, std::string name, PetscReal rotfreq, std::string param)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;

    ierr = SetOFile(system,name); CHKERRQ(ierr);		//Open File, set file name
    ierr = WriteSystemParameters(system); CHKERRQ(ierr);	//write system parameters into the output file

    Observable	*ptrace		= new Observable();
    Observable	*pdens11	= new Observable();
    Observable	*ppol10		= new Observable();
    Observable	*p2pol10	= new Observable();
    Observable	*pdens11full	= new Observable();
    Observable	*pdens00full	= new Observable();
    Observable	*pinter01	= new Observable();
    Observable	*pjzsq11_00	= new Observable();
    Observable	*pjz11_00	= new Observable();
    Observable	*ptotalspin	= new Observable();
    Observable	*pmodeocc	= new Observable();
    Observable	*pmodepol	= new Observable();

    MLSDim	n11 (1,1), n10 (1,0), n01 (0,1), n00 (0,0);

    PetscReal	domega_tls;
    ierr = system->GetParam("domega_tls",&domega_tls);CHKERRQ(ierr);

    ierr = ptrace->SetupTrMinus1(system); CHKERRQ(ierr);					//good quantitiy for convergence monitoring
    ierr = AddElem(ptrace,"tr[]-1\t"); CHKERRQ(ierr);						//the "..." is the name that will be printed into the file header, you can adjust hat to your personal taste

    ierr = pdens11->SetupMlsOccupation(system,n11); CHKERRQ(ierr);				//this computes <J_11> for hermitian operators the program just prints the real part and checks whether the imaginary part is below a certain threshold,
    ierr = AddElem(pdens11,"<J_11>\t"); CHKERRQ(ierr);						//which is defined in the constants.hpp file

    ierr = ppol10->SetupMlsPolarization(system,n10,2.0-domega_tls); CHKERRQ(ierr);		//nonhermitian operators are plotted with real and imaginary part by default
    ierr = AddElem(ppol10,"Re<J_10>\t\tIm<J_10>"); CHKERRQ(ierr);				//therefore we put two names into the header

    ierr = p2pol10->SetupMlsHigherPolarization(system,n10,2,2.0-domega_tls); CHKERRQ(ierr);	//this computes <J_10^2>
    ierr = AddElem(p2pol10,"Re<J_10^2>\t\tIm<J_10^2>"); CHKERRQ(ierr);				//which is also nonhermitian

    ierr = pdens11full->SetupMLSOccupationFull(system,n01); CHKERRQ(ierr);			//this computes <J_10 J_01> which is unequal to <J_11>
    ierr = AddElem(pdens11full,"<J_+ J_->"); CHKERRQ(ierr);

    ierr = pdens00full->SetupMLSOccupationFull(system,n10); CHKERRQ(ierr);			//this computes <J_01 J_10> which is unequal to <J_00>
    ierr = AddElem(pdens00full,"<J_- J_+>"); CHKERRQ(ierr);

    ierr = pinter01->SetupMLSIntercoupling(system,n01); CHKERRQ(ierr);				//that is the actual difference i.e. <J_10 J_01 - J_11>
    ierr = AddElem(pinter01,"<J_+J_--J_11>"); CHKERRQ(ierr);

    ierr = pjzsq11_00->SetupMlsJzSquaredNorm(system,n11,n00); CHKERRQ(ierr);			//that is <(J_11 - J_00)^2>
    ierr = AddElem(pjzsq11_00,"<(J^z)^2>"); CHKERRQ(ierr);

    ierr = pjz11_00->SetupMlsJzNorm(system,n11,n00); CHKERRQ(ierr);				//that is <(J_11 - J_00)>
    ierr = AddElem(pjz11_00,"<J^z>"); CHKERRQ(ierr);

    ierr = ptotalspin->SetupTotalSpin(system,n11,n00); CHKERRQ(ierr);				//total pseudo spin expectation value
    ierr = AddElem(ptotalspin,"<J^2>\t"); CHKERRQ(ierr);

    ierr = pmodeocc->SetupModeOccupation(system,0); CHKERRQ(ierr);				// <bd b>
    ierr = AddElem(pmodeocc,"<bdb>\t"); CHKERRQ(ierr);

    ierr = pmodepol->SetupModePolarization(system,0,0.0); CHKERRQ(ierr);			// <b>
    ierr = AddElem(pmodepol,"Re<b>\t\tIm<b>"); CHKERRQ(ierr);

    ierr = MakeHeaderGen(param);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SetupMyGnFile"

PetscErrorCode MyGnFile::SetupMyGnFile(Phononlaser * sys,std::string name, std::string param)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;

    ierr = SetOFile(sys,name); CHKERRQ(ierr);					//Open File, set file name
    ierr = WriteSystemParameters(sys); CHKERRQ(ierr);				//write system parameters into the header of the output file

    Gnfct	*mode0secorder		= new Gnfct();
    Gnfct	*mode0thirdorder	= new Gnfct();
    Gnfct	*n11secorder		= new Gnfct();
    Gnfct	*n11thirdorder		= new Gnfct();

    MLSDim	n01 (0,1);

    ierr = mode0secorder->SetupModeGnfct(sys,0,2); CHKERRQ(ierr);		//this is the g^(2) function of the phonon mode
    ierr = AddElem(mode0secorder,"g(2)(m0)"); CHKERRQ(ierr);

    ierr = mode0thirdorder->SetupModeGnfct(sys,0,3); CHKERRQ(ierr);		//this is the g^(3) function of the phonon mode
    ierr = AddElem(mode0thirdorder,"g(3)(m0)"); CHKERRQ(ierr);

    ierr = n11secorder->SetupMLSGnfct(sys,n01,2); CHKERRQ(ierr);		//this is the g^(2) function of the excited state distribution
    ierr = AddElem(n11secorder,"g(2)(n11)"); CHKERRQ(ierr);

    ierr = n11thirdorder->SetupMLSGnfct(sys,n01,3); CHKERRQ(ierr);		//this is the g^(3) function of the excited state distribution
    ierr = AddElem(n11thirdorder,"g(3)(n11)"); CHKERRQ(ierr);

    ierr = MakeHeaderGen(param);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  main function
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

static char help[] = "Phononlaser time evolution.\n\n";

#undef __FUNCT__
#define __FUNCT__ "main"

/**
 * @brief	User provided main function.
 *
 */

int main(int argc, char **args)
{
    PetscInitialize(&argc,&args,(char*)0,help);

    PetscErrorCode	ierr;
    Vec			dm;			//density matrix
    Mat			AA;			//Liouvillian


    //psiquasp setup
    Phononlaser	phaser;								//system specification
    ierr	= phaser.Setup(&dm,&AA); CHKERRQ(ierr);

    MyOut	*out = new MyOut;						//output specification
    ierr	= out->SetupMyOut(&phaser,"time"); CHKERRQ(ierr);


    //computing stage: use Petsc to solve the problem using explicit Runge-Kutta time integration
    TS		ts;							//time stepper
    TSAdapt	adapt;							//adaptive time step context

    //time step solver
    ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);				//create time stepper context
    ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);				//tell petsc that we solve a linear diff. eq.
    ierr = TSSetType(ts,TSRK);CHKERRQ(ierr);						//set the time stepper to runge kutta
    ierr = TSRKSetType(ts,TSRK3BS);CHKERRQ(ierr);					//set it to 3rd order RK scheme of Bogacki-Shampine with 2nd order embedded method, this is an adaptive step width Runge-Kutta
    ierr = TSSetMaxTime(ts,1.e+6);CHKERRQ(ierr);                                     //set the maximum integration time
    ierr = TSSetMaxSteps(ts,100000);CHKERRQ(ierr);                                   //set the maximum integration steps
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


  //-------------------------------------------------------------------------------------
  //clean up stage: free everything that is not needed anymore
  //-------------------------------------------------------------------------------------
    ierr = MatDestroy(&AA); CHKERRQ(ierr);
    ierr = VecDestroy(&dm); CHKERRQ(ierr);
    ierr = TSDestroy(&ts); CHKERRQ(ierr);

    delete out;												//need to write it explicitely, before PetscFinalize because destructors need MPI for PetscFClose()

    ierr = PetscFinalize(); CHKERRQ(ierr);
    return 0;
}
