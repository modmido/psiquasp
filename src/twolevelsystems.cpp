
/**
 * @file	twolevelsystems.cpp
 *
 * 		Function defintions for the derived TLS class.
 *
 * @author	Michael Gegg
 *
 */

#include"../include/twolevelsystems.hpp"




//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  TLS class members
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

#undef __FUNCT__
#define __FUNCT__ "TLSAdd"

/**
 * @brief	Adds all two-level system dimensions. If modes are needed they have to be set after calling this function. The user has to call PQSPSetup() before setting any Liouvillians, etc.
 *
 * @param	ntls		the number of two-level systems
 * @param	n10cutoff	the cutoff value for the n10 offidagonals
 * @param	n01cutoff	the cutoff value for the n01 offdiagonals, usually these two parameters should be the same since they have the same physical meaning
 * @param	energy		the energy gap between the ground and exited state
 *
 */

PetscErrorCode	TLS::TLSAdd(PetscInt ntls, PetscInt n10cutoff, PetscInt n01cutoff, PetscReal energy)
{
    PetscFunctionBeginUser;

    PetscErrorCode	ierr;

    N_MLS[0] = ntls;

    MLSDim	n11 (1,1), n10 (1,0), n01 (0,1);

    ierr  = MLSAddDens(n11,ntls+1,energy); CHKERRQ(ierr);
    ierr  = MLSAddPol(n10,n10cutoff+1); CHKERRQ(ierr);
    ierr  = MLSAddPol(n01,n01cutoff+1); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddTLSH0"

/**
 * @brief	Adds the TLS H0 master equation contribution to a matrix, i.e. \f$ \mathcal{L} \rho \leftrightarrow matrixelem*[\rho ,J_{11}] \f$
 *
 * @param	AA		the Liouvillan matrix
 * @param	d_nnz		counts the diagonal elements per row
 * @param	o_nnz		counts the offdiagonal elements per row
 * @param	choose		0 for preallocation, i.e. counting of diagonal and offdiagonal elements, 1 for filling the matrix with the actual values
 * @param	matrixelem	the prefactor arising in H0
 *
 */

PetscErrorCode TLS::AddTLSH0(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;

    PetscErrorCode	ierr;
    MLSDim		n10 (1,0);

    ierr = AddMLSH0(AA,d_nnz,o_nnz,choose,n10,matrixelem); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddTavisCummingsHamiltonianRWA"

/**
 * @brief	Adds the Tavis-Cummings Hamiltonian master equation contribution to the matrix, i.e. \f$ \mathcal{L} \rho \leftrightarrow i/\hbar [\rho ,H_{TC}] \f$ with \f$ H_{TC} = \hbar g( J_{01} b^\dagger + J_{10} b )  \f$
 *
 * @param	AA		the Liouvillan matrix
 * @param	d_nnz		counts the diagonal elements per row
 * @param	o_nnz		counts the offdiagonal elements per row
 * @param	choose		0 for preallocation, i.e. counting of diagonal and offdiagonal elements, 1 for filling the matrix with the actual values
 * @param	photonnumber	the index of the mode that couples to the two-level systems
 * @param	matrixelem	the prefactor arising in the H_{TC} contribution, i.e. \f$ i*g \f$
 *
 */

PetscErrorCode	TLS::AddTavisCummingsHamiltonianRWA(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt photonnumber, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;

    PetscErrorCode	ierr;
    MLSDim		n11 (1,1);
    MLSDim		n10 (1,0);
    MLSDim		n01 (0,1);
    MLSDim		n00 (0,0);
    ModeDim		mket (0,photonnumber);
    ModeDim		mbra (1,photonnumber);

    ierr = AddMLSModeInt(AA,d_nnz,o_nnz,choose,n10,n11,mbra,-matrixelem); CHKERRQ(ierr);		//Tavis-Cummings Hamiltonian
    ierr = AddMLSModeInt(AA,d_nnz,o_nnz,choose,n01,n11,mket,matrixelem); CHKERRQ(ierr);
    ierr = AddMLSModeInt(AA,d_nnz,o_nnz,choose,n00,n10,mket,matrixelem); CHKERRQ(ierr);
    ierr = AddMLSModeInt(AA,d_nnz,o_nnz,choose,n00,n01,mbra,-matrixelem); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddTavisCummingsHamiltonianNoRWA"

/**
 * @brief	Adds the Tavis-Cummings Hamiltonian master equation contribution to the matrix, without RWA, i.e. \f$ \mathcal{L} \rho \leftrightarrow i/\hbar [\rho ,H_{TC,full}] \f$ with \f$ H_{TC,full} = \hbar g( J_{01}  + J_{10}  )( b^\dagger + b)  \f$
 *
 * @param	AA		the Liouvillan matrix
 * @param	d_nnz		counts the diagonal elements per row
 * @param	o_nnz		counts the offdiagonal elements per row
 * @param	choose		0 for preallocation, i.e. counting of diagonal and offdiagonal elements, 1 for filling the matrix with the actual values
 * @param	photonnumber	the index of the mode that couples to the two-level systems
 * @param	matrixelem	the prefactor arising in the H_{TC,full} contribution, i.e. \f$ i*g \f$
 *
 */

PetscErrorCode TLS::AddTavisCummingsHamiltonianNoRWA(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt photonnumber, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;

    PetscErrorCode	ierr;
    MLSDim		n11 (1,1);
    MLSDim		n10 (1,0);
    MLSDim		n01 (0,1);
    MLSDim		n00 (0,0);
    ModeDim		mket (0,photonnumber);
    ModeDim		mbra (1,photonnumber);

    ierr = AddMLSModeInt(AA,d_nnz,o_nnz,choose,n10,n11,mbra,-matrixelem);CHKERRQ(ierr);
    ierr = AddMLSModeInt(AA,d_nnz,o_nnz,choose,n11,n10,mbra,-matrixelem);CHKERRQ(ierr);

    ierr = AddMLSModeInt(AA,d_nnz,o_nnz,choose,n01,n11,mket,matrixelem);CHKERRQ(ierr);
    ierr = AddMLSModeInt(AA,d_nnz,o_nnz,choose,n11,n01,mket,matrixelem);CHKERRQ(ierr);

    ierr = AddMLSModeInt(AA,d_nnz,o_nnz,choose,n00,n10,mket,matrixelem);CHKERRQ(ierr);
    ierr = AddMLSModeInt(AA,d_nnz,o_nnz,choose,n10,n00,mket,matrixelem);CHKERRQ(ierr);

    ierr = AddMLSModeInt(AA,d_nnz,o_nnz,choose,n00,n01,mbra,-matrixelem);CHKERRQ(ierr);
    ierr = AddMLSModeInt(AA,d_nnz,o_nnz,choose,n01,n00,mbra,-matrixelem);CHKERRQ(ierr);


    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddTLSCoherentDrive"

/**
 * @brief	Adds the semiclassical collective, coherent driving Hamiltonian master equation contribution to the matrix, i.e. \f$ \mathcal{L} \rho \leftrightarrow i/\hbar [\rho ,H_{d}] \f$ with \f$ H_{d} = \hbar E ( J_{01}  + J_{10}  )\f$
 * 		Since this matrix is time independent, the user needs to define an appropriate rotating frame for this to be valid.
 *
 * @param	AA		the Liouvillan matrix
 * @param	d_nnz		counts the diagonal elements per row
 * @param	o_nnz		counts the offdiagonal elements per row
 * @param	choose		0 for preallocation, i.e. counting of diagonal and offdiagonal elements, 1 for filling the matrix with the actual values
 * @param	matrixelem	the prefactor arising in the H_{d} contribution, i.e. \f$ i*E \f$
 *
 */

PetscErrorCode TLS::AddTLSCoherentDrive(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;

    PetscErrorCode	ierr;
    MLSDim		n11 (1,1);
    MLSDim		n10 (1,0);
    MLSDim		n01 (0,1);
    MLSDim		n00 (0,0);

    ierr = AddMLSCohDrive(AA,d_nnz,o_nnz,choose,n10,n11,-matrixelem); CHKERRQ(ierr);
    ierr = AddMLSCohDrive(AA,d_nnz,o_nnz,choose,n01,n11,matrixelem); CHKERRQ(ierr);
    ierr = AddMLSCohDrive(AA,d_nnz,o_nnz,choose,n00,n10,matrixelem); CHKERRQ(ierr);
    ierr = AddMLSCohDrive(AA,d_nnz,o_nnz,choose,n00,n01,-matrixelem); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddTLSSpontaneousEmission"

/**
 * @brief	Adds the individual spontaneous emission master equation contribution to the matrix, i.e. \f$ \mathcal{L} \rho \leftrightarrow matrixelem \sum_i (2 \sigma_{01}^i \rho \sigma_{10}^i - \sigma_{11}^i\rho - \rho \sigma_{11}^i) \f$
 *
 * @param	AA		the Liouvillan matrix
 * @param	d_nnz		counts the diagonal elements per row
 * @param	o_nnz		counts the offdiagonal elements per row
 * @param	choose		0 for preallocation, i.e. counting of diagonal and offdiagonal elements, 1 for filling the matrix with the actual values
 * @param	matrixelem	the prefactor
 *
 */

PetscErrorCode TLS::AddTLSSpontaneousEmission(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscReal matrixelem)
{
    PetscFunctionBeginUser;

    PetscErrorCode	ierr;
    MLSDim		n11 (1,1);
    MLSDim		n10 (1,0);
    MLSDim		n01 (0,1);
    MLSDim		n00 (0,0);

    ierr = AddLindbladRelaxMLS(AA,d_nnz,o_nnz,choose,n11,n00,matrixelem); CHKERRQ(ierr);		//Individual spontaneous emission of the TLS into vacuum
    ierr = AddLindbladDephMLS(AA,d_nnz,o_nnz,choose,n10,matrixelem); CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(AA,d_nnz,o_nnz,choose,n01,matrixelem); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddTLSPureDephasing"

/**
 * @brief	Adds the pure dephasing master equation contribution to the matrix, i.e. \f$ \mathcal{L} \rho \leftrightarrow matrixelem \sum_i (2 \sigma_{10}^i \rho \sigma_{01}^i - \sigma_{00}^i\rho - \rho \sigma_{00}^i) \f$
 *
 * @param	AA		the Liouvillan matrix
 * @param	d_nnz		counts the diagonal elements per row
 * @param	o_nnz		counts the offdiagonal elements per row
 * @param	choose		0 for preallocation, i.e. counting of diagonal and offdiagonal elements, 1 for filling the matrix with the actual values
 * @param	matrixelem	the prefactor
 *
 */

PetscErrorCode TLS::AddTLSPureDephasing(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscReal matrixelem)
{
    PetscFunctionBeginUser;

    PetscErrorCode	ierr;
    MLSDim		n11 (1,1);
    MLSDim		n10 (1,0);
    MLSDim		n01 (0,1);
    MLSDim		n00 (0,0);

    ierr = AddLindbladDephMLS(AA,d_nnz,o_nnz,choose,n10,2*matrixelem); CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(AA,d_nnz,o_nnz,choose,n01,2*matrixelem); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AddTLSIncoherentPump"

/**
 * @brief	Adds the incoherent pumping master equation contribution to the matrix, i.e. \f$ \mathcal{L} \rho \leftrightarrow matrixelem \sum_i (2 \sigma_{z}^i \rho \sigma_{z}^i - (\sigma_{z}^i)^2\rho - \rho (\sigma_{z}^i)^2) = 2*matrixelem \sum_i ( \sigma_{z}^i \rho \sigma_{z}^i - \rho ) \f$
 *
 * @param	AA		the Liouvillan matrix
 * @param	d_nnz		counts the diagonal elements per row
 * @param	o_nnz		counts the offdiagonal elements per row
 * @param	choose		0 for preallocation, i.e. counting of diagonal and offdiagonal elements, 1 for filling the matrix with the actual values
 * @param	matrixelem	the prefactor
 *
 */

PetscErrorCode TLS::AddTLSIncoherentPump(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscReal matrixelem)
{
    PetscFunctionBeginUser;

    PetscErrorCode	ierr;
    MLSDim		n11 (1,1);
    MLSDim		n10 (1,0);
    MLSDim		n01 (0,1);
    MLSDim		n00 (0,0);

    ierr = AddLindbladRelaxMLS(AA,d_nnz,o_nnz,choose,n00,n11,matrixelem); CHKERRQ(ierr);		//Individual incoherent pumping
    ierr = AddLindbladDephMLS(AA,d_nnz,o_nnz,choose,n10,matrixelem); CHKERRQ(ierr);
    ierr = AddLindbladDephMLS(AA,d_nnz,o_nnz,choose,n01,matrixelem); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatTLSJ10Left"

/**
 * @brief	Creates a Liouvillan matrix that corresponds to the two-level system self energy Hamiltonian, i.e. \f$ \mathcal{L} \rho \leftrightarrow matrixelem*J_{11} \rho \f$
 *
 * @param	AA		the Liouvillan matrix
 * @param	matrixelem	the prefactor
 *
 */

PetscErrorCode TLS::MatTLSJ10Left(Mat* AA, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;
    PetscErrorCode	ierr;

    PetscInt		*d_nnz	= new PetscInt [loc_size] ();
    PetscInt		*o_nnz	= new PetscInt [loc_size] ();
    MLSDim		n11 (1,1);
    MLSDim		n10 (1,0);
    MLSDim		n01 (0,1);
    MLSDim		n00 (0,0);


    //create matrix
    ierr = PQSPCreateMat(AA); CHKERRQ(ierr);


    //preallocation
    ierr = AddMLSSingleArrowConnecting(*AA,d_nnz,o_nnz,0,n10,n00,1.0); CHKERRQ(ierr);		//Jl_{10}
    ierr = AddMLSSingleArrowConnecting(*AA,d_nnz,o_nnz,0,n11,n01,1.0); CHKERRQ(ierr);		//...

    ierr = MatMPIAIJSetPreallocation(*AA,0,d_nnz,0,o_nnz); CHKERRQ(ierr);			//parallel
    ierr = MatSeqAIJSetPreallocation(*AA,0,d_nnz); CHKERRQ(ierr);				//and sequential


    //allocation
    ierr = AddMLSSingleArrowConnecting(*AA,d_nnz,o_nnz,1,n10,n00,1.0); CHKERRQ(ierr);		//Jl_{10}
    ierr = AddMLSSingleArrowConnecting(*AA,d_nnz,o_nnz,1,n11,n01,1.0); CHKERRQ(ierr);		//...

    ierr = MatAssemblyBegin(*AA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);				//Assemble the matrix
    ierr = MatAssemblyEnd(*AA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);				//


    //clean up
    delete[] d_nnz;
    delete[] o_nnz;


    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatTLSJ11Left"

/**
 * @brief	Creates a Liouvillan matrix that corresponds to the two-level system self energy Hamiltonian, i.e. \f$ \mathcal{L} \rho \leftrightarrow matrixelem*J_{11} \rho \f$
 *
 * @param	AA		the Liouvillan matrix
 * @param	matrixelem	the prefactor
 *
 */

PetscErrorCode TLS::MatTLSJ11Left(Mat *AA, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;

    PetscErrorCode	ierr;

    ierr = PQSPCreateMat(AA); CHKERRQ(ierr);

    PetscInt	*d_nnz	= new PetscInt [loc_size] ();
    PetscInt	*o_nnz	= new PetscInt [loc_size] ();
    MLSDim	n11 (1,1);
    MLSDim	n01 (0,1);

    ierr = AddMLSSingleArrowNonconnecting(*AA,d_nnz,o_nnz,0,n11,1.0); CHKERRQ(ierr);
    ierr = AddMLSSingleArrowNonconnecting(*AA,d_nnz,o_nnz,0,n01,1.0); CHKERRQ(ierr);

    ierr = MatMPIAIJSetPreallocation(*AA,0,d_nnz,0,o_nnz); CHKERRQ(ierr);		//preassembly, makes it faster if many elements are added
    ierr = MatSeqAIJSetPreallocation(*AA,0,d_nnz); CHKERRQ(ierr);

    ierr = AddMLSSingleArrowNonconnecting(*AA,d_nnz,o_nnz,1,n11,matrixelem); CHKERRQ(ierr);
    ierr = AddMLSSingleArrowNonconnecting(*AA,d_nnz,o_nnz,1,n01,matrixelem); CHKERRQ(ierr);

    ierr = MatAssemblyBegin(*AA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);			//actual assembly
    ierr = MatAssemblyEnd(*AA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    delete[] d_nnz;
    delete[] o_nnz;

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatTLSTCHamiltonianRWALeft"

/**
 * @brief	Creates a Liouvillan matrix that corresponds to the Tavis-Cummings interaction Hamiltonian, i.e. \f$ \mathcal{L} \rho \leftrightarrow matrixelem*(J_{10}b + J_{01}b^\dagger) \rho \f$
 *
 * @param	AA		the Liouvillan matrix
 * @param	photonnumber	the index of the mode that couples to the two-level systems
 * @param	matrixelem	the prefactor
 *
 */

PetscErrorCode TLS::MatTLSTCHamiltonianRWALeft(Mat *AA, PetscInt photonnumber, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;

    PetscErrorCode	ierr;

    ierr = PQSPCreateMat(AA); CHKERRQ(ierr);

    PetscInt		*d_nnz	= new PetscInt [loc_size] ();
    PetscInt		*o_nnz	= new PetscInt [loc_size] ();
    MLSDim		n11 (1,1);
    MLSDim		n10 (1,0);
    MLSDim		n01 (0,1);
    MLSDim		n00 (0,0);
    ModeDim		mbra (1,photonnumber);

    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,0,n10,n11,mbra,1.0); CHKERRQ(ierr);				//Tavis-Cummings Hamiltonian
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,0,n00,n01,mbra,1.0); CHKERRQ(ierr);

    ierr = MatMPIAIJSetPreallocation(*AA,0,d_nnz,0,o_nnz); CHKERRQ(ierr);		//preassembly, makes it faster if many elements are added
    ierr = MatSeqAIJSetPreallocation(*AA,0,d_nnz); CHKERRQ(ierr);

    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,1,n10,n11,mbra,matrixelem); CHKERRQ(ierr);			//Tavis-Cummings Hamiltonian
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,1,n00,n01,mbra,matrixelem); CHKERRQ(ierr);

    ierr = MatAssemblyBegin(*AA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);			//actual assembly
    ierr = MatAssemblyEnd(*AA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    delete[] d_nnz;
    delete[] o_nnz;

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatTLSTCHamiltonianNoRWALeft"

/**
 * @brief	Creates a Liouvillan matrix that corresponds to the Tavis-Cummings interaction Hamiltonian without RWA, i.e. \f$ \mathcal{L} \rho \leftrightarrow matrixelem*(J_{10} + J_{01})(b+b^\dagger ) \rho \f$
 *
 * @param	AA		the Liouvillan matrix
 * @param	photonnumber	the index of the mode that couples to the two-level systems
 * @param	matrixelem	the prefactor
 *
 */

PetscErrorCode TLS::MatTLSTCHamiltonianNoRWALeft(Mat *AA, PetscInt photonnumber, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;

    PetscErrorCode	ierr;

    ierr = PQSPCreateMat(AA); CHKERRQ(ierr);

    PetscInt		*d_nnz	= new PetscInt [loc_size] ();
    PetscInt		*o_nnz	= new PetscInt [loc_size] ();
    MLSDim		n11 (1,1);
    MLSDim		n10 (1,0);
    MLSDim		n01 (0,1);
    MLSDim		n00 (0,0);
    ModeDim		mbra (1,photonnumber);

    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,0,n10,n11,mbra,1.0); CHKERRQ(ierr);
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,0,n11,n10,mbra,1.0); CHKERRQ(ierr);
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,0,n00,n01,mbra,1.0); CHKERRQ(ierr);
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,0,n01,n00,mbra,1.0); CHKERRQ(ierr);

    ierr = MatMPIAIJSetPreallocation(*AA,0,d_nnz,0,o_nnz); CHKERRQ(ierr);		//preassembly, makes it faster if many elements are added
    ierr = MatSeqAIJSetPreallocation(*AA,0,d_nnz); CHKERRQ(ierr);

    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,1,n10,n11,mbra,matrixelem); CHKERRQ(ierr);
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,1,n11,n10,mbra,matrixelem); CHKERRQ(ierr);
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,1,n00,n01,mbra,matrixelem); CHKERRQ(ierr);
    ierr = AddMLSModeInt(*AA,d_nnz,o_nnz,1,n01,n00,mbra,matrixelem); CHKERRQ(ierr);

    ierr = MatAssemblyBegin(*AA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);			//actual assembly
    ierr = MatAssemblyEnd(*AA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    delete[] d_nnz;
    delete[] o_nnz;

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatTLSCoherentDriveLeft"

/**
 * @brief	Creates a Liouvillan matrix that corresponds to the Tavis-Cummings interaction Hamiltonian, i.e. \f$ \mathcal{L} \rho \leftrightarrow matrixelem*(J_{10} + J_{01}) \rho \f$
 *
 * @param	AA		the Liouvillan matrix
 * @param	matrixelem	the prefactor
 *
 */

PetscErrorCode TLS::MatTLSCoherentDriveLeft(Mat *AA, PetscScalar matrixelem)
{
    PetscFunctionBeginUser;

    PetscErrorCode	ierr;

    ierr = PQSPCreateMat(AA); CHKERRQ(ierr);

    PetscInt	*d_nnz	= new PetscInt [loc_size] ();
    PetscInt	*o_nnz	= new PetscInt [loc_size] ();
    MLSDim	n11 (1,1);
    MLSDim	n10 (1,0);
    MLSDim	n01 (0,1);
    MLSDim	n00 (0,0);

    ierr = AddMLSCohDrive(*AA,d_nnz,o_nnz,0,n10,n11,1.0); CHKERRQ(ierr);
    ierr = AddMLSCohDrive(*AA,d_nnz,o_nnz,0,n00,n01,1.0); CHKERRQ(ierr);

    ierr = MatMPIAIJSetPreallocation(*AA,0,d_nnz,0,o_nnz); CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(*AA,0,d_nnz); CHKERRQ(ierr);

    ierr = AddMLSCohDrive(*AA,d_nnz,o_nnz,1,n10,n11,matrixelem); CHKERRQ(ierr);
    ierr = AddMLSCohDrive(*AA,d_nnz,o_nnz,1,n00,n01,matrixelem); CHKERRQ(ierr);

    ierr = MatAssemblyBegin(*AA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);			//Assemble the matrix
    ierr = MatAssemblyEnd(*AA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);			//

    ierr = MatSetOption(*AA,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE); CHKERRQ(ierr);

    delete[] d_nnz;
    delete[] o_nnz;

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatMLSLeftRightCollectiveRaisingOperator"

/**
 * @brief	Creates a matrix corresponding to applying a tls raising operator on both sides of a Liouville space projector, needed for the successive creation of Dicke state projectors. <br>
 * 		Please note the difference between raising the Liovuville space projector and applying raising lowering operators to the density matrix: \f$ J_{10} \hat{\mathcal{P}}[n11,n10,n01] J_{01} \leftrightarrow J_{01} \rho J_{10} \f$.
 *
 * @param	AA	the Liouvillan matrix.
 *
 */

PetscErrorCode TLS::MatMLSLeftRightCollectiveRaisingOperator(Mat * AA)
{
    PetscFunctionBeginUser;

    PetscErrorCode	ierr;


    //two matrices: Jleft_{10} and Jright_{01}
    Mat			left,right;							//the matrix corresponding to Jl_{10} Jr_{01}
    PetscInt		*d_nnz	= new PetscInt [loc_size] ();
    PetscInt		*o_nnz	= new PetscInt [loc_size] ();
    MLSDim		n11 (1,1);
    MLSDim		n10 (1,0);
    MLSDim		n01 (0,1);
    MLSDim		n00 (0,0);

    ierr = PQSPCreateMat(&left); CHKERRQ(ierr);
    ierr = PQSPCreateMat(&right); CHKERRQ(ierr);


    //preallocation
    ierr = AddMLSSingleArrowConnecting(left,d_nnz,o_nnz,0,n10,n00,1.0); CHKERRQ(ierr);	//Jl_{10}
    ierr = AddMLSSingleArrowConnecting(left,d_nnz,o_nnz,0,n11,n01,1.0); CHKERRQ(ierr);	//...

    ierr = AddMLSSingleArrowConnecting(right,d_nnz,o_nnz,0,n01,n00,1.0); CHKERRQ(ierr);	//Jr_{01}
    ierr = AddMLSSingleArrowConnecting(right,d_nnz,o_nnz,0,n11,n10,1.0); CHKERRQ(ierr);	//...

    ierr = MatMPIAIJSetPreallocation(left,0,d_nnz,0,o_nnz); CHKERRQ(ierr);			//parallel
    ierr = MatSeqAIJSetPreallocation(left,0,d_nnz); CHKERRQ(ierr);				//and sequential

    ierr = MatMPIAIJSetPreallocation(right,0,d_nnz,0,o_nnz); CHKERRQ(ierr);			//parallel
    ierr = MatSeqAIJSetPreallocation(right,0,d_nnz); CHKERRQ(ierr);				//and sequential


    //allocation
    ierr = AddMLSSingleArrowConnecting(left,d_nnz,o_nnz,1,n10,n00,1.0); CHKERRQ(ierr);	//Jl_{10}
    ierr = AddMLSSingleArrowConnecting(left,d_nnz,o_nnz,1,n11,n01,1.0); CHKERRQ(ierr);	//...

    ierr = AddMLSSingleArrowConnecting(right,d_nnz,o_nnz,1,n01,n00,1.0); CHKERRQ(ierr);	//Jr_{01}
    ierr = AddMLSSingleArrowConnecting(right,d_nnz,o_nnz,1,n11,n10,1.0); CHKERRQ(ierr);	//...

    ierr = MatAssemblyBegin(left,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);				//Assemble the matrix
    ierr = MatAssemblyEnd(left,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);				//

    ierr = MatAssemblyBegin(right,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);				//Assemble the matrix
    ierr = MatAssemblyEnd(right,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);				//


    //multiply the two matrices to get the two sided operator
    ierr = MatMatMult(left,right,MAT_INITIAL_MATRIX,PETSC_DEFAULT,AA); CHKERRQ(ierr);


    //clean up
    ierr = MatDestroy(&left); CHKERRQ(ierr);
    ierr = MatDestroy(&right); CHKERRQ(ierr);
    delete[] d_nnz;
    delete[] o_nnz;


    PetscFunctionReturn(0);
}


//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  DickeDistribution class members
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

#undef __FUNCT__
#define __FUNCT__ "SetupDickeDist"

/**
 * @brief	Setup function for the Dicke distribution.
 *
 * @param	sys	the System object.
 */

PetscErrorCode	DickeDistribution::SetupDickeDist(TLS * sys)
{
    PetscFunctionBeginUser;

    PetscErrorCode	ierr;


    //allocate first level storage, just needs the number of states
    PetscInt		add,ndickestates = 0;
    add			= sys->NMls() +1;		//the superradiant subspace has dimension N+1

    while( add > 0 )
    {
      ndickestates += add;				//add the current subspace dimension
      add -= 2;						//the dimension of the next subspace is two less
    }

    ierr = AllocateLocStorage(ndickestates); CHKERRQ(ierr);	//allocate



    //set Jl_{10} Jr_{01} matrix
    Mat			diagstepup;

    ierr = sys->MatMLSLeftRightCollectiveRaisingOperator(&diagstepup); CHKERRQ(ierr);



    //create a tls ground state projecton vector and a dummy vector for storage
    Vec			elem,elem2;

    ierr = sys->PQSPCreateVec(&elem,NULL,NULL); CHKERRQ(ierr);			//storage vectors
    ierr = sys->PQSPCreateVec(&elem2,NULL,NULL); CHKERRQ(ierr);			//

    ierr = sys->VecMLSGroundStateModeTraceout(elem);				//mls ground state projector, includes a trace over the mode dofs



    //initialize the Dicke quantum numbers
    PetscReal		m,l;							//the quantum numbers m and l
    l			= sys->NMls()/2.0;					//initialize quantum numbers for
    m			= -l;							//ground state projector |N/2,-N/2><N/2,-N/2|



    //create all projection vectors, contract them and store them in the dmindex and prefactors arrays
    PetscInt		step,j,count;
    step 		= 0;								//the integer counting the l subspaces, starts with zero in the superrradiant subspace
    count		= 0;								//the index counting all (superradiant and all other) dicke state projectors

    while( step < sys->NMls()/2 +1 )							//works with integer division, the total number of Dicke subspaces
    {
      for(j = 0; j < sys->NMls()+1-2*step; j++)						//NMLS+1 possibilities for l_max, for each successive subspace l->l-1 the dimension drops by two
      {
	ierr	= sys->VecContractReal(elem,&lengths[count],&dmindex[count],&prefactors[count]); CHKERRQ(ierr);		//extract the local nonzeros out of the vector and store them into arrays plus allocate them -> less overhead in computation

	ierr	= MatMultTranspose(diagstepup,elem,elem2); CHKERRQ(ierr);		//create the unnormalized higher lying dicke projector |l,m +1><l,m +1|
	ierr	= VecCopy(elem2,elem); CHKERRQ(ierr);					//copy it back into elem

	ierr	= VecScale(elem,1.0/((l-m)*(l+m+1))); CHKERRQ(ierr);			//normalization
	m++;										//increase inversion m by one

	count++;									//increase the running index by one
      }

      step++;		//new l subspace
      l--;		//set quantum numbers to ground state of the next śpin subspace
      m = -l;		//   -->   |l-1,-(l-1)><l-1,-(l-1)|

      if( step != sys->NMls()/2+1 )	//if its not the last subspace already
      {
	ierr	= JBlockShift(sys,step,elem); CHKERRQ(ierr);	//find the ground state of the next subspace via the trace condition
      }
    }

    ierr = MatDestroy(&diagstepup); CHKERRQ(ierr);
    ierr = VecDestroy(&elem); CHKERRQ(ierr);
    ierr = VecDestroy(&elem2); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "JBlockShift"

/**
 * @brief	Step into the lower Dicke l subspace using the trace condition.
 *
 * 		The trace condition is as follows:
 * 		The ground state of the next l subspace has an inversion quantum number m. States with the same inversion quantum number exist only in preceding l subspaces. The SUM of all Dicke state projector vectors with the same inversion quantum number must yield a valid
 * 		diagonal element i.e. SUM = 1*P[m,0,0] +0*P[m-1,1,1] +0*P[m-2,2,2] ... for the P[n11,n10,n01] two-level system vector entries. For that to be true the diagonal entry (x*P[m,0,0]) of the new ground state vector is set to one minus all diagonal entries of all other vectors of
 * 		matching inversion, and the offdiagonal entries (P[m,x,x]) are set to minus the sum of all respective other offdiagonal entries of the other vectors.
 *
 * @param	sys		the System object.
 * @param	root		the vector of the higher subspace
 * @param	elem		the root of the lower subspace
 *
 */

PetscErrorCode	DickeDistribution::JBlockShift(TLS * sys, PetscInt step, Vec elem)
{
    PetscFunctionBeginUser;

    PetscErrorCode	ierr;
    PetscInt		i,j,k,diff;
    PetscInt		numbers[step];					//the indices of the relevant contracted vectors in the dmindex and prefactors arrays


  //-------------------------------------------------------------------------------------
  //find out which vectors are needed
  //-------------------------------------------------------------------------------------
    numbers[0]		= step;						//the first one is easy
    diff		= 0;						//

    for(i=1; i < step; i++)						//the next ones are a bit more difficult
    {
      //ierr = PetscPrintf(PETSC_COMM_WORLD,"i = %d, (sys->NMls()+1) = %d, diff =%d, step = %d\n",i,(sys->NMls()+1),diff,step);
      numbers[i]	=  i*(sys->NMls()+1) - 2*diff + step-i;		// ( (N+1)*i -2*diff = N+1 +N-1 +N-3 +N-5 ... the j^2 blocksizes added up ) + the position of the relevant element in each block
      diff		+= i;						// 1,3,6,10,15
    }

    PetscInt		newindices[lengths[numbers[0]]];		//the indices of the nonzero elements of the new j^2 subspace ground state projector
    PetscReal		newentries[lengths[numbers[0]]];		//the corresponding entries of the vector

    /**
     * use lengths[numbers[0]] instead of sys->NMLS()+1 because there might be offdiagonal truncation, the symmetric vector is a symmetric superposition of all P[n,x,x] of matching inverison,
     * so it contains all relevant dm entires, therefore it is the most accurate lengthscale for this function.
     */


  //-------------------------------------------------------------------------------------
  //use the first (symmetric) Dicke state projection vector to initialize
  //-------------------------------------------------------------------------------------
    for(i=0; i < lengths[numbers[0]]; i++)				//for every element in the first subspace vector
    {
      sys->index->SetIndexLocal(dmindex[numbers[0]][i]);		//set the index
      newindices[i]	= dmindex[numbers[0]][i];			//and store the global index

      if( !sys->index->IsPol() )					//check whether its a polarization
      {
	newentries[i]	= 1-prefactors[numbers[0]][i];			//the diagonal has to add up to one, because of this 1 we need this extra single loop
	//ierr 	= PetscPrintf(PETSC_COMM_WORLD,"zero elem diag entries %e\n",(double) newentries[i]);
      }
      else
      {
	newentries[i]	= -prefactors[numbers[0]][i];			//the offdiagonals have to add up to zero
	//ierr 	= PetscPrintf(PETSC_COMM_WORLD,"zero elem offidag entries %e\n",(double) newentries[i]);
      }
    }


  //-------------------------------------------------------------------------------------
  //then go through all other vectors
  //-------------------------------------------------------------------------------------
    for(i=1; i < step; i++)						//for every subspace other than the first
    {
      for(j=0; j < lengths[numbers[i]]; j++)				//for every element in the relevant subspace vector
      {
	for(k=0; k < lengths[numbers[0]]; k++)				//for every index that we stored already
	{
	  if( dmindex[numbers[i]][j] == newindices[k] )			//compare it to the index at hand
	  {
	    newentries[k]	-= prefactors[numbers[i]][j];		//and then subtract the entries
	    //ierr 	= PetscPrintf(PETSC_COMM_WORLD,"update entries %e\n",(double) newentries[k]);
	  }
	}
      }
    }


  //-------------------------------------------------------------------------------------
  //and finally fill the new vector with the computed entries
  //-------------------------------------------------------------------------------------
    ierr = VecSet(elem,0.0); CHKERRQ(ierr);				//set it zero first, safety measure

    for(i=0; i < lengths[numbers[0]]; i++)				//for every relevant element
    {
      ierr = VecSetValue(elem,newindices[i]+sys->index->LocStart(),newentries[i],INSERT_VALUES); CHKERRQ(ierr);	//set the value
      //sys->index->SetIndexLocal(newindices[i]);
      //sys->index->PrintIndices();
      //ierr 	= PetscPrintf(PETSC_COMM_WORLD,"entries %e\n",(double) newentries[i]);
    }

    ierr = VecAssemblyBegin(elem);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(elem);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}



//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//----
//----  DickeDistFile class member functions
//----
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------


#undef __FUNCT__
#define __FUNCT__ "MakeHeaderDicke"

/**
 * @brief	Make generic header for the distributions files for some other parameter to be changed
 *
 */

PetscErrorCode DickeDistFile::MakeHeaderDicke(std::string var)
{
    PetscFunctionBeginUser;

    PetscErrorCode	ierr;
    std::string		out = "#" + var;

    ierr = PetscFPrintf(PETSC_COMM_WORLD,file,"%s\t\t",out.c_str());CHKERRQ(ierr);

    ierr = PetscFPrintf(PETSC_COMM_WORLD,file, "#<|N/2,-N/2><N/2,-N/2|>\t\t");CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file, "#<|N/2,-N/2+1><N/2,-N/2+1|>\t\t...\t\t");CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_WORLD,file, "#<|N/2-1,-N/2><N/2-1,-N/2|>\t\t...\n");CHKERRQ(ierr);

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SetupDickeDistFile"

/**
 * @brief	Setup for the mls distribution file. This function has return type PetscErrorCode for Petsc error handling, which is why I prefer this instead of a constructor.
 *
 * @param	system		the system specification object (polymorphic). Needed mainly for the header.
 * @param	filename	the name of the density.
 *
 */

PetscErrorCode DickeDistFile::SetupDickeDistFile(TLS * system, std::string filename)
{
    PetscFunctionBeginUser;

    PetscErrorCode	ierr;

  //-------------------------------------------------------------------------------------
  //open file, make gen. header
  //-------------------------------------------------------------------------------------
    ierr = SetOFile(system,filename); CHKERRQ(ierr);		//Open File, set file name
    ierr = WriteSystemParameters(system); CHKERRQ(ierr);	//write system parameters into the output file

    DickeDistribution	*ddist = new DickeDistribution();

    ierr = ddist->SetupDickeDist(system); CHKERRQ(ierr);

    length = ddist->PrintTotalNum();

    ierr = AddElem(ddist,"Dicke distribution"); CHKERRQ(ierr);

    ierr = MakeHeaderTEV(); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SetupDickeDistFile"

/**
 * @brief	Setup for the mls distribution file. This function has return type PetscErrorCode for Petsc error handling, which is why I prefer this instead of a constructor.
 *
 * @param	system		the system specification object (polymorphic). Needed mainly for the header.
 * @param	filename	the name of the density.
 * @param	var		the name of the parameter to be changed in the output file
 *
 */

PetscErrorCode DickeDistFile::SetupDickeDistFile(TLS * system, std::string filename, std::string var)
{
    PetscFunctionBeginUser;

    PetscErrorCode	ierr;

  //-------------------------------------------------------------------------------------
  //open file, make gen. header
  //-------------------------------------------------------------------------------------
    ierr = SetOFile(system,filename); CHKERRQ(ierr);		//Open File, set file name
    ierr = WriteSystemParameters(system); CHKERRQ(ierr);	//write system parameters into the output file

    DickeDistribution	*ddist = new DickeDistribution();

    ierr = ddist->SetupDickeDist(system); CHKERRQ(ierr);

    length = ddist->PrintTotalNum();

    ierr = AddElem(ddist,"Dicke distribution"); CHKERRQ(ierr);

    ierr = MakeHeaderDicke(var); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}
