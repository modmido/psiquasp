
/**
 * @file	twolevelsystems.hpp
 * 
 * 		Specification class of System. This class simplifies the usage of the System class utilities since for two-level systems they can be grouped together to provide premade setup functions for typical Liouvillians, distributions and observables.
 * 
 * @author	Michael Gegg
 * 
 */

#include"system.hpp"
#include"output.hpp"
#include"distributions.hpp"
#include"dim.hpp"
#include"index.hpp"

/**
 * @brief	Special two-level systems class. Simplifies the usage of the System class functions since for two-level systems many setup functions can be grouped together to provide actual physical functionality.
 * 
 */

class TLS: public System
{
  protected:
    //dimension setup
    PetscErrorCode	TLSAdd(PetscInt ntls, PetscInt n10cutoff, PetscInt n01cutoff, PetscReal energy);
    
    
    //specialized Liouvillian setup functions
    PetscErrorCode	AddTLSH0(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscScalar matrixelem);
    PetscErrorCode	AddTavisCummingsHamiltonianRWA(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt photonnumber, PetscScalar matrixelem);
    PetscErrorCode	AddTavisCummingsHamiltonianNoRWA(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscInt photonnumber, PetscScalar matrixelem);
    PetscErrorCode	AddTLSCoherentDrive(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscScalar matrixelem);
    
    PetscErrorCode	AddTLSSpontaneousEmission(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscReal matrixelem);
    PetscErrorCode	AddTLSPureDephasing(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscReal matrixelem);
    PetscErrorCode	AddTLSIncoherentPump(Mat AA, PetscInt * d_nnz, PetscInt * o_nnz, PetscInt choose, PetscReal matrixelem);
    
    
  public:
    //Liouville operators acting from the left, e.g. for custom observable types
    PetscErrorCode	MatTLSJ10Left(Mat *AA, PetscScalar matrixelem);
    PetscErrorCode	MatTLSJ11Left(Mat *AA, PetscScalar matrixelem);
    PetscErrorCode	MatTLSTCHamiltonianRWALeft(Mat *AA, PetscInt photonnumber, PetscScalar matrixelem);
    PetscErrorCode	MatTLSTCHamiltonianNoRWALeft(Mat *AA, PetscInt photonnumber, PetscScalar matrixelem);
    PetscErrorCode	MatTLSCoherentDriveLeft(Mat *AA, PetscScalar matrixelem);
    
    //useful stuff
    PetscErrorCode	MatMLSLeftRightCollectiveRaisingOperator(Mat * AA);
};


/**
 * @brief	DickeDistribution class, child of DModular. Prints the occupation in the diagonal Dicke states, i.e. computes sth like <|l,m>< l,m|> for all l and m
 * 
 */

class DickeDistribution: public DModular
{
  protected:
    PetscErrorCode	JBlockShift(TLS * sys, PetscInt step, Vec elem);	//create a single projector in the next pseudospin subspace
    
  public:
    PetscErrorCode	SetupDickeDist(TLS * sys);
};


/**
 * @brief	DickeDistribution class, child of DModular. Prints the occupation in the diagonal Dicke states, i.e. computes sth like <|l,m>< l,m|> for all l and m
 * 
 */

class DickeDistFile: public DistFile
{
  protected:
    PetscErrorCode	MakeHeaderDicke(std::string var);
    
  public:
    PetscErrorCode	SetupDickeDistFile(TLS * sys, std::string filename);
    PetscErrorCode	SetupDickeDistFile(TLS * sys, std::string filename, std::string var);
};