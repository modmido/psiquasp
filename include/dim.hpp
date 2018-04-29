
/**
 * @file	dim.hpp
 * 
 * 		This header contains the definintion of the Dim class and its children.
 * 		The Dim class has two functions:
 * 			- it represents a dimension, either a nxy "bubble" for the MLS or a bosonic mode bet/bra degree of freedom
 * 			- it is used as an identifier to locate already set dimensions for e.g. the Liouvillian matrix setup functions
 * 
 * 		At the beginning of each application code the user is supposed to add all needed dimensions, each dimension is then represented by a Dim object and stored internally in a std::list
 * 		This list is then used for all sorts of purposes, like generating the Index object, providing information for output, identifying dimensions.
 * 
 * @author	Michael Gegg
 * 
 */

#ifndef _Names
#define _Names

#include"system.hpp"

/**
 * @brief	Abstract base class for all dimensions.<br>
 * 
 */

class Dim
{
public:
    PetscReal	energy;
    PetscInt	dimlength;		//the number of different allowed number states for this dimension
    PetscInt	n00;			//is it the n00, yes, no
    
    Dim() { dimlength = 0; energy = 0.0; }
    virtual	~Dim() { } 
    
    virtual	std::string	    ToString() = 0;
    virtual	PetscInt	    IsEqual(Dim * compare) = 0;
    virtual	PetscErrorCode	PrintName() = 0;
};


/**
 * @brief	Class for all MLS dimensions. Contains all necessary information about the dimension. Also comes with some utility functions.
 * 
 */

class MLSDim : public Dim
{
public:
    PetscInt	ispol;		    //!< is it a polarization or not
    PetscInt	ket;		    //!< the ket number
    PetscInt	bra;		    //!< the bra number
    PetscInt    mlsTypeNumber;  //!< the mls type number, only really needed for multi mls usage

    //constructors
    MLSDim(PetscInt left, PetscInt right, PetscInt polflag, PetscInt indimlength, PetscReal inenergy);
    MLSDim(const MLSDim& dim, PetscInt polflag, PetscInt indimlength, PetscReal inenergy);
    MLSDim(PetscInt left, PetscInt right);
    MLSDim(PetscInt which, const MLSDim& name);
    MLSDim(const MLSDim& ketname, const MLSDim& braname);
    
    static MLSDim	Swap(MLSDim swap);
    
    //checks, I/O
    PetscInt		IsEqual(Dim * compare);
    PetscInt		IsDensity();
    std::string		ToString();
    PetscErrorCode	PrintName();
    PetscInt        TypeNumber() { return 0; }
};


/**
 * @brief    Class for all multi MLS dimensions. Contains all necessary information about the dimension. Also comes with some utility functions.
 *
 */

class MultiMLSDim: public MLSDim
{
public:
    
    //constructors
    MultiMLSDim(PetscInt left, PetscInt right, PetscInt polflag, PetscInt indimlength, PetscReal inenergy, PetscInt typeNumber);
    MultiMLSDim(const MultiMLSDim& dim, PetscInt polflag, PetscInt indimlength, PetscReal inenergy);
    MultiMLSDim(PetscInt left, PetscInt right, PetscInt typeNumber);
    MultiMLSDim(PetscInt which, const MultiMLSDim& name);
    MultiMLSDim(const MultiMLSDim& ketname, const MultiMLSDim& braname);
    
    static MultiMLSDim    Swap(MultiMLSDim swap);
    
    //checks, I/O
    PetscInt            IsEqual(Dim * compare);
    std::string         ToString();
    PetscErrorCode      PrintName();
    PetscInt            SameType(Dim * compare);
};


/**
 * @brief	Class for all mode dimensions. Contains all necessary information about the dimension. Also comes with some utility functions.
 * 
 */

class ModeDim : public Dim
{
public:
    PetscInt	number;
    PetscInt	ket;

    //constructors
    ModeDim(PetscInt choose, PetscInt n, PetscInt dimlength, PetscReal energy);
    ModeDim(PetscInt choose, PetscInt n);
    
    //checks, I/O
    std::string		ToString();
    PetscInt		IsEqual(Dim* compare);
    PetscErrorCode	PrintName();
    PetscErrorCode	PrintMode();
};

#endif		// _Names
