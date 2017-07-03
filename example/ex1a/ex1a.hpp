
/**
 * @file	ex1a.hpp
 * 
 * 		User specific application header.
 * 		Contains derived classes that specify setup functions.
 * 
 * @author	Michael Gegg
 * 
 */
 
#include"../../include/psiquasp.hpp"

#define		hbar	0.658212196			// hbar in [eV*fs]

/*
 * Derived class for the open Tavis-Cummings model (OTC).
 * This just provides a setup function
 */

class OTC: public TLS
{
  public:
    void	Setup(Vec * dm, Mat * AA);
};


/*
 * Dervived class for the program output. Provides a capsule for all output related things.
 * Also just provides a setup function.
 */

class MyOut: public Output
{
  public:
    void	SetupMyOut(OTC * system);
};


/*
 * Class for the user specified observables output file, child of PropFile.
 * The only thing that needs to be specified here are the actual observables that should be printed to the file.
 */

class ObservablesFile: public PropFile
{
  public:
    void	SetupMyObsFile(OTC * system, std::string name);
};


/*
 * Class for the user specified correlation function file. Child of PropFile. 
 */

class CorrelationsFile: public PropFile
{
  public:
    void	SetupMyGnFile(OTC * sys, std::string name);			//set it up
};
