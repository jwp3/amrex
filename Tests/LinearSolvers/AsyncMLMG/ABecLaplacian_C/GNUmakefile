DEBUG = FALSE

USE_MPI  = TRUE
USE_OMP  = FALSE


USE_HYPRE = FALSE
USE_PETSC = FALSE

COMP = gnu

DIM = 3

USE_CUDA        = TRUE
USE_HIP         = FALSE

DEBUG           = FALSE
PRECISION       = DOUBLE
VERBOSE         = FALSE
TINY_PROFILE    = TRUE

AMREX_HOME = ../../../..

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package

Pdirs 	:= Base Boundary LinearSolvers/MLMG

Ppack	+= $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)

include $(Ppack)

include $(AMREX_HOME)/Tools/GNUMake/Make.rules

