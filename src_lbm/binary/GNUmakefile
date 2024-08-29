AMREX_HOME ?= ../../../amrex
FHDEX_HOME ?= ../../../FHDeX
HDF5_HOME = /usr/lib/x86_64-linux-gnu/hdf5/openmpi

DEFINES += -DMAX_SPECIES=2

DEBUG	= FALSE
TINY_PROFILE = TRUE

DIM	= 3

COMP    = gnu

USE_MPI   = TRUE
USE_HDF5  = TRUE
USE_OMP   = FALSE
USE_CUDA  = FALSE
USE_HIP   = FALSE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package

include $(FHDEX_HOME)/src_common/Make.package
VPATH_LOCATIONS   += $(FHDEX_HOME)/src_common/
INCLUDE_LOCATIONS += $(FHDEX_HOME)/src_common/

include $(FHDEX_HOME)/src_analysis/Make.package
VPATH_LOCATIONS   += $(FHDEX_HOME)/src_analysis/
INCLUDE_LOCATIONS += $(FHDEX_HOME)/src_analysis/

include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules

ifeq ($(findstring cgpu, $(HOST)), cgpu)
  CXXFLAGS += $(FFTW)
endif

ifeq ($(USE_CUDA),TRUE)
  LIBRARIES += -lcufft
else
  LIBRARIES += $(shell pkg-config --libs fftw3)
  LIBRARIES += $(shell pkg-config --libs blas)
  LIBRARIES += $(shell pkg-config --libs lapack)
  LIBRARIES += $(shell pkg-config --libs lapacke)
endif
