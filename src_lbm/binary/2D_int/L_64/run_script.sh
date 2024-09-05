#!/bin/bash

mpirun -n 4 main3d.gnu.TPROF.MPI.ex inputs
mpirun -n 4 main3d.gnu.TPROF.MPI.ex inputs_restart
