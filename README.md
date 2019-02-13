# Parallel-Conway's-Game-of-Life.
Conway's Game of Life using MPI, OpenMP and CUDA.

The main goal of this project is to parallelize the Conway's Game of Life.

We have made 3 different implementation. 
The first one is based on MPI library and the second is a combination of MPI and OpenMP.
The third implementation uses CUDA library and requires a NVIDIA GPU in order to be executed.

All our results can be found in the Report.pdf. This document is written in Greek but there are tables that illustrate our measurements.
These measurements have been made to a cluster with 15 machines, each of these has an Intel Core Duo processor (MPI and OpenMP) and to a server with one NVIDIA GeForce GTX780 GPU. To analyse the overhead of communication we used Paraver.

Concerning our implementation with MPI and OpenMP, we designed our application in compliance with the below rules.
First of all, we use non-blocking communication by using MPI_Isend and MPI_Irecv commands instead of MPI_Send and MPI_Recv in order to speed up the communication between processors.
In addition, we create special data types by using MPI_Type_vector command to keep the data that will be send to neighbor-processor near to each other.
Another way to reduce the overhead of communication is the use of right topology of processors. The problem that we want to solve is represented to a NxN grid and each processor works on a specific part of this grid and has to send information to neighbor processors.
By using grid topology, we achieve to keep neighbor processor near. As a result, we minimize the cost of communication.
Finally, we experiment with MPI_Allreduce command to find out the optimized termination condition.

Authors: Apostolos Florakis(https://github.com/apof) - Dimitrios Tsesmelis (https://github.com/JimTsesm)

