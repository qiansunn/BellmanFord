# Parallel Implementation of Bellman Ford Algorithm
----------
Bellman-Ford algorithm is a well-known solution to “the single-source shortest
path(SSSP)” problem. It is slower than Dijkstra's algorithm, but more versatile, as it is
capable of handling graphs in which some of the edge weights are negative numbers.
The input graph G(V, E) for this assignment is connected, directed and may contain
negative weights. The algorithm finds a shortest path from a specified vertex (the
‘source vertex’) to every other vertex in the graph. If there is a negative cycle (a cycle
on which the sum of weights is negative) in the graph, there will be no shortest path. In
this case, the algorithm will find no result. Ths repository consists mpi, openmp and CUDA versions of Bellman-Ford algorithm.  

## Package list

serial_bellman_ford.cpp: a serial version of bellman-ford algorithm  
mpi_bellman_ford.cpp: a mpi version of bellman-ford algorithm  
openmp_bellman_ford.cpp: an openmp version of bellman-ford algorithm  
cuda_bellman_ford.cu: a cuda version of bellman-ford algorithm  
cuda_dijkstra_solution.cu: a solution code for dijkstra algorithm  
two sets of sample input/output files  

## Compile and run:

### serial_bellman_ford
Compile:
```Bash
$ g++ -std=c++11 -o serial_bellman_ford serial_bellman_ford.cpp
```
Run:
```Bash
$ ./serial_bellman_ford <input file>
```
e.g.
```
./serial_bellman_ford input1.txt
```
The output is file output.txt

### mpi_bellman_ford
Compile:
```Bash
$ mpic++ -std=c++11 -o mpi_bellman_ford mpi_bellman_ford.cpp
```
Run:
```Bash
$ mpiexec -n <number of processes> ./mpi_bellman_ford <intput file>
```

### openmp_bellman_ford
Compile:
```Bash
$ g++ -std=c++11 -fopenmp -o openmp_bellman_ford openmp_bellman_ford.cpp
```
Run:
```Bash
$ ./openmp_bellman_ford <intput file> <number of threads>
```

### cuda_bellman_ford
Compile:
```Bash
$ nvcc -std=c++11 -arch=sm_52 -o cuda_bellman_ford cuda_bellman_ford.cu
```
Run:
```Bash
$ ./cuda_bellman_ford <intput file> <number of blocks per grid> <number of threads
per block>
```


## Input and output files

The input file will be in following format:
1. The first line is an integer N, the number of vertices in the input graph.
2. The following lines are an N*N adjacency matrix mat, one line per row. The entry in row v and column w, mat[v][w], is the distance (weight) from vertex v to vertex w. All distances are integers. If there is no edge joining vertex v and w, mat[v][w] will be 1000000 to represent infinity.

The vertex labels are non-negative, consecutive integers, for an input graph with N
vertices, the vertices will be labeled by 0, 1, 2, …, N-1. We always use vertex 0
as the source vertex.

The output file consists the distances from vertex 0 to all vertices, in the increasing order of the vertex label (vertex 0, 1, 2, … and so on), one distance per line. If there are at least one negative cycle (the sum of the weights of the cycle is
negative in the graph), the program will set variable has_negative_cycle to true and print "FOUND NEGATIVE CYCLE!" as there will be no shortest path.


