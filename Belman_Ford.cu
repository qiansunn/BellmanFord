#include <string>
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using std::string;
using std::cout;
using std::endl;

#define INF 1000000

/*
 * This is a CHECK function to check CUDA calls
 */
#define CHECK(call)                                                            \
    {                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
                exit(1);                                                               \
    }                                                                          \
    }

/**
 * utils is a namespace for utility functions
 * including I/O (read input file and print results) and matrix dimension convert(2D->1D) function
 */
namespace utils {
int N; //number of vertices
int *mat; // the adjacency matrix

void abort_with_error_message(string msg) {
    std::cerr << msg << endl;
    abort();
}

//translate 2-dimension coordinate to 1-dimension
int convert_dimension_2D_1D(int x, int y, int n) {
    return x * n + y;
}

int read_file(string filename) {
    std::ifstream inputf(filename, std::ifstream::in);
    if (!inputf.good()) {
        abort_with_error_message("ERROR OCCURRED WHILE READING INPUT FILE");
    }
    inputf >> N;
    //input matrix should be smaller than 20MB * 20MB (400MB, we don't have too much memory for multi-processors)
    assert(N < (1024 * 1024 * 20));
    mat = (int *) malloc(N * N * sizeof(int));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            inputf >> mat[convert_dimension_2D_1D(i, j, N)];
        }
    return 0;
}

int print_result(bool has_negative_cycle, int *dist) {
    std::ofstream outputf("output.txt", std::ofstream::out);
    if (!has_negative_cycle) {
        for (int i = 0; i < N; i++) {
            if (dist[i] > INF)
                dist[i] = INF;
            outputf << dist[i] << '\n';
        }
        outputf.flush();
    } else {
        outputf << "FOUND NEGATIVE CYCLE!" << endl;
    }
    outputf.close();
    return 0;
}
}//namespace utils

__global__ void bellman_ford_one_iter(int n, int *d_mat, int *d_dist, bool *d_has_next, int iter_num){
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    int elementSkip = blockDim.x * gridDim.x;

    if(global_tid >= n) return;
    for(int u = 0 ; u < n ; u ++){
        for(int v = global_tid; v < n; v+= elementSkip){
            int weight = d_mat[u * n + v];
            if(weight < INF){
                int new_dist = d_dist[u] + weight;
                if(new_dist < d_dist[v]){
                    d_dist[v] = new_dist;
                    *d_has_next = true;
                }
            }
        }
    }
}

/**
 * Bellman-Ford algorithm. Find the shortest path from vertex 0 to other vertices.
 * @param blockPerGrid number of blocks per grid
 * @param threadsPerBlock number of threads per block
 * @param n input size
 * @param *mat input adjacency matrix
 * @param *dist distance array
 * @param *has_negative_cycle a bool variable to recode if there are negative cycles
 */
void bellman_ford(int blocksPerGrid, int threadsPerBlock, int n, int *mat, int *dist, bool *has_negative_cycle) {
    dim3 blocks(blocksPerGrid);
    dim3 threads(threadsPerBlock);

    int iter_num = 0;
    int *d_mat, *d_dist;
    bool *d_has_next, h_has_next;

    cudaMalloc(&d_mat, sizeof(int) * n * n);
    cudaMalloc(&d_dist, sizeof(int) *n);
    cudaMalloc(&d_has_next, sizeof(bool));

    *has_negative_cycle = false;

    for(int i = 0 ; i < n; i ++){
        dist[i] = INF;
    }

    dist[0] = 0;
    cudaMemcpy(d_mat, mat, sizeof(int) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, dist, sizeof(int) * n, cudaMemcpyHostToDevice);

    for(;;){
        h_has_next = false;
        cudaMemcpy(d_has_next, &h_has_next, sizeof(bool), cudaMemcpyHostToDevice);

        bellman_ford_one_iter<<<blocks, threads>>>(n, d_mat, d_dist, d_has_next, iter_num);
        CHECK(cudaDeviceSynchronize());
        cudaMemcpy(&h_has_next, d_has_next, sizeof(bool), cudaMemcpyDeviceToHost);

        iter_num++;
        if(iter_num >= n-1){
            *has_negative_cycle = true;
            break;
        }
        if(!h_has_next){
            break;
        }

    }
    if(! *has_negative_cycle){
        cudaMemcpy(dist, d_dist, sizeof(int) * n, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_mat);
    cudaFree(d_dist);
    cudaFree(d_has_next);
}

// CPU-based Bellman-Ford algorithm
void bellman_ford_cpu(int n, int *mat, int *dist, bool *has_negative_cycle) {
    *has_negative_cycle = false;

    for (int i = 0; i < n; ++i) {
        dist[i] = INF;
    }
    dist[0] = 0;

    for (int i = 0; i < n - 1; ++i) {
        for (int u = 0; u < n; ++u) {
            for (int v = 0; v < n; ++v) {
                int weight = mat[u * n + v];
                if (weight < INF) {
                    int new_dist = dist[u] + weight;
                    if (new_dist < dist[v]) {
                        dist[v] = new_dist;
                    }
                }
            }
        }
    }

    // Check for negative cycles
    for (int u = 0; u < n; ++u) {
        for (int v = 0; v < n; ++v) {
            int weight = mat[u * n + v];
            if (weight < INF) {
                int new_dist = dist[u] + weight;
                if (new_dist < dist[v]) {
                    *has_negative_cycle = true;
                    return;
                }
            }
        }
    }
}

// Function to identify GPU
string identify_gpu() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        return "No GPU found";
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // Assuming only one GPU is used
    return prop.name;
}

int main(int argc, char **argv) {
    if (argc <= 1) {
        utils::abort_with_error_message("INPUT FILE WAS NOT FOUND!");
    }
    if (argc <= 3) {
        utils::abort_with_error_message("blocksPerGrid or threadsPerBlock WAS NOT FOUND!");
    }

    string filename = argv[1];
    int blockPerGrid = atoi(argv[2]);
    int threadsPerBlock = atoi(argv[3]);

    int *dist;
    bool has_negative_cycle = false;

    assert(utils::read_file(filename) == 0);
    dist = (int *) calloc(sizeof(int), utils::N);

    // CPU and GPU timing variables
    float gpu_ms_wall, cpu_ms_wall;

    // Start GPU timer
    timeval start_wall_time_t_gpu, end_wall_time_t_gpu;
    cudaDeviceReset();
    gettimeofday(&start_wall_time_t_gpu, nullptr);
    // Bellman-Ford algorithm on GPU
    bellman_ford(blockPerGrid, threadsPerBlock, utils::N, utils::mat, dist, &has_negative_cycle);
    gettimeofday(&end_wall_time_t_gpu, nullptr);
    gpu_ms_wall = ((end_wall_time_t_gpu.tv_sec - start_wall_time_t_gpu.tv_sec) * 1000 * 1000
            + end_wall_time_t_gpu.tv_usec - start_wall_time_t_gpu.tv_usec) / 1000.0;

    // Start CPU timer
    timeval start_wall_time_t_cpu, end_wall_time_t_cpu;
    gettimeofday(&start_wall_time_t_cpu, nullptr);
    // Bellman-Ford algorithm on CPU
    bellman_ford_cpu(utils::N, utils::mat, dist, &has_negative_cycle);
    gettimeofday(&end_wall_time_t_cpu, nullptr);
    cpu_ms_wall = ((end_wall_time_t_cpu.tv_sec - start_wall_time_t_cpu.tv_sec) * 1000 * 1000
            + end_wall_time_t_cpu.tv_usec - start_wall_time_t_cpu.tv_usec) / 1000.0;

    // Print GPU and CPU times
    cout << "Used GPU: " << identify_gpu() << endl;
    cout << "GPU TIME: " << gpu_ms_wall / 1000.0 << " s" << endl;
    cout << "CPU TIME: " << cpu_ms_wall / 1000.0 << " s" << endl;

    utils::print_result(has_negative_cycle, dist);
    free(dist);
    free(utils::mat);

    return 0;
}
