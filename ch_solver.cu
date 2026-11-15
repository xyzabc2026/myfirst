// -----------------------------------------------------------------------------
// CUDA C++ port of ch_solver.f90 (Finite Difference Cahn-Hilliard, 2D, periodic BCs)
//
// Compile : nvcc -O3 -o ch_solver ch_solver.cu
// Run     : ./ch_solver        (writes ch.dat)
// -----------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err), __FILE__, __LINE__);               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// --- simulation cell parameters
constexpr int    Nx = 1024;
constexpr int    Ny = 1024;
constexpr float dx = 1.0;
constexpr float dy = 1.0;

// --- time integration parameters
constexpr int    nsteps = 10000;
constexpr int    nprint = 1000;
constexpr float dt     = 0.01;

// --- material parameters
constexpr float c0        = 0.4;
constexpr float mobility  = 1.0;
constexpr float grad_coef = 0.5;

// --- microstructure parameters
constexpr float noise = 0.02;
constexpr float A     = 1.0;

// Row-major index: i -> x (0..Nx-1), j -> y (0..Ny-1), j fastest
__device__ __forceinline__ int idx(int i, int j) 
{ 
    return i * Ny + j; 
}

// Kernel 1: mu = df/dc - kappa * laplacian(c)
__global__ void chemical_potential(const float* __restrict__ con,
                                   float* __restrict__ mu)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Nx || j >= Ny) return;

    // periodic neighbours
    int ip = (i + 1) % Nx, im = (i - 1 + Nx) % Nx;
    int jp = (j + 1) % Ny, jm = (j - 1 + Ny) % Ny;

    float c = con[idx(i, j)];

    // free energy derivative
    float dfdcon = A * (2.0 * c * (1.0 - c) * (1.0 - c)
                       - 2.0 * c * c * (1.0 - c));

    float lap_con = (con[idx(ip, j)] + con[idx(im, j)] +
                      con[idx(i, jm)] + con[idx(i, jp)] - 4.0 * c) / (dx * dy);

    mu[idx(i, j)] = dfdcon - grad_coef * lap_con;
}

// Kernel 2: c += dt * M * laplacian(mu), then clamp
__global__ void update_concentration(float* __restrict__ con,
                                     const float* __restrict__ mu)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Nx || j >= Ny) return;

    int ip = (i + 1) % Nx, im = (i - 1 + Nx) % Nx;
    int jp = (j + 1) % Ny, jm = (j - 1 + Ny) % Ny;

    float lap_mu = (mu[idx(ip, j)] + mu[idx(im, j)] +
                     mu[idx(i, jm)] + mu[idx(i, jp)] - 4.0 * mu[idx(i, j)])
                    / (dx * dy);

    float c = con[idx(i, j)] + dt * mobility * lap_mu;

    // for small deviations
    if (c >= 0.99999) c = 0.99999;
    if (c <  0.00001) c = 0.00001;

    con[idx(i, j)] = c;
}

int main()
{
    const size_t N     = (size_t)Nx * Ny;
    const size_t bytes = N * sizeof(float);

    // --- initial microstructure (host)
    std::vector<float> h_con(N);
    std::mt19937_64 gen(12345);
    std::uniform_real_distribution<float> uni(0.0, 1.0);
    for (size_t k = 0; k < N; ++k)
        h_con[k] = c0 + noise * (0.5 - uni(gen));

    // --- device memory
    float *d_con = nullptr, *d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_con, bytes));
    CUDA_CHECK(cudaMalloc(&d_mu,  bytes));
    CUDA_CHECK(cudaMemcpy(d_con, h_con.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((Ny + block.x - 1) / block.x, (Nx + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // --- start microstructure evolution
    for (int step = 1; step <= nsteps; ++step) {
        chemical_potential<<<grid, block>>>(d_con, d_mu);
        update_concentration<<<grid, block>>>(d_con, d_mu);

        if (step % nprint == 0) {
            CUDA_CHECK(cudaGetLastError());
            printf(" Done steps = %d\n", step);
        }
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // --- copy back and write result
    CUDA_CHECK(cudaMemcpy(h_con.data(), d_con, bytes, cudaMemcpyDeviceToHost));

    FILE* fp = fopen("ch.dat", "w");
    if (!fp) { fprintf(stderr, "Cannot open ch.dat\n"); return EXIT_FAILURE; }
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j)
            fprintf(fp, " %.15e", h_con[(size_t)i * Ny + j]);
        fprintf(fp, "\n");
    }
    fclose(fp);

    printf("---------------------------------\n");
    printf(" Time = %10.3f seconds.\n", ms / 1000.0f);

    cudaFree(d_con);
    cudaFree(d_mu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

// To visualize with gnuplot.
// plot 'ch.dat' matrix with image notitle

// or write this in  terminal 
//  gnuplot plot_ch.gp