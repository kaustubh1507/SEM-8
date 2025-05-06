// %%writefile cuda_example.cu
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// ---------------- VECTOR ADDITION ----------------
__global__ void vectorAdd(const int* A, const int* B, int* C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

// ---------------- MATRIX MULTIPLICATION ----------------
__global__ void matrixMul(const int* A, const int* B, int* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;
    if (row < width && col < width) {
        for (int k = 0; k < width; ++k)
            sum += A[row * width + k] * B[k * width + col];
        C[row * width + col] = sum;
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        cerr << msg << " " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // ----------------- VECTOR ADDITION -----------------
    const int N = 5;
    cout << "=== Vector Addition ===\n";
    int h_A[N] = {1, 2, 3, 4, 5};
    int h_B[N] = {10, 20, 30, 40, 50};
    int h_C[N];

    int *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc((void**)&d_A, N * sizeof(int)), "cudaMalloc A failed");
    checkCudaError(cudaMalloc((void**)&d_B, N * sizeof(int)), "cudaMalloc B failed");
    checkCudaError(cudaMalloc((void**)&d_C, N * sizeof(int)), "cudaMalloc C failed");

    checkCudaError(cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice), "Memcpy A failed");
    checkCudaError(cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice), "Memcpy B failed");

    vectorAdd<<<1, N>>>(d_A, d_B, d_C, N);
    checkCudaError(cudaDeviceSynchronize(), "vectorAdd failed");
    checkCudaError(cudaMemcpy(h_C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost), "Memcpy C failed");

    cout << "A: ";
    for (int i = 0; i < N; i++) cout << h_A[i] << " ";
    cout << "\nB: ";
    for (int i = 0; i < N; i++) cout << h_B[i] << " ";
    cout << "\nC = A + B: ";
    for (int i = 0; i < N; i++) cout << h_C[i] << " ";
    cout << "\n";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    // ----------------- MATRIX MULTIPLICATION -----------------
    cout << "\n=== Matrix Multiplication (2x2) ===\n";
    const int WIDTH = 2;
    int h_MatA[4] = {1, 2, 3, 4}; // 2x2: [1 2; 3 4]
    int h_MatB[4] = {5, 6, 7, 8}; // 2x2: [5 6; 7 8]
    int h_MatC[4];

    int *d_MatA, *d_MatB, *d_MatC;
    checkCudaError(cudaMalloc((void**)&d_MatA, 4 * sizeof(int)), "cudaMalloc MatA failed");
    checkCudaError(cudaMalloc((void**)&d_MatB, 4 * sizeof(int)), "cudaMalloc MatB failed");
    checkCudaError(cudaMalloc((void**)&d_MatC, 4 * sizeof(int)), "cudaMalloc MatC failed");

    checkCudaError(cudaMemcpy(d_MatA, h_MatA, 4 * sizeof(int), cudaMemcpyHostToDevice), "Memcpy MatA failed");
    checkCudaError(cudaMemcpy(d_MatB, h_MatB, 4 * sizeof(int), cudaMemcpyHostToDevice), "Memcpy MatB failed");

    dim3 threadsPerBlock2D(2, 2);
    dim3 blocksPerGrid2D(1, 1);
    matrixMul<<<blocksPerGrid2D, threadsPerBlock2D>>>(d_MatA, d_MatB, d_MatC, WIDTH);
    checkCudaError(cudaDeviceSynchronize(), "matrixMul failed");
    checkCudaError(cudaMemcpy(h_MatC, d_MatC, 4 * sizeof(int), cudaMemcpyDeviceToHost), "Memcpy MatC failed");

    cout << "Matrix A:\n";
    cout << h_MatA[0] << " " << h_MatA[1] << "\n" << h_MatA[2] << " " << h_MatA[3] << "\n";

    cout << "Matrix B:\n";
    cout << h_MatB[0] << " " << h_MatB[1] << "\n" << h_MatB[2] << " " << h_MatB[3] << "\n";

    cout << "Matrix C = A x B:\n";
    cout << h_MatC[0] << " " << h_MatC[1] << "\n" << h_MatC[2] << " " << h_MatC[3] << "\n";

    cudaFree(d_MatA); cudaFree(d_MatB); cudaFree(d_MatC);
    return 0;
}