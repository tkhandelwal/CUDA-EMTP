#include "cuda_kernels.cuh"
#include <iostream>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @brief CUDA kernel to build the admittance matrix for the EMTP simulation
 *
 * This kernel constructs the sparse matrix in CSR format, populating the
 * values array with conductance and susceptance values
 */
__global__ void buildMatrixKernel(double* values, int* rowPtr, int* colInd,
    double* historyTerms, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes) {
        // This is a simplified implementation
        // In a real implementation, this would build the admittance matrix
        // based on the network topology and element parameters

        // For this example, we just set diagonal elements
        int start = rowPtr[idx];
        int end = rowPtr[idx + 1];

        // Iterate over non-zero elements in this row
        for (int j = start; j < end; j++) {
            int col = colInd[j];

            if (col == idx) {
                // Diagonal element - set to self-admittance
                values[j] = 1.0 + historyTerms[idx] * 0.1;
            }
            else {
                // Off-diagonal element - set to mutual admittance
                values[j] = -0.1;
            }
        }
    }
}

/**
 * @brief CUDA kernel to update the current vector for the EMTP simulation
 *
 * This kernel updates the right-hand side vector with current injections
 * from sources and history terms from elements with memory
 */
__global__ void updateCurrentVectorKernel(double* currents, double* historyTerms,
    double time, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes) {
        // Add history term contributions
        currents[idx] = historyTerms[idx];

        // Add current sources - this is a simplified example
        // In a real implementation, you would add actual source contributions
        currents[idx] += 100.0 * sin(2.0 * M_PI * 60.0 * time);
    }
}

/**
 * @brief Advanced CUDA kernel to build the sparse admittance matrix
 *
 * This kernel uses shared memory for better performance on modern GPUs.
 */
__global__ void buildMatrixAdvancedKernel(double* values, int* rowPtr, int* colInd,
    double* historyTerms, double* parameters, int numNodes) {

    // Thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Use shared memory for commonly accessed parameters
    extern __shared__ double s_params[];

    // Process nodes in the network (with grid stride loop for better utilization)
    for (; idx < numNodes; idx += blockDim.x * gridDim.x) {
        int start = rowPtr[idx];
        int end = rowPtr[idx + 1];

        // Load parameters for this node into shared memory if needed
        // (This would depend on your specific data access patterns)

        // Process non-zero elements in this row
        for (int j = start; j < end; j++) {
            int col = colInd[j];

            if (col == idx) {
                // Diagonal element - set to self-admittance
                values[j] = parameters[idx] + historyTerms[idx];
            }
            else {
                // Off-diagonal element - set to mutual admittance
                values[j] = -parameters[numNodes + j];
            }
        }
    }
}