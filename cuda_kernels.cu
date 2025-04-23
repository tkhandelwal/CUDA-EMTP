#include "cuda_kernels.cuh"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <iostream>

// Error checking implementation
void check_cuda_error(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(result)
            << " at " << file << ":" << line << " '" << func << "'" << std::endl;
        // In a commercial product, we might want a more graceful error handling mechanism
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

// Implementation of CUDA kernels
__global__ void updateHistoryTermsKernel(double* history, double* voltages, double* currents,
    int numElements, double timeStep) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        // This is a placeholder implementation
        // In a real implementation, this would update history terms for
        // elements with memory (inductors, capacitors, transmission lines)

        // Simple dummy implementation for testing
        if (history != nullptr) {
            history[idx] = 0.9 * history[idx] + 0.1 * (voltages[idx] + currents[idx]);
        }
    }
}

__global__ void updateCurrentVectorKernel(double* currents, double* historyTerms,
    double time, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes) {
        // Add history term contributions if available
        if (historyTerms != nullptr) {
            currents[idx] = historyTerms[idx];
        }
        else {
            currents[idx] = 0.0;
        }

        // Add current sources - this is a simplified example
        // In a real implementation, you would add actual source contributions
        currents[idx] += 100.0 * sin(2.0 * 3.14159265358979323846 * 60.0 * time);
    }
}

__global__ void calculateBranchCurrentsKernel(double* branchCurrents, double* nodeVoltages,
    double* branchParameters, int numBranches) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numBranches) {
        // This is a simplified implementation
        // In a real implementation, this would calculate branch currents based on node voltages

        // For this example, set some dummy values
        branchCurrents[idx] = idx * 0.1 + nodeVoltages[0] * 0.01;
    }
}