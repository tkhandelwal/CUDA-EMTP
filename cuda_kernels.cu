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
        // This is a critical function that needs improvement for correct branch currents

        // For inductors: history = v(t) - L*di/dt
        // For capacitors: history = i(t) - C*dv/dt
        // For transmission lines: involves delayed terms

        // Current implementation is just a placeholder
        // In a real implementation, this would use proper element data

        // For demonstration, let's implement a standard inductor history update
        // Assuming history[idx] represents an inductor's history term
        // and voltages[idx], currents[idx] are terminal values

        // Update inductor history term
        // For inductor with terminals a and b: v_ab = L*di/dt + history
        // history = v_ab(t) - L*di/dt

        // Constants (would come from proper element data)
        double inductance = 0.01;  // 10 mH example inductor
        double resistance = 0.001; // 1 mΩ series resistance

        // Previous current (would need to be stored properly)
        static double prevCurrent = 0.0;

        // Calculate di/dt
        double didt = (currents[idx] - prevCurrent) / timeStep;

        // Calculate new history term
        history[idx] = voltages[idx] - inductance * didt - resistance * currents[idx];

        // Store current for next iteration
        prevCurrent = currents[idx];

        // For capacitors (example for another element type)
        // double capacitance = 10e-6;  // 10 µF example capacitor
        // double dvdt = (voltages[idx] - prevVoltage) / timeStep;
        // history[idx] = currents[idx] - capacitance * dvdt;
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