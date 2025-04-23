#include "cuda_kernels.cuh"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__global__ void transformerKernel(double* history, double* voltages, double* currents,
    double* parameters, int numTransformers, double timeStep) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numTransformers) {
        // Extract parameters for this transformer
        double primaryV = parameters[idx * 5];
        double secondaryV = parameters[idx * 5 + 1];
        double rating = parameters[idx * 5 + 2]; // MVA
        double leakageX = parameters[idx * 5 + 3]; // % on transformer base
        double ratio = primaryV / secondaryV;

        // Get node indices
        int primaryNode = static_cast<int>(parameters[idx * 5 + 4]);
        int secondaryNode = static_cast<int>(parameters[idx * 5 + 5]);

        // Get voltages and currents
        double primaryVoltage = voltages[primaryNode];
        double secondaryVoltage = voltages[secondaryNode];
        double primaryCurrent = currents[2 * idx];
        double secondaryCurrent = currents[2 * idx + 1];

        // Calculate base impedance
        double baseZ = (primaryV * primaryV) / (rating * 1e6);
        double leakageZ = leakageX * baseZ / 100.0;

        // Update history term for this transformer
        history[idx] = (primaryVoltage - secondaryVoltage * ratio) / leakageZ - primaryCurrent;
    }
}

__global__ void transmissionLineKernel(double* history, double* voltages, double* currents,
    double* parameters, int numLines, double timeStep) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numLines) {
        // Extract parameters for this transmission line
        double R = parameters[idx * 4];
        double L = parameters[idx * 4 + 1];
        double C = parameters[idx * 4 + 2];
        double length = parameters[idx * 4 + 3];

        // Compute characteristic impedance and propagation velocity
        double Z0 = sqrt(L / C);
        double v = 1.0 / sqrt(L * C);

        // Compute travel time
        double travelTime = length / v;

        // Number of time steps for travel time
        int delaySteps = static_cast<int>(travelTime / timeStep);

        // Update history terms based on delayed values
        // Note: This is a simplified model; a real implementation would
        // use a more accurate method with interpolation and proper
        // handling of the boundary conditions

        // For now, we'll use a simple approximation
        int fromNode = static_cast<int>(parameters[idx * 4 + 4]);
        int toNode = static_cast<int>(parameters[idx * 4 + 5]);

        double fromVoltage = voltages[fromNode];
        double toVoltage = voltages[toNode];
        double fromCurrent = currents[2 * idx];     // Current from "from" node
        double toCurrent = currents[2 * idx + 1];   // Current from "to" node

        // Update history terms
        // These would be stored in a circular buffer in a real implementation

        // For the sake of this example, we'll just use a simple update rule
        history[2 * idx] = (fromVoltage - Z0 * fromCurrent) * exp(-R * length / (2 * Z0));
        history[2 * idx + 1] = (toVoltage - Z0 * toCurrent) * exp(-R * length / (2 * Z0));
    }
}


/**
 * @brief CUDA kernel to calculate branch currents based on node voltages
 *
 * This kernel computes currents in each branch of the network based on
 * the voltage difference and branch parameters
 */
__global__ void calculateBranchCurrentsKernel(double* branchCurrents, double* nodeVoltages,
    double* branchParameters, int numBranches) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numBranches) {
        // Extract branch parameters and node indices
        int fromNode = static_cast<int>(branchParameters[idx * 6]);
        int toNode = static_cast<int>(branchParameters[idx * 6 + 1]);
        double resistance = branchParameters[idx * 6 + 2];
        double reactance = branchParameters[idx * 6 + 3];
        double susceptance = branchParameters[idx * 6 + 4];
        int branchType = static_cast<int>(branchParameters[idx * 6 + 5]);

        // Get node voltages
        double fromVoltage = nodeVoltages[fromNode];
        double toVoltage = nodeVoltages[toNode];

        // Calculate voltage difference
        double voltageDiff = fromVoltage - toVoltage;

        // Calculate branch current based on branch type
        switch (branchType) {
        case 0: // Transmission line
            if (resistance > 0.0) {
                branchCurrents[idx] = voltageDiff / resistance;
            }
            else {
                branchCurrents[idx] = 0.0;
            }
            break;

        case 1: // Transformer
        {
            double ratio = branchParameters[idx * 6 + 6];  // Turns ratio
            branchCurrents[idx] = (fromVoltage - toVoltage * ratio) / resistance;
        }
        break;

        case 2: // Impedance
        {
            double impedance = sqrt(resistance * resistance + reactance * reactance);
            if (impedance > 0.0) {
                branchCurrents[idx] = voltageDiff / impedance;
            }
            else {
                branchCurrents[idx] = 0.0;
            }
        }
        break;

        default: // Default case
            if (resistance > 0.0) {
                branchCurrents[idx] = voltageDiff / resistance;
            }
            else {
                branchCurrents[idx] = 0.0;
            }
        }
    }
}

/**
 * @brief Advanced CUDA kernel for calculating branch currents with waveform memory
 */
__global__ void calculateBranchCurrentsAdvancedKernel(double* branchCurrents, double* nodeVoltages,
    double* branchParameters, double* historyTerms, double time, int numBranches) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numBranches) {
        // Extract branch parameters and node indices
        int fromNode = static_cast<int>(branchParameters[idx * 8]);
        int toNode = static_cast<int>(branchParameters[idx * 8 + 1]);
        double resistance = branchParameters[idx * 8 + 2];
        double reactance = branchParameters[idx * 8 + 3];
        double susceptance = branchParameters[idx * 8 + 4];
        int branchType = static_cast<int>(branchParameters[idx * 8 + 5]);
        double param1 = branchParameters[idx * 8 + 6];
        double param2 = branchParameters[idx * 8 + 7];

        // Get node voltages
        double fromVoltage = nodeVoltages[fromNode];
        double toVoltage = nodeVoltages[toNode];

        // Calculate voltage difference
        double voltageDiff = fromVoltage - toVoltage;

        // Get history term if applicable
        double historyTerm = historyTerms[idx];

        // Calculate branch current based on branch type and time-domain behavior
        double current = 0.0;

        switch (branchType) {
        case 0: // Transmission line
            // Use history term for traveling wave model
            current = voltageDiff / resistance + historyTerm;
            break;

        case 1: // Transformer
        {
            double ratio = param1;  // Turns ratio
            current = (fromVoltage - toVoltage * ratio) / resistance + historyTerm;
        }
        break;

        case 2: // Inductive branch
        {
            // For inductive branch, history term represents the integration
            // of voltage over time
            current = historyTerm;
        }
        break;

        case 3: // Capacitive branch
        {
            // For capacitive branch, current is related to rate of change of voltage
            // and history term represents the previously accumulated charge
            current = susceptance * voltageDiff + historyTerm;
        }
        break;

        case 4: // Source
        {
            // For source, param1 is amplitude, param2 is frequency
            double sourceValue = param1 * sin(2.0 * M_PI * param2 * time);
            current = (fromVoltage - sourceValue) / (resistance + 1e-6);
        }
        break;

        default: // Default case - simple resistive branch
            if (resistance > 0.0) {
                current = voltageDiff / resistance;
            }
            else {
                current = 0.0;
            }
        }

        branchCurrents[idx] = current;
    }
}