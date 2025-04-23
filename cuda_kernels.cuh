#pragma once

#include <cuda_runtime.h>

// CUDA error checking
#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)
void check_cuda_error(cudaError_t result, const char* func, const char* file, int line);

// Kernel declarations
__global__ void updateHistoryTermsKernel(double* history, double* voltages, double* currents,
    int numElements, double timeStep);

__global__ void buildMatrixKernel(double* values, int* rowPtr, int* colInd,
    double* historyTerms, int numNodes);

__global__ void updateCurrentVectorKernel(double* currents, double* historyTerms,
    double time, int numNodes);

__global__ void calculateBranchCurrentsKernel(double* branchCurrents, double* nodeVoltages,
    double* branchParameters, int numBranches);

// Optional: Add more specialized kernels for different network elements
__global__ void transmissionLineKernel(double* history, double* voltages, double* currents,
    double* parameters, int numLines, double timeStep);

__global__ void transformerKernel(double* history, double* voltages, double* currents,
    double* parameters, int numTransformers, double timeStep);

// Additional CUDA kernels for specific power electronic components
__global__ void powerElectronicsKernel(double* states, double* controls, double* parameters,
    double time, double timeStep, int numConverters);

__global__ void controlSystemKernel(double* inputs, double* outputs, double* parameters,
    double time, double timeStep, int numControllers);

__global__ void thermalModelKernel(double* temperatures, double* powerLosses, double* parameters,
    double timeStep, int numThermalModels);

// Advanced matrix assembly kernels
__global__ void buildAdmittanceMatrixKernel(double* values, int* rowPtr, int* colInd,
    double* branchParameters, double* historyTerms, int numNodes, int numBranches);

__global__ void buildAdvancedMatrixKernel(double* values, int* rowPtr, int* colInd,
    double* branchParameters, double* historyTerms, double time, int numNodes, int numBranches);

// Power electronics specific kernels
__global__ void mmcCapacitorVoltageBalancingKernel(double* capacitorVoltages, double* armCurrents,
    int* switchingStates, double* switchingTimes, double time, int numSubModules, int numArms);

__global__ void windTurbineControlKernel(double* windSpeed, double* rotorSpeed,
    double* powerCommand, double* pitchAngle, double time, double timeStep, int numTurbines);

__global__ void solarPVPowerCalculationKernel(double* irradiance, double* temperature,
    double* dcVoltage, double* dcCurrent, double* parameters, int numPanels);