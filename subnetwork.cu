#include "subnetwork.h"
#include "cuda_kernels.cuh"
#include "cuda_helpers.h"
#include <iostream>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <thrust/copy.h>
#include <chrono>

Subnetwork::Subnetwork(int id, int deviceID, cudaStream_t stream, double timeStep, int numNodes)
    : id(id), deviceID(deviceID), stream(stream), timeStep(timeStep), numNodes(numNodes) {

    // Set device for this subnetwork
    CHECK_CUDA_ERROR(cudaSetDevice(deviceID));

    // Initialize cuSPARSE
    CHECK_CUSPARSE_ERROR(cusparseCreate(&cusparseHandle));
    CHECK_CUSPARSE_ERROR(cusparseSetStream(cusparseHandle, stream));

    // Initialize cuSOLVER
    CHECK_CUSOLVER_ERROR(cusolverSpCreate(&cusolverHandle));
    CHECK_CUSOLVER_ERROR(cusolverSpSetStream(cusolverHandle, stream));

    // Initialize cuBLAS
    CHECK_CUBLAS_ERROR(cublasCreate(&cublasHandle));
    CHECK_CUBLAS_ERROR(cublasSetStream(cublasHandle, stream));

    // Create matrix descriptor
    CHECK_CUSPARSE_ERROR(cusparseCreateMatDescr(&matDescr));
    CHECK_CUSPARSE_ERROR(cusparseSetMatType(matDescr, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE_ERROR(cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ZERO));

    // Allocate host and device memory
    h_voltages.resize(numNodes, 0.0);
    h_currents.resize(numNodes, 0.0);

    d_voltages.resize(numNodes, 0.0);
    d_currents.resize(numNodes, 0.0);

    buildMatrixTime = 0.0;
    solverTime = 0.0;
    iterationsPerformed = 0;
    numBranches = 0;
}

Subnetwork::~Subnetwork() {
    // Clean up CUDA resources
    cusparseDestroyMatDescr(matDescr);
    cublasDestroy(cublasHandle);
    cusolverSpDestroy(cusolverHandle);
    cusparseDestroy(cusparseHandle);
}

void Subnetwork::addNode(int globalIndex, const std::string& nodeName) {
    int localIndex = nodeMap.size();
    nodeMap[nodeName] = localIndex;
    globalNodeIndices.push_back(globalIndex);
}

void Subnetwork::addBranch(const std::string& branchName) {
    int localIndex = branchMap.size();
    branchMap[branchName] = localIndex;
    numBranches++;

    // Resize branch currents storage
    h_branchCurrents.resize(branchMap.size(), 0.0);
    d_branchCurrents.resize(branchMap.size(), 0.0);
}

void Subnetwork::addElement(NetworkElement* element) {
    elements.push_back(element);
}

int Subnetwork::getLocalNodeIndex(int globalIndex) const {
    // Find global index in our map
    for (size_t i = 0; i < globalNodeIndices.size(); i++) {
        if (globalNodeIndices[i] == globalIndex) {
            return static_cast<int>(i);
        }
    }

    // Not found
    return -1;
}

// Implementation of the getNodeMap method
const std::unordered_map<std::string, int>& Subnetwork::getNodeMap() const {
    return nodeMap;
}

// Implementation of the getBranchMap method
const std::unordered_map<std::string, int>& Subnetwork::getBranchMap() const {
    return branchMap;
}

double Subnetwork::getNodeVoltage(int localIndex) const {
    // Create a temporary variable to store the value
    double voltage = 0.0;

    // Copy directly to the temporary variable
    thrust::copy(d_voltages.begin() + localIndex, d_voltages.begin() + localIndex + 1, &voltage);

    return voltage;
}

void Subnetwork::setNodeVoltage(int localIndex, double voltage) {
    h_voltages[localIndex] = voltage;

    // Copy directly from the variable
    thrust::copy(&voltage, &voltage + 1, d_voltages.begin() + localIndex);
}

double Subnetwork::getBranchCurrent(int localIndex) const {
    // Create a temporary variable to store the value
    double current = 0.0;

    // Copy directly to the temporary variable
    thrust::copy(d_branchCurrents.begin() + localIndex, d_branchCurrents.begin() + localIndex + 1, &current);

    return current;
}

void Subnetwork::solve(double time, int iteration) {
    // Set device for this operation
    CHECK_CUDA_ERROR(cudaSetDevice(deviceID));

    // 1. Update history terms for all elements with memory (inductors, capacitors)
    updateHistoryTerms(time);

    // 2. Update the right-hand side current vector
    updateCurrentVector(time);

    // 3. Solve the linear system (Ax=b) using a GPU-accelerated sparse solver
    solveLinearSystem();

    // 4. Update branch currents
    updateBranchCurrents();

    // Increment iteration counter
    iterationsPerformed++;
}

void Subnetwork::buildAdmittanceMatrix() {
    // Start performance tracking
    auto startTime = std::chrono::high_resolution_clock::now();

    // Initialize matrix structure if not already done
    if (d_values.size() == 0 || d_rowPtr.size() == 0 || d_colInd.size() == 0) {
        initializeMatrixStructure();
    }

    // Update non-zero values based on network elements
    // In a full implementation, this would involve updating the admittance matrix
    // based on the current state of all network elements

    // Get raw pointers for CUDA operations
    double* d_valsPtr = thrust::raw_pointer_cast(d_values.data());

    // Update matrix values on device (example kernel launch)
    if (d_values.size() > 0) {
        int threadsPerBlock = 256;
        int numBlocks = (d_values.size() + threadsPerBlock - 1) / threadsPerBlock;

        // This kernel would update the admittance matrix values
        // In a real implementation, this would be a specific kernel
        // For now, just assume we're using a placeholder
        /* Example:
        updateMatrixValuesKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
            d_valsPtr,
            thrust::raw_pointer_cast(d_rowPtr.data()),
            thrust::raw_pointer_cast(d_colInd.data()),
            thrust::raw_pointer_cast(d_historyTerms.data()),
            numNodes,
            time
        );
        */
    }

    // End performance tracking
    auto endTime = std::chrono::high_resolution_clock::now();
    buildMatrixTime += std::chrono::duration<double>(endTime - startTime).count();
}

void Subnetwork::initializeMatrixStructure() {
    // This function creates the sparsity pattern for the admittance matrix
    // For a real power system, this would be based on network topology
    // Here we'll create a simple banded matrix structure

    // First, determine the number of non-zeros
    int bandWidth = 5; // Example: pentadiagonal matrix
    int nnz = 0;

    h_rowPtr.resize(numNodes + 1, 0);

    for (int i = 0; i < numNodes; i++) {
        h_rowPtr[i] = nnz;

        // Add diagonal element
        nnz++;

        // Add off-diagonal elements (within bandwidth)
        for (int j = std::max(0, i - bandWidth); j <= std::min(numNodes - 1, i + bandWidth); j++) {
            if (j != i) {
                nnz++;
            }
        }
    }
    h_rowPtr[numNodes] = nnz;

    // Allocate column indices and values
    h_colInd.resize(nnz);
    h_values.resize(nnz, 0.0);

    // Fill column indices
    int pos = 0;
    for (int i = 0; i < numNodes; i++) {
        // Add diagonal element
        h_colInd[pos++] = i;

        // Add off-diagonal elements (within bandwidth)
        for (int j = std::max(0, i - bandWidth); j <= std::min(numNodes - 1, i + bandWidth); j++) {
            if (j != i) {
                h_colInd[pos++] = j;
            }
        }
    }

    // Copy to device
    d_rowPtr = h_rowPtr;
    d_colInd = h_colInd;
    d_values.resize(nnz, 0.0);

    // Initialize history terms if not already allocated
    if (d_historyTerms.size() != numNodes) {
        d_historyTerms.resize(numNodes, 0.0);
    }
}

void Subnetwork::updateHistoryTerms(double time) {
    // In a full implementation, this would launch a CUDA kernel to update history terms
    // for all elements with memory (inductors, capacitors, transmission lines, etc.)

    // Example kernel launch:
    if (d_historyTerms.size() > 0) {
        int threadsPerBlock = 256;
        int numBlocks = (numNodes + threadsPerBlock - 1) / threadsPerBlock;
        updateHistoryTermsKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
            thrust::raw_pointer_cast(d_historyTerms.data()),
            thrust::raw_pointer_cast(d_voltages.data()),
            thrust::raw_pointer_cast(d_currents.data()),
            numNodes,
            timeStep
            );
    }
}

void Subnetwork::updateCurrentVector(double time) {
    // In a full implementation, this would launch a CUDA kernel to update the right-hand side
    // current vector based on current sources and history term contributions

    // Example kernel launch:
    int threadsPerBlock = 256;
    int numBlocks = (numNodes + threadsPerBlock - 1) / threadsPerBlock;
    updateCurrentVectorKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        thrust::raw_pointer_cast(d_currents.data()),
        thrust::raw_pointer_cast(d_historyTerms.size() > 0 ? d_historyTerms.data() : nullptr),
        time,
        numNodes
        );
}

void Subnetwork::solveLinearSystem() {
    // Start performance tracking
    auto startTime = std::chrono::high_resolution_clock::now();

    // In this implementation, we'll use cuSOLVER direct sparse solver
    // for the linear system Ax=b where:
    // - A is the admittance matrix (in CSR format)
    // - x is the unknown voltage vector
    // - b is the current vector

    if (d_values.size() > 0 && d_rowPtr.size() > 0 && d_colInd.size() > 0) {
        // Get raw pointers for CUDA calls
        double* d_valsPtr = thrust::raw_pointer_cast(d_values.data());
        int* d_rowPtrRaw = thrust::raw_pointer_cast(d_rowPtr.data());
        int* d_colIndPtr = thrust::raw_pointer_cast(d_colInd.data());
        double* d_currentsPtr = thrust::raw_pointer_cast(d_currents.data());
        double* d_voltagesPtr = thrust::raw_pointer_cast(d_voltages.data());

        // Copy currents to voltages as initial guess
        thrust::copy(d_currents.begin(), d_currents.end(), d_voltages.begin());

        // Solve using Cholesky factorization for symmetric positive definite matrices
        // Note: In a real power system, the admittance matrix is often not SPD, 
        // so LU factorization might be more appropriate

        // Create info structure for solver
        cusolverSpHandle_t handle = cusolverHandle;
        cusolverSpDcsrlsvchol(
            handle,
            numNodes,             // Number of rows
            h_rowPtr.size() - 1,  // Number of non-zeros
            matDescr,             // Matrix descriptor
            d_valsPtr,            // Matrix values
            d_rowPtrRaw,          // Row pointers (use renamed variable here)
            d_colIndPtr,          // Column indices
            d_currentsPtr,        // Right-hand side (currents)
            1e-10,                // Tolerance
            0,                    // Reordering
            d_voltagesPtr,        // Solution (voltages)
            nullptr               // Singularity (output)
        );

        // If cuSOLVER fails, fall back to a simpler method like Jacobi iteration
        /*
        int maxIter = 100;
        double tol = 1e-6;

        // Jacobi iteration
        thrust::device_vector<double> d_xNew(numNodes);
        for (int iter = 0; iter < maxIter; iter++) {
            // Copy current currents
            thrust::copy(d_currents.begin(), d_currents.end(), d_xNew.begin());

            // Subtract off-diagonal contributions
            // [This would require a custom CUDA kernel in a real implementation]

            // Check convergence
            // [This would require a norm calculation using thrust]

            // Update voltages
            thrust::copy(d_xNew.begin(), d_xNew.end(), d_voltages.begin());
        }
        */
    }
    else {
        // If matrix is not initialized, use a simplified approach
        // Here we just copy currents to voltages as a placeholder
        thrust::copy(d_currents.begin(), d_currents.end(), d_voltages.begin());
    }

    // End performance tracking
    auto endTime = std::chrono::high_resolution_clock::now();
    solverTime += std::chrono::duration<double>(endTime - startTime).count();
}

void Subnetwork::updateBranchCurrents() {
    // In a full implementation, this would launch a CUDA kernel to calculate branch currents
    // based on node voltages

    if (numBranches > 0 && d_branchParameters.size() > 0) {
        // Example kernel launch:
        int threadsPerBlock = 256;
        int numBlocks = (branchMap.size() + threadsPerBlock - 1) / threadsPerBlock;
        calculateBranchCurrentsKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
            thrust::raw_pointer_cast(d_branchCurrents.data()),
            thrust::raw_pointer_cast(d_voltages.data()),
            thrust::raw_pointer_cast(d_branchParameters.data()),
            branchMap.size()
            );
    }
}


void Subnetwork::debugMatrixAndVectors(const std::string& phase) {
    // Debug function to print matrix and vectors
    std::cout << "Debug [" << phase << "] for Subnetwork " << id << std::endl;
    // Add debugging code here
}