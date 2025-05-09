#include "subnetwork.h"
#include "cuda_kernels.cuh"
#include "cuda_helpers.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <ctime>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <thrust/copy.h>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

    // Initialize d_info to nullptr
    d_info = nullptr;
}

Subnetwork::~Subnetwork() {
    // Clean up CUDA resources
    cusparseDestroyMatDescr(matDescr);
    cublasDestroy(cublasHandle);
    cusolverSpDestroy(cusolverHandle);
    cusparseDestroy(cusparseHandle);

    // Free d_info if allocated
    if (d_info) {
        cudaFree(d_info);
        d_info = nullptr;
    }
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
    // Create stream for detailed logging
    std::ofstream logStream("subnetwork_" + std::to_string(id) + "_log.txt",
        std::ios::app);

    auto logMessage = [&](const std::string& message) {
        // Simple timestamp without put_time
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        char timeStr[20];
        std::strftime(timeStr, sizeof(timeStr), "%H:%M:%S", std::localtime(&now_c));

        std::string fullMessage = std::string("[") + timeStr + "] Subnetwork " +
            std::to_string(id) + " (t=" + std::to_string(time) +
            ", iter=" + std::to_string(iteration) + "): " + message;
        std::cout << fullMessage << std::endl;
        logStream << fullMessage << std::endl;
        };

    // Check CUDA device
    cudaError_t err;
    int currentDevice;
    err = cudaGetDevice(&currentDevice);
    if (err != cudaSuccess) {
        logMessage("ERROR getting current device: " + std::string(cudaGetErrorString(err)));
        throw std::runtime_error("CUDA device query failed");
    }

    if (currentDevice != deviceID) {
        logMessage("WARNING: Current device " + std::to_string(currentDevice) +
            " doesn't match expected device " + std::to_string(deviceID));
    }

    // Set device for this operation
    logMessage("Setting CUDA device to " + std::to_string(deviceID));
    err = cudaSetDevice(deviceID);
    if (err != cudaSuccess) {
        logMessage("ERROR setting device: " + std::string(cudaGetErrorString(err)));
        throw std::runtime_error("CUDA device selection failed");
    }

    // For first iteration of a new time step, improve initial guess with extrapolation
    if (iteration == 0) {
        logMessage("First iteration of time step, checking for extrapolation");

        // Initialize vectors if needed
        if (prev_voltages.size() != numNodes) {
            logMessage("Initializing previous voltage vectors");
            prev_voltages.resize(numNodes, 0.0);
            prev_prev_voltages.resize(numNodes, 0.0);

            // Copy current voltages for initial state
            logMessage("Copying initial voltages");
            thrust::copy(d_voltages.begin(), d_voltages.end(), prev_voltages.begin());
            thrust::copy(d_voltages.begin(), d_voltages.end(), prev_prev_voltages.begin());
        }

        // For the first few time steps, we can't extrapolate
        if (time_step_index >= 2) {
            logMessage("Performing extrapolation for time step " + std::to_string(time_step_index));

            // Copy current voltages to host for manipulation
            thrust::host_vector<double> current_voltages = d_voltages;

            // Linear extrapolation: 2*v(n) - v(n-1)
            for (int i = 0; i < numNodes; i++) {
                double extrapolated = 2.0 * prev_voltages[i] - prev_prev_voltages[i];
                current_voltages[i] = extrapolated;
            }

            // Copy improved initial guess back to device
            thrust::copy(current_voltages.begin(), current_voltages.end(), d_voltages.begin());

            if (time_step_index % 100 == 0) {
                logMessage("Applied extrapolation at time step " + std::to_string(time_step_index));
            }
        }
    }

    // 1. Update history terms for all elements with memory (inductors, capacitors)
    logMessage("Updating history terms");
    try {
        updateHistoryTerms(time);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            logMessage("CUDA error after updateHistoryTerms: " + std::string(cudaGetErrorString(err)));
            throw std::runtime_error("CUDA error in updateHistoryTerms");
        }
    }
    catch (const std::exception& e) {
        logMessage("Exception in updateHistoryTerms: " + std::string(e.what()));
        throw;
    }

    // 2. Update the right-hand side current vector
    logMessage("Updating current vector");
    try {
        updateCurrentVector(time);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            logMessage("CUDA error after updateCurrentVector: " + std::string(cudaGetErrorString(err)));
            throw std::runtime_error("CUDA error in updateCurrentVector");
        }
    }
    catch (const std::exception& e) {
        logMessage("Exception in updateCurrentVector: " + std::string(e.what()));
        throw;
    }

    // 3. Solve the linear system (Ax=b) using a GPU-accelerated sparse solver
    logMessage("Solving linear system");
    try {
        solveLinearSystem();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            logMessage("CUDA error after solveLinearSystem: " + std::string(cudaGetErrorString(err)));
            throw std::runtime_error("CUDA error in solveLinearSystem");
        }
    }
    catch (const std::exception& e) {
        logMessage("Exception in solveLinearSystem: " + std::string(e.what()));
        throw;
    }

    // 4. Update branch currents
    logMessage("Updating branch currents");
    try {
        updateBranchCurrents();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            logMessage("CUDA error after updateBranchCurrents: " + std::string(cudaGetErrorString(err)));
            throw std::runtime_error("CUDA error in updateBranchCurrents");
        }
    }
    catch (const std::exception& e) {
        logMessage("Exception in updateBranchCurrents: " + std::string(e.what()));
        throw;
    }

    // After all iterations for this time step are done, store voltages for next time step
    if (iteration == 0) {
        logMessage("Storing voltages for next time step");

        // Store voltages for next time
        prev_prev_voltages = prev_voltages;

        // Copy current device voltages to host
        thrust::copy(d_voltages.begin(), d_voltages.end(), prev_voltages.begin());

        // Increment time step counter
        time_step_index++;
    }

    // Increment iteration counter
    iterationsPerformed++;

    logMessage("Solve completed successfully");
    logStream.close();
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
    std::cout << "Updating current vector at time " << time << " for subnetwork " << id << std::endl;

    // Input validation - critical for preventing CUDA errors
    if (numNodes <= 0) {
        std::cerr << "Error: Cannot launch kernel with numNodes = " << numNodes << std::endl;
        return;
    }

    // Check that memory is allocated
    if (d_currents.size() < numNodes) {
        std::cerr << "Error: CUDA currents vector not properly sized (size: "
            << d_currents.size() << ", required: " << numNodes << ")" << std::endl;
        return;
    }

    // Get device properties to check maximum block size
    cudaDeviceProp deviceProp;
    cudaError_t propErr = cudaGetDeviceProperties(&deviceProp, deviceID);
    if (propErr != cudaSuccess) {
        std::cerr << "Error getting device properties: " << cudaGetErrorString(propErr) << std::endl;
        return;
    }

    // Reset the current vector first
    thrust::fill(d_currents.begin(), d_currents.end(), 0.0);

    // Choose calculation method:
    // Option 1: Set values on host and copy to device
    bool useHostCalculation = true;  // Set to false to test kernel approach instead

    if (useHostCalculation) {
        // Create and initialize host vector
        thrust::host_vector<double> h_currents_local(numNodes, 0.0);

        // Add three-phase current injections with proper phase shifts
        double baseAmplitude = 1.0;  // 1 A (adjust as needed)
        double frequency = 60.0;     // 60 Hz

        // Generate 3-phase balanced set of currents (with 120° phase shifts)
        int phaseCount = std::min(3, numNodes);
        for (int i = 0; i < phaseCount; i++) {
            double phaseShift = i * 2.0 * M_PI / 3.0;  // 0°, 120°, 240° for 3-phase
            h_currents_local[i] = baseAmplitude * sin(2.0 * M_PI * frequency * time + phaseShift);

            // Debug output to verify three-phase currents
            std::cout << "Phase " << i << " current: " << h_currents_local[i]
                << " A (phase shift: " << (phaseShift * 180.0 / M_PI) << "°)" << std::endl;
        }

        // Copy the values back to device
        thrust::copy(h_currents_local.begin(), h_currents_local.end(), d_currents.begin());

        // Debug verification - copy back to host and print
        thrust::host_vector<double> verification = d_currents;
        std::cout << "Verification of first 3 currents after copy: ";
        for (int i = 0; i < std::min(3, numNodes); i++) {
            std::cout << verification[i] << " ";
        }
        std::cout << std::endl;
    }
    // Option 2: Use CUDA kernel for calculation
    else {
        // Safety checks for kernel launch
        if (d_historyTerms.size() < numNodes && numNodes > 0) {
            // Resize history terms if needed
            std::cout << "Resizing history terms vector to match numNodes" << std::endl;
            d_historyTerms.resize(numNodes, 0.0);
        }

        // Use a safe block size based on device capabilities
        int threadsPerBlock = std::min(256, deviceProp.maxThreadsPerBlock);
        int numBlocks = (numNodes + threadsPerBlock - 1) / threadsPerBlock;

        // Ensure at least one block for valid launch
        numBlocks = std::max(1, numBlocks);

        // Log kernel launch parameters
        std::cout << "Launching updateCurrentVectorKernel with:"
            << " numNodes=" << numNodes
            << " threadsPerBlock=" << threadsPerBlock
            << " numBlocks=" << numBlocks
            << std::endl;

        // Raw pointers for kernel
        double* d_currents_ptr = thrust::raw_pointer_cast(d_currents.data());
        double* d_historyTerms_ptr = (d_historyTerms.size() > 0) ?
            thrust::raw_pointer_cast(d_historyTerms.data()) : nullptr;

        // Launch the kernel with explicit error checking
        updateCurrentVectorKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
            d_currents_ptr,
            d_historyTerms_ptr,
            time,
            numNodes
            );

        // Check for launch errors
        cudaError_t kernelErr = cudaGetLastError();
        if (kernelErr != cudaSuccess) {
            std::cerr << "CUDA error in updateCurrentVectorKernel: "
                << cudaGetErrorString(kernelErr) << std::endl;
            throw std::runtime_error("CUDA error in updateCurrentVector");
        }

        // Synchronize to ensure completion (helps with debugging)
        cudaError_t syncErr = cudaStreamSynchronize(stream);
        if (syncErr != cudaSuccess) {
            std::cerr << "CUDA stream synchronization error: "
                << cudaGetErrorString(syncErr) << std::endl;
        }
    }

    std::cout << "Current vector update completed for subnetwork " << id << std::endl;
}

void Subnetwork::solveLinearSystem() {
    // Create stream for detailed logging
    std::ofstream logStream("subnetwork_" + std::to_string(id) + "_solver_log.txt",
        std::ios::app);

    auto logMessage = [&](const std::string& message) {
        // Simple timestamp without put_time
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        char timeStr[20];
        std::strftime(timeStr, sizeof(timeStr), "%H:%M:%S", std::localtime(&now_c));

        std::string fullMessage = std::string("[") + timeStr + "] Function: " + message;
        std::cout << fullMessage << std::endl;
        logStream << fullMessage << std::endl;
        };

    // Start performance tracking
    auto startTime = std::chrono::high_resolution_clock::now();
    logMessage("Starting linear system solution");

    if (d_values.size() > 0 && d_rowPtr.size() > 0 && d_colInd.size() > 0) {
        logMessage("Matrix initialized with " + std::to_string(d_values.size()) + " non-zeros");

        // Get raw pointers for CUDA calls
        double* d_valsPtr = thrust::raw_pointer_cast(d_values.data());
        int* d_rowPtrRaw = thrust::raw_pointer_cast(d_rowPtr.data());
        int* d_colIndPtr = thrust::raw_pointer_cast(d_colInd.data());
        double* d_currentsPtr = thrust::raw_pointer_cast(d_currents.data());
        double* d_voltagesPtr = thrust::raw_pointer_cast(d_voltages.data());

        // Copy currents to host for inspection
        logMessage("Copying currents to host for inspection");
        thrust::copy(d_currents.begin(), d_currents.end(), h_currents.begin());

        // Check if the current vector contains all zeros or very small values
        bool allZerosOrSmall = true;
        double smallThreshold = 1e-10;
        double maxCurrent = 0.0;
        double minCurrent = 0.0;

        for (const auto& current : h_currents) {
            if (std::abs(current) > smallThreshold) {
                allZerosOrSmall = false;
            }
            maxCurrent = std::max(maxCurrent, current);
            minCurrent = std::min(minCurrent, current);
        }

        logMessage("Current vector stats: min=" + std::to_string(minCurrent) +
            ", max=" + std::to_string(maxCurrent));

        if (allZerosOrSmall) {
            // If all currents are zero or very small, set a small non-zero value
            // to at least one element to avoid singular system
            double smallInjection = 1e-6;
            logMessage("WARNING: All currents near zero, adding small injection of " +
                std::to_string(smallInjection) + " to current[0]");
            h_currents[0] = smallInjection;
            thrust::copy(h_currents.begin(), h_currents.end(), d_currents.begin());
        }

        // Create info structure for solver
        cusolverSpHandle_t handle = cusolverHandle;
        logMessage("Using cuSolver handle " + std::to_string((uint64_t)handle));

        // Add error handling for solver
        int singularity = 0;

        // Allocate d_info if not already allocated
        if (!d_info) {
            logMessage("Allocating d_info");
            cudaError_t err = cudaMalloc(&d_info, sizeof(int));
            if (err != cudaSuccess) {
                logMessage("CUDA error allocating d_info: " + std::string(cudaGetErrorString(err)));
                throw std::runtime_error("CUDA memory allocation failed");
            }
        }

        // Try with diagonal boosting for better conditioning
        // Copy matrix values to host for manipulation
        logMessage("Copying matrix values to host for diagonal boosting");
        thrust::host_vector<double> h_vals_copy = d_values;

        // Boost diagonal entries for better conditioning
        double diagonalBoostValue = 1e-4; // Increased from 1e-6 for better conditioning
        int diagonalCount = 0;

        for (int i = 0; i < numNodes; i++) {
            bool foundDiagonal = false;
            for (int j = h_rowPtr[i]; j < h_rowPtr[i + 1]; j++) {
                if (h_colInd[j] == i) {
                    // Add small value to diagonal to improve conditioning
                    h_vals_copy[j] += diagonalBoostValue;
                    foundDiagonal = true;
                    diagonalCount++;
                    break;
                }
            }

            if (!foundDiagonal) {
                logMessage("WARNING: No diagonal element found for row " + std::to_string(i));
            }
        }

        logMessage("Boosted " + std::to_string(diagonalCount) + " diagonal elements by " +
            std::to_string(diagonalBoostValue));

        // Copy boosted matrix back to device
        logMessage("Copying boosted matrix back to device");
        thrust::copy(h_vals_copy.begin(), h_vals_copy.end(), d_values.begin());

        // Try Cholesky factorization with improved matrix
        logMessage("Calling cusolverSpDcsrlsvchol");
        cusolverStatus_t status;
        try {
            status = cusolverSpDcsrlsvchol(
                handle,
                numNodes,             // Number of rows
                h_rowPtr.size() - 1,  // Number of non-zeros
                matDescr,             // Matrix descriptor
                d_valsPtr,            // Matrix values
                d_rowPtrRaw,          // Row pointers
                d_colIndPtr,          // Column indices
                d_currentsPtr,        // Right-hand side (currents)
                1e-10,                // Tolerance
                0,                    // Reordering
                d_voltagesPtr,        // Solution (voltages)
                d_info                // Singularity (output)
            );
        }
        catch (const std::exception& e) {
            logMessage("Exception during solver call: " + std::string(e.what()));
            throw;
        }

        // Check for CUDA errors
        cudaError_t cudaErr = cudaGetLastError();
        if (cudaErr != cudaSuccess) {
            logMessage("CUDA error after solver: " + std::string(cudaGetErrorString(cudaErr)));
        }

        // Check for errors
        if (status != CUSOLVER_STATUS_SUCCESS) {
            logMessage("WARNING: cuSolver error, status: " + std::to_string(status));

            // Fallback to iterative solver (QR) for more stability
            logMessage("Falling back to simple diagonal solver");

            // Use QMR or GMRES solver instead (this would need actual implementation)
            // For now, we'll do a simple diagonal scaling fallback
            for (int i = 0; i < numNodes; i++) {
                // Extract diagonal element for scaling
                double diag = 1.0; // Default
                for (int j = h_rowPtr[i]; j < h_rowPtr[i + 1]; j++) {
                    if (h_colInd[j] == i) {
                        diag = h_vals_copy[j];
                        break;
                    }
                }

                // Simple scaling (on host)
                if (std::abs(diag) > 1e-10) {
                    h_voltages[i] = h_currents[i] / diag;
                }
                else {
                    h_voltages[i] = h_currents[i] * 0.001;  // 1000 ohm impedance as fallback
                }
            }

            // Copy back to device
            logMessage("Copying fallback solution to device");
            thrust::copy(h_voltages.begin(), h_voltages.end(), d_voltages.begin());
        }
        else {
            // Check for singularity
            int h_singularity;
            cudaMemcpy(&h_singularity, d_info, sizeof(int), cudaMemcpyDeviceToHost);
            if (h_singularity != -1) {
                logMessage("WARNING: Singular matrix detected at row: " + std::to_string(h_singularity));

                // Apply fallback solution at singularity point
                h_voltages[h_singularity] = 0.0; // Set to ground or a reasonable value

                // Copy back to device
                logMessage("Copying fixed solution to device after handling singularity");
                thrust::copy(h_voltages.begin(), h_voltages.end(), d_voltages.begin());
            }
            else {
                // Copy results from device to host for checking/limiting
                logMessage("Copying solution from device to host for validation");
                thrust::copy(d_voltages.begin(), d_voltages.end(), h_voltages.begin());

                // Log voltage statistics
                double minVoltage = std::numeric_limits<double>::max();
                double maxVoltage = std::numeric_limits<double>::lowest();
                double sumVoltage = 0.0;

                for (int i = 0; i < numNodes; i++) {
                    minVoltage = std::min(minVoltage, h_voltages[i]);
                    maxVoltage = std::max(maxVoltage, h_voltages[i]);
                    sumVoltage += h_voltages[i];
                }

                logMessage("Voltage stats: min=" + std::to_string(minVoltage) +
                    ", max=" + std::to_string(maxVoltage) +
                    ", avg=" + std::to_string(sumVoltage / numNodes));
            }
        }

        // Check for unrealistic voltage values and limit them
        bool voltageWasLimited = false;
        double maxVoltage = 1000.0; // 1000 kV as an upper limit
        int limitedCount = 0;

        for (int i = 0; i < numNodes; i++) {
            // Get the current voltage
            double voltage = h_voltages[i];

            // Perform sanity check on voltage value
            if (std::isnan(voltage) || std::isinf(voltage) || std::abs(voltage) > maxVoltage) {
                logMessage("WARNING: Invalid voltage value " + std::to_string(voltage) +
                    " at node " + std::to_string(i));

                // Use previous value or zero instead
                h_voltages[i] = (voltage > 0) ? maxVoltage : -maxVoltage;
                if (std::isnan(voltage) || std::isinf(voltage)) {
                    h_voltages[i] = 0.0; // Reset to zero for NaN or infinity
                }
                voltageWasLimited = true;
                limitedCount++;
            }
        }

        if (voltageWasLimited) {
            logMessage("WARNING: Limited " + std::to_string(limitedCount) +
                " excessive voltage values");

            // Copy back to device
            logMessage("Copying limited voltage values back to device");
            thrust::copy(h_voltages.begin(), h_voltages.end(), d_voltages.begin());
        }
    }
    else {
        // If matrix is not initialized, use a simplified approach
        logMessage("WARNING: Matrix not initialized, using direct copy approach");
        // Here we just copy currents to voltages as a placeholder
        thrust::copy(d_currents.begin(), d_currents.end(), d_voltages.begin());
    }

    // End performance tracking
    auto endTime = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(endTime - startTime).count();
    solverTime += duration;

    logMessage("Linear system solution completed in " + std::to_string(duration) + " seconds");
    logStream.close();
}

void Subnetwork::updateBranchCurrents() {
    // Enhanced function with proper current calculation
    if (numBranches > 0) {
        // Copy voltages to host for processing
        thrust::host_vector<double> h_voltages_local = d_voltages;

        // Resize currents vector if needed
        if (h_branchCurrents.size() != branchMap.size()) {
            h_branchCurrents.resize(branchMap.size(), 0.0);
            d_branchCurrents.resize(branchMap.size(), 0.0);
        }

        // First, calculate branch currents on host
        for (const auto& branch_pair : branchMap) {
            std::string branchName = branch_pair.first;
            int branchIdx = branch_pair.second;

            // Parse branch information to find connected nodes
            // Assuming branch name format is "TYPE_FROMNODE_TONODE"
            size_t first_underscore = branchName.find('_');
            size_t last_underscore = branchName.rfind('_');

            if (first_underscore != std::string::npos && last_underscore != std::string::npos && first_underscore != last_underscore) {
                std::string fromNodeName = branchName.substr(first_underscore + 1, last_underscore - first_underscore - 1);
                std::string toNodeName = branchName.substr(last_underscore + 1);

                // Find node indices
                auto fromNodeIt = nodeMap.find(fromNodeName);
                auto toNodeIt = nodeMap.find(toNodeName);

                if (fromNodeIt != nodeMap.end() && toNodeIt != nodeMap.end()) {
                    int fromNodeIdx = fromNodeIt->second;
                    int toNodeIdx = toNodeIt->second;

                    // Get voltages
                    double v1 = h_voltages_local[fromNodeIdx];
                    double v2 = h_voltages_local[toNodeIdx];

                    // Calculate current based on voltage difference and impedance
                    // Assuming a default impedance if not available
                    double impedance = 10.0; // Default impedance in ohms

                    // For transmission lines, use distance-based impedance
                    if (branchName.substr(0, 4) == "LINE") {
                        // Approximate impedance based on length (assumed in name format LINE_X_Y_Z where Z is length)
                        impedance = 0.5; // Base impedance per unit length
                    }
                    else if (branchName.substr(0, 11) == "TRANSFORMER") {
                        // Higher impedance for transformers
                        impedance = 50.0;
                    }

                    // Calculate current: I = (V1 - V2) / Z
                    double current = (v1 - v2) / impedance;

                    // Store current
                    h_branchCurrents[branchIdx] = current;
                }
            }
        }

        // Copy calculated currents back to device
        thrust::copy(h_branchCurrents.begin(), h_branchCurrents.end(), d_branchCurrents.begin());
    }
    else {
        // If no branches defined, launch the kernel with minimal work
        int threadsPerBlock = 256;
        int numBlocks = 1;
        calculateBranchCurrentsKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
            thrust::raw_pointer_cast(d_branchCurrents.data()),
            thrust::raw_pointer_cast(d_voltages.data()),
            thrust::raw_pointer_cast(d_branchParameters.size() > 0 ? d_branchParameters.data() : nullptr),
            0
            );
    }
}


void Subnetwork::debugMatrixAndVectors(const std::string& phase) {
    // Debug function to print matrix and vectors
    std::cout << "Debug [" << phase << "] for Subnetwork " << id << std::endl;
    // Add debugging code here
}