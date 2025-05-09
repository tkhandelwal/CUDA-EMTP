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
    std::cout << "[" << getCurrentTimeString() << "] SOLVER: Starting linear system solution for subnetwork " << id << std::endl;
    std::cout << "[" << getCurrentTimeString() << "] SOLVER: Matrix size = " << numNodes << "x" << numNodes
        << " with " << h_values.size() << " non-zeros" << std::endl;

    // Set CUDA device
    cudaError_t err = cudaSetDevice(deviceID);
    if (err != cudaSuccess) {
        std::cerr << "[" << getCurrentTimeString() << "] SOLVER ERROR: Failed to set CUDA device "
            << deviceID << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA device selection failed in solver");
    }

    // Check matrix structure validity
    if (h_rowPtr.size() != numNodes + 1) {
        std::cerr << "[" << getCurrentTimeString() << "] SOLVER ERROR: Invalid row pointer size. Expected "
            << (numNodes + 1) << ", got " << h_rowPtr.size() << std::endl;

        // If this happens, try to fix it
        std::cout << "[" << getCurrentTimeString() << "] SOLVER: Reinitializing matrix structure" << std::endl;
        initializeMatrixStructure();
        return;  // Exit and try again in the next iteration
    }

    // Verify values size matches non-zero count
    if (h_colInd.size() != h_values.size()) {
        std::cerr << "[" << getCurrentTimeString() << "] SOLVER ERROR: Column indices and values size mismatch: "
            << h_colInd.size() << " vs " << h_values.size() << std::endl;
        return;  // Exit and try again in the next iteration
    }

    // Prepare device pointers with error checking
    double* d_valuesPtr = nullptr;
    int* d_rowPtrPtr = nullptr;
    int* d_colIndPtr = nullptr;
    double* d_currentsPtr = nullptr;
    double* d_voltagesPtr = nullptr;

    try {
        d_valuesPtr = thrust::raw_pointer_cast(d_values.data());
        d_rowPtrPtr = thrust::raw_pointer_cast(d_rowPtr.data());
        d_colIndPtr = thrust::raw_pointer_cast(d_colInd.data());
        d_currentsPtr = thrust::raw_pointer_cast(d_currents.data());
        d_voltagesPtr = thrust::raw_pointer_cast(d_voltages.data());
    }
    catch (const std::exception& e) {
        std::cerr << "[" << getCurrentTimeString() << "] SOLVER ERROR: Failed to get raw pointers: "
            << e.what() << std::endl;
        return;
    }

    // Sanity check device pointers
    if (!d_valuesPtr || !d_rowPtrPtr || !d_colIndPtr || !d_currentsPtr || !d_voltagesPtr) {
        std::cerr << "[" << getCurrentTimeString() << "] SOLVER ERROR: One or more device pointers are null" << std::endl;
        return;
    }

    // Debug: Log the current vector values
    std::cout << "[" << getCurrentTimeString() << "] SOLVER: Current vector (first 5 elements): ";
    for (int i = 0; i < std::min(5, numNodes); i++) {
        std::cout << h_currents[i] << " ";
    }
    std::cout << std::endl;

    // Check for NaN or Inf in current vector
    bool has_bad_value = false;
    for (int i = 0; i < numNodes; i++) {
        if (std::isnan(h_currents[i]) || std::isinf(h_currents[i])) {
            std::cerr << "[" << getCurrentTimeString() << "] SOLVER ERROR: Invalid value in current vector at index "
                << i << ": " << h_currents[i] << std::endl;
            has_bad_value = true;
            h_currents[i] = 0.0;  // Replace with zero
        }
    }

    if (has_bad_value) {
        std::cout << "[" << getCurrentTimeString() << "] SOLVER: Fixed invalid values in current vector" << std::endl;
        d_currents = h_currents;  // Update device copy
        d_currentsPtr = thrust::raw_pointer_cast(d_currents.data());
    }

    // Copy matrix to host for inspection and modification
    std::cout << "[" << getCurrentTimeString() << "] SOLVER: Copying matrix for diagonal boosting" << std::endl;
    thrust::host_vector<double> h_values_copy = d_values;

    // Add small values to diagonal for stability
    int diagonalBoostCount = 0;
    for (int i = 0; i < numNodes; i++) {
        bool foundDiagonal = false;
        // Find diagonal entry
        for (int j = h_rowPtr[i]; j < h_rowPtr[i + 1]; j++) {
            if (h_colInd[j] == i) {
                // Add small value to diagonal for stability
                h_values_copy[j] += 0.0001;
                diagonalBoostCount++;
                foundDiagonal = true;
                break;
            }
        }

        // If diagonal not found, report error
        if (!foundDiagonal) {
            std::cerr << "[" << getCurrentTimeString() << "] SOLVER ERROR: Diagonal element missing for row "
                << i << std::endl;
        }
    }

    std::cout << "[" << getCurrentTimeString() << "] SOLVER: Boosted " << diagonalBoostCount
        << " diagonal elements by 0.0001" << std::endl;

    // Copy boosted values back to device
    d_values = h_values_copy;
    d_valuesPtr = thrust::raw_pointer_cast(d_values.data());

    // Ensure d_info is allocated
    if (!d_info) {
        std::cout << "[" << getCurrentTimeString() << "] SOLVER: Allocating d_info" << std::endl;
        cudaMalloc(&d_info, sizeof(int));
    }

    // Initialize voltage vector with zeros or previous solution
    std::cout << "[" << getCurrentTimeString() << "] SOLVER: Initializing voltage vector" << std::endl;
    if (iterationsPerformed == 0 && time_step_index == 0) {
        // First iteration of first time step - use zeros
        thrust::fill(d_voltages.begin(), d_voltages.end(), 0.0);
    }
    else {
        // Use previous values as initial guess
        d_voltages = h_voltages;
    }

    // Try QR factorization instead of Cholesky
    std::cout << "[" << getCurrentTimeString() << "] SOLVER: Using QR factorization" << std::endl;

    // Step 1: Calculate buffer sizes
    size_t internalDataInBytes = 0;
    size_t workspaceInBytes = 0;

    cusolverStatus_t status = cusolverSpDcsrlsvqr_bufferSize(
        cusolverHandle, numNodes, h_values.size(),
        matDescr, d_valuesPtr, d_rowPtrPtr, d_colIndPtr,
        d_currentsPtr, 0.0001, // Tolerance
        1, // Reorder
        d_voltagesPtr,
        &internalDataInBytes, &workspaceInBytes);

    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "[" << getCurrentTimeString() << "] SOLVER ERROR: Buffer size calculation failed, status = "
            << status << std::endl;
        // Use previous voltages as fallback
        std::cout << "[" << getCurrentTimeString() << "] SOLVER: Using previous voltages as fallback" << std::endl;
        h_voltages = prev_voltages;
        d_voltages = h_voltages;
        return;
    }

    std::cout << "[" << getCurrentTimeString() << "] SOLVER: Buffer sizes: internal = "
        << internalDataInBytes << ", workspace = " << workspaceInBytes << std::endl;

    // Allocate workspace
    void* buffer = nullptr;
    if (workspaceInBytes > 0) {
        err = cudaMalloc(&buffer, workspaceInBytes);
        if (err != cudaSuccess) {
            std::cerr << "[" << getCurrentTimeString() << "] SOLVER ERROR: Workspace allocation failed: "
                << cudaGetErrorString(err) << std::endl;
            // Use previous voltages as fallback
            h_voltages = prev_voltages;
            d_voltages = h_voltages;
            return;
        }
        std::cout << "[" << getCurrentTimeString() << "] SOLVER: Allocated workspace buffer of "
            << workspaceInBytes << " bytes" << std::endl;
    }

    // Step 2: Solve the system using QR factorization
    std::cout << "[" << getCurrentTimeString() << "] SOLVER: Calling cusolverSpDcsrlsvqr" << std::endl;
    status = cusolverSpDcsrlsvqr(
        cusolverHandle, numNodes, h_values.size(),
        matDescr, d_valuesPtr, d_rowPtrPtr, d_colIndPtr,
        d_currentsPtr, 0.0001, // Tolerance
        1, // Reorder
        d_voltagesPtr,
        d_info, buffer);

    // Check solver status
    std::cout << "[" << getCurrentTimeString() << "] SOLVER: QR solver status = " << status << std::endl;

    // Check for solve errors
    int h_info = 0;
    err = cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "[" << getCurrentTimeString() << "] SOLVER ERROR: Failed to get solver info: "
            << cudaGetErrorString(err) << std::endl;
        h_info = -1;  // Assume error
    }

    std::cout << "[" << getCurrentTimeString() << "] SOLVER: Solver info = " << h_info << std::endl;

    if (status != CUSOLVER_STATUS_SUCCESS || h_info != 0) {
        std::cerr << "[" << getCurrentTimeString() << "] SOLVER ERROR: QR solver failed with status = "
            << status << ", info = " << h_info << std::endl;

        // Try LU solver as fallback
        std::cout << "[" << getCurrentTimeString() << "] SOLVER: Trying LU solver as fallback" << std::endl;

        // Ensure d_info is reset
        int resetInfo = 0;
        cudaMemcpy(d_info, &resetInfo, sizeof(int), cudaMemcpyHostToDevice);

        // Call LU solver
        status = cusolverSpDcsrlsvlu(
            cusolverHandle, numNodes, h_values.size(),
            matDescr, d_valuesPtr, d_rowPtrPtr, d_colIndPtr,
            d_currentsPtr, 0.0001, // Tolerance
            1, // Reorder
            d_voltagesPtr,
            d_info, buffer);

        std::cout << "[" << getCurrentTimeString() << "] SOLVER: LU solver status = " << status << std::endl;

        err = cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "[" << getCurrentTimeString() << "] SOLVER ERROR: Failed to get LU solver info: "
                << cudaGetErrorString(err) << std::endl;
            h_info = -1;
        }

        std::cout << "[" << getCurrentTimeString() << "] SOLVER: LU solver info = " << h_info << std::endl;

        if (status != CUSOLVER_STATUS_SUCCESS || h_info != 0) {
            std::cerr << "[" << getCurrentTimeString() << "] SOLVER ERROR: Both QR and LU solvers failed" << std::endl;

            // Use previous values as final fallback
            std::cout << "[" << getCurrentTimeString() << "] SOLVER: Using previous values as final fallback" << std::endl;
            h_voltages = prev_voltages;
            d_voltages = h_voltages;
        }
        else {
            // LU solve succeeded
            std::cout << "[" << getCurrentTimeString() << "] SOLVER: LU solve succeeded" << std::endl;
            try {
                h_voltages = d_voltages;
            }
            catch (const std::exception& e) {
                std::cerr << "[" << getCurrentTimeString() << "] SOLVER ERROR: Failed to copy voltage result: "
                    << e.what() << std::endl;
                h_voltages = prev_voltages;
            }
        }
    }
    else {
        // QR solve succeeded
        std::cout << "[" << getCurrentTimeString() << "] SOLVER: QR solve succeeded" << std::endl;
        try {
            h_voltages = d_voltages;
        }
        catch (const std::exception& e) {
            std::cerr << "[" << getCurrentTimeString() << "] SOLVER ERROR: Failed to copy voltage result: "
                << e.what() << std::endl;
            h_voltages = prev_voltages;
        }
    }

    // Free workspace
    if (buffer) {
        cudaFree(buffer);
    }

    // Debug: Log the voltage vector values
    std::cout << "[" << getCurrentTimeString() << "] SOLVER: Voltage vector (first 5 elements): ";
    for (int i = 0; i < std::min(5, numNodes); i++) {
        std::cout << h_voltages[i] << " ";
    }
    std::cout << std::endl;

    // Check for NaN or Inf in voltage vector
    has_bad_value = false;
    for (int i = 0; i < numNodes; i++) {
        if (std::isnan(h_voltages[i]) || std::isinf(h_voltages[i]) || std::abs(h_voltages[i]) > 1e6) {
            std::cerr << "[" << getCurrentTimeString() << "] SOLVER ERROR: Invalid value in voltage vector at index "
                << i << ": " << h_voltages[i] << std::endl;
            has_bad_value = true;
            // Replace with previous value
            h_voltages[i] = prev_voltages[i];
        }
    }

    if (has_bad_value) {
        std::cout << "[" << getCurrentTimeString() << "] SOLVER: Fixed invalid values in voltage vector" << std::endl;
    }

    // Apply relaxation to improve convergence
    if (iterationsPerformed > 0) {
        // Adjusting relaxation factor based on iterations
        double current_relaxation = relaxation_factor;
        if (iterationsPerformed > 5) {
            // More conservative relaxation for slow convergence
            current_relaxation = std::max(0.5, relaxation_factor - 0.1);
        }

        std::cout << "[" << getCurrentTimeString() << "] SOLVER: Applying relaxation factor "
            << current_relaxation << std::endl;

        // Apply relaxation: v_new = alpha * v_calc + (1-alpha) * v_old
        for (int i = 0; i < numNodes; i++) {
            h_voltages[i] = current_relaxation * h_voltages[i] +
                (1.0 - current_relaxation) * previous_iteration_voltages[i];
        }
    }

    // Copy final voltages back to device
    try {
        d_voltages = h_voltages;
    }
    catch (const std::exception& e) {
        std::cerr << "[" << getCurrentTimeString() << "] SOLVER ERROR: Failed to update device voltages: "
            << e.what() << std::endl;
    }

    std::cout << "[" << getCurrentTimeString() << "] SOLVER: Linear system solution completed for subnetwork "
        << id << std::endl;
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