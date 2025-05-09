#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// Forward declarations
class NetworkElement;

/**
 * @brief Represents a portion of the network that runs on a single GPU
 */
class Subnetwork {
private:
    int id;
    int deviceID;
    int* d_info;  // Device memory for solver info/error codes
    cudaStream_t stream;
    double timeStep;

    // Previous two sets of voltages for extrapolation
    thrust::host_vector<double> prev_voltages;
    thrust::host_vector<double> prev_prev_voltages;
    int time_step_index = 0;

    thrust::host_vector<double> previous_iteration_voltages;
    double relaxation_factor = 0.7; // Adjust as needed: 0.5-0.8 works well

    // Node and branch mappings
    std::unordered_map<std::string, int> nodeMap;  // Maps node names to local indices
    std::unordered_map<std::string, int> branchMap;  // Maps branch names to local indices
    std::vector<int> globalNodeIndices;  // Maps local indices to global indices

    // Network elements in this subnetwork
    std::vector<NetworkElement*> elements;

    // Sparse matrix format for the admittance matrix
    int numNodes;
    int numBranches;
    thrust::host_vector<double> h_values;  // Non-zero values
    thrust::host_vector<int> h_rowPtr;     // CSR row pointers
    thrust::host_vector<int> h_colInd;     // Column indices
    thrust::host_vector<double> h_voltages;  // Host copy of node voltages
    thrust::host_vector<double> h_currents;  // Host copy of current vector
    thrust::host_vector<double> h_branchCurrents;  // Host copy of branch currents

    // Device memory for matrices and vectors
    thrust::device_vector<double> d_values;
    thrust::device_vector<int> d_rowPtr;
    thrust::device_vector<int> d_colInd;
    thrust::device_vector<double> d_voltages;
    thrust::device_vector<double> d_currents;
    thrust::device_vector<double> d_historyTerms;
    thrust::device_vector<double> d_branchCurrents;
    thrust::device_vector<double> d_branchParameters;

    // CUDA libraries
    cusparseHandle_t cusparseHandle;
    cusolverSpHandle_t cusolverHandle;
    cublasHandle_t cublasHandle;

    // CUDA Sparse matrix descriptor
    cusparseMatDescr_t matDescr;

    // Performance counters
    double buildMatrixTime;
    double solverTime;
    int iterationsPerformed;

public:
    Subnetwork(int id, int deviceID, cudaStream_t stream, double timeStep, int numNodes);
    ~Subnetwork();

    void addNode(int globalIndex, const std::string& nodeName);
    void addBranch(const std::string& branchName);
    void addElement(NetworkElement* element);
    int getLocalNodeIndex(int globalIndex) const;

    int getId() const { return id; }
    int getDeviceId() const { return deviceID; }
    double getTimeStep() const { return timeStep; }

    // Get number of nodes and branches
    int getNumNodes() const { return numNodes; }
    int getNumBranches() const { return numBranches; }

    // Get performance metrics
    double getBuildMatrixTime() const { return buildMatrixTime; }
    double getSolverTime() const { return solverTime; }
    int getIterationsPerformed() const { return iterationsPerformed; }

    void resetPerformanceCounters() {
        buildMatrixTime = 0.0;
        solverTime = 0.0;
        iterationsPerformed = 0;
    }

    const std::unordered_map<std::string, int>& getNodeMap() const;
    const std::unordered_map<std::string, int>& getBranchMap() const;

    double getNodeVoltage(int localIndex) const;
    void setNodeVoltage(int localIndex, double voltage);
    double getBranchCurrent(int localIndex) const;

    void solve(double time, int iteration);
    void updateHistoryTerms(double time);
    void buildAdmittanceMatrix();
    void updateCurrentVector(double time);
    void solveLinearSystem();
    void updateBranchCurrents();

    // Helper method to initialize matrix structure based on network topology
    void initializeMatrixStructure();

    // Helper method to debug matrix and vectors
    void debugMatrixAndVectors(const std::string& phase);
};