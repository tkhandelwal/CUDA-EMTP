// emtp_solver.h
#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cusolverSp.h>
#include "simulation_results.h"
#include "cuda_helpers.h"

// Forward declarations
class Subnetwork;
class BoundaryNode;
class NetworkElement;
class Visualization;
class DataCommunicator;
class PowerElectronicConverter;
class ControlSystem;
class SemiconductorDevice;
class ThermalModel;

/**
 * @brief Main class for the commercial EMTP solver
 */
class EMTPSolver {
private:
    // Simulation parameters
    double timeStep;
    double endTime;
    int numTimeSteps;
    int numGPUs;
    bool realTimeVisualization;
    std::string outputFileName;
    int solverType;
    int maxWaveformIterations;
    double convergenceTolerance;
    int outputDecimation;
    bool initFromLoadflow;
    double accelerationFactor;
    int sparseMatrixSolver;


    // Network extraction
    std::unordered_map<std::string, std::pair<float, float>> extractNodePositions(const std::string& networkFile);
    std::vector<std::pair<std::string, std::string>> extractBranches(const std::string& networkFile);
    void connectPowerElectronicsComponents();

    // Network partitioning
    std::vector<std::unique_ptr<Subnetwork>> subnetworks;
    std::vector<BoundaryNode> boundaryNodes;
    std::unordered_map<std::string, int> nodeNameToIndex;
    std::vector<std::string> nodeNames;
    std::unordered_map<std::string, double> nodeNominalVoltages;
    std::unordered_map<std::string, int> nodeTypes;

    // Power Electronics
    std::vector<std::unique_ptr<PowerElectronicConverter>> converters;
    std::unordered_map<std::string, int> converterNameToIndex;

    // Control Systems
    std::vector<std::unique_ptr<ControlSystem>> controlSystems;
    std::unordered_map<std::string, int> controlSystemNameToIndex;

    // Semiconductor Devices
    std::vector<std::unique_ptr<SemiconductorDevice>> semiconductorDevices;
    std::unordered_map<std::string, int> deviceNameToIndex;

    // Thermal Models
    std::vector<std::unique_ptr<ThermalModel>> thermalModels;
    std::unordered_map<std::string, int> thermalModelNameToIndex;

    // CUDA resources
    std::vector<cudaStream_t> streams;
    std::vector<int> deviceIDs;

    // Visualization
    std::unique_ptr<Visualization> visualizer;
    std::unique_ptr<DataCommunicator> dataCommunicator;

    // Results
    SimulationResults results;

    // Multithreading for real-time updates
    std::mutex resultsMutex;
    std::condition_variable resultsCV;
    std::atomic<bool> simulationRunning;
    std::atomic<bool> visualizationActive;
    std::atomic<bool> pauseSimulation;

    // Performance tracking
    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point simulationEndTime;

public:
    /**
     * @brief Constructor for EMTP solver
     * @param timeStep Simulation time step in seconds
     * @param endTime Total simulation time in seconds
     * @param enableVisualization Enable real-time visualization
     * @param maxIterations Maximum waveform relaxation iterations
     * @param tolerance Convergence tolerance
     */
    EMTPSolver(double timeStep = 50e-6, double endTime = 0.1,
        bool enableVisualization = true,
        int maxIterations = 10, double tolerance = 1e-6);

    /**
     * @brief Destructor to clean up CUDA resources
     */
    ~EMTPSolver();

    /**
     * @brief Load network topology and components from a file
     * @param networkFile Path to network description file
     * @return True if successfully loaded, false otherwise
     */
    bool loadNetwork(const std::string& networkFile);

    /**
     * @brief Process a node definition line from the input file
     * @param line Line containing node definition
     */
    void processNodeLine(const std::string& line);

    /**
     * @brief Process a branch definition line from the input file
     * @param line Line containing branch definition
     */
    void processBranchLine(const std::string& line);

    /**
     * @brief Process a source definition line from the input file
     * @param line Line containing source definition
     */
    void processSourceLine(const std::string& line);

    /**
     * @brief Process a power electronics device line from the input file
     * @param line Line containing power electronics definition
     */
    void processPowerElectronicsLine(const std::string& line);

    /**
     * @brief Process a semiconductor device line from the input file
     * @param line Line containing semiconductor device definition
     */
    void processSemiconductorDeviceLine(const std::string& line);

    /**
     * @brief Process a control system line from the input file
     * @param line Line containing control system definition
     */
    void processControlSystemLine(const std::string& line);

    /**
     * @brief Process a thermal model line from the input file
     * @param line Line containing thermal model definition
     */
    void processThermalModelLine(const std::string& line);

    /**
     * @brief Process a switch definition line from the input file
     * @param line Line containing switch definition
     */
    void processSwitchLine(const std::string& line);

    /**
     * @brief Process a simulation parameters line from the input file
     * @param line Line containing simulation parameters
     */
    void processSimulationParametersLine(const std::string& line);

    /**
     * @brief Process an output control line from the input file
     * @param line Line containing output control
     */
    void processOutputLine(const std::string& line);

    /**
     * @brief Partition the network into subnetworks for multi-GPU processing
     * @return True if successfully partitioned, false otherwise
     */
    bool partitionNetwork();

    /**
     * @brief Identify nodes that exist at the boundary between subnetworks
     * @return True if successfully identified, false otherwise
     */
    bool identifyBoundaryNodes();

    /**
     * @brief Run the EMTP simulation
     * @param useRealTimeWeb If true, sends data to a web-based dashboard
     * @return True if simulation completes successfully, false otherwise
     */
    bool runSimulation(bool useRealTimeWeb = false);

    /**
     * @brief Exchange data at boundary nodes between subnetworks
     */
    void exchangeBoundaryData();

    /**
     * @brief Check if waveform relaxation has converged
     * @return True if converged, false otherwise
     */
    bool checkConvergence();

    /**
     * @brief Collect and store simulation results for the current time step
     * @param timeStep Current time step index
     */
    void collectResults(int timeStep);

    /**
     * @brief Initialize the simulation from a load flow solution
     * @return True if initialization is successful, false otherwise
     */
    bool initializeFromLoadflow();

    /**
     * @brief Start the visualization thread
     */
    void startVisualization();

    /**
     * @brief Update visualization with latest results
     * @param currentTimeStep Current time step index
     */
    void updateVisualization(int currentTimeStep);

    /**
     * @brief Stop the visualization thread
     */
    void stopVisualization();

    /**
     * @brief Pause the simulation temporarily
     */
    void setPauseSimulation(bool pause) { pauseSimulation.store(pause); }

    /**
     * @brief Resume a paused simulation
     */
    void resumeSimulation() { this->pauseSimulation.store(false); }

    /**
     * @brief Stop the simulation if it's running
     */
    void stopSimulation();

    /**
     * @brief Print performance metrics
     */
    void printPerformanceMetrics();

    /**
     * @brief Export results to database
     * @param databaseFile Database file name, empty for in-memory
     * @return True if successfully exported, false otherwise
     */
    bool exportToDatabase(const std::string& databaseFile = "");

    /**
     * @brief Set output file name
     * @param filename Output file name
     */
    void setOutputFileName(const std::string& filename) { outputFileName = filename; }

    /**
     * @brief Get simulation status
     * @return True if simulation is running, false otherwise
     */
    bool isRunning() const { return simulationRunning.load(); }

    /**
     * @brief Get simulation results
     * @return Reference to simulation results structure
     */
    const SimulationResults& getResults() const { return results; }
};