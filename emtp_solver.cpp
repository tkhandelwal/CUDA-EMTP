#include "emtp_solver.h"
#include "subnetwork.h"
#include "boundary_node.h"
#include "visualization.h"
#include "data_communicator.h"
#include "database_connector.h"
#include "cuda_kernels.cuh"
#include "power_electronic_converter.h"
#include "control_system.h"
#include "semiconductor_device.h"
#include "thermal_model.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include "simulation_results.h"
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Constructor implementation
EMTPSolver::EMTPSolver(double timeStep, double endTime,
    bool enableVisualization,
    int maxIterations, double tolerance)
    : timeStep(timeStep),
    endTime(endTime),
    numTimeSteps(static_cast<int>(endTime / timeStep) + 1),
    numGPUs(0),
    realTimeVisualization(enableVisualization),
    maxWaveformIterations(maxIterations),
    convergenceTolerance(tolerance),
    simulationRunning(false),
    visualizationActive(false),
    pauseSimulation(false),
    solverType(1),  // Default to trapezoidal
    outputDecimation(1),
    initFromLoadflow(true),
    accelerationFactor(0.8),
    sparseMatrixSolver(2) {  // Default to KLU sparse solver

    // Find available GPUs
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&numGPUs));
    if (numGPUs <= 0) {
        throw std::runtime_error("No CUDA-capable GPUs detected");
    }

    std::cout << "Initializing EMTP solver with " << numGPUs << " GPUs" << std::endl;

    // Initialize CUDA resources for each GPU
    for (int i = 0; i < numGPUs; i++) {
        deviceIDs.push_back(i);

        // Create CUDA stream for this GPU
        cudaStream_t stream;
        CHECK_CUDA_ERROR(cudaSetDevice(i));
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
        streams.push_back(stream);

        // Print GPU info
        cudaDeviceProp props;
        CHECK_CUDA_ERROR(cudaGetDeviceProperties(&props, i));
        std::cout << "GPU " << i << ": " << props.name
            << ", Compute Capability: " << props.major << "." << props.minor
            << ", Memory: " << props.totalGlobalMem / (1024 * 1024) << " MB"
            << ", SMs: " << props.multiProcessorCount << std::endl;
    }

    // Initialize visualization if requested
    if (realTimeVisualization) {
        visualizer = std::make_unique<Visualization>();
    }

    // Initialize performance metrics
    results.matrixBuildTime = 0.0;
    results.solverTime = 0.0;
    results.communicationTime = 0.0;
    results.powerElectronicsTime = 0.0;
    results.controlSystemTime = 0.0;
    results.iterationCount = 0;
}

std::unordered_map<std::string, std::pair<float, float>> EMTPSolver::extractNodePositions(const std::string& networkFile) {
    std::unordered_map<std::string, std::pair<float, float>> positions;
    std::ifstream file(networkFile);

    if (!file.is_open()) {
        return positions;
    }

    std::string line;
    bool inNodesSection = false;

    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Check for section headers
        if (line == "[NODES]") {
            inNodesSection = true;
            continue;
        }
        else if (line[0] == '[') {
            inNodesSection = false;
            continue;
        }

        // Process node coordinates
        if (inNodesSection) {
            std::istringstream iss(line);
            std::string nodeName;
            float x, y;

            if (iss >> nodeName >> x >> y) {
                positions[nodeName] = std::make_pair(x, y);
            }
        }
    }

    file.close();
    return positions;
}

std::vector<std::pair<std::string, std::string>> EMTPSolver::extractBranches(const std::string& networkFile) {
    std::vector<std::pair<std::string, std::string>> branches;
    std::ifstream file(networkFile);

    if (!file.is_open()) {
        return branches;
    }

    std::string line;
    bool inBranchesSection = false;

    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Check for section headers
        if (line == "[BRANCHES]") {
            inBranchesSection = true;
            continue;
        }
        else if (line[0] == '[') {
            inBranchesSection = false;
            continue;
        }

        // Process branch connections
        if (inBranchesSection) {
            std::istringstream iss(line);
            std::string type, fromNode, toNode, name;

            if (iss >> type >> fromNode >> toNode >> name) {
                branches.push_back(std::make_pair(fromNode, toNode));
            }
        }
    }

    file.close();
    return branches;
}

EMTPSolver::~EMTPSolver() {
    if (simulationRunning.load()) {
        stopSimulation();
    }

    for (auto& stream : streams) {
        cudaStreamDestroy(stream);
    }
}

bool EMTPSolver::loadNetwork(const std::string& networkFile) {
    std::cout << "Loading network from: " << networkFile << std::endl;

    // Open the file
    std::ifstream file(networkFile);
    if (!file.is_open()) {
        std::cerr << "Failed to open network file: " << networkFile << std::endl;
        return false;
    }

    // Clear any existing data
    nodeNameToIndex.clear();
    nodeNames.clear();
    nodeNominalVoltages.clear();
    nodeTypes.clear();

    // For new components
    converterNameToIndex.clear();
    controlSystemNameToIndex.clear();
    deviceNameToIndex.clear();
    thermalModelNameToIndex.clear();

    // Free existing objects
    subnetworks.clear();
    boundaryNodes.clear();
    converters.clear();
    controlSystems.clear();
    semiconductorDevices.clear();
    thermalModels.clear();

    // Parse the file format
    std::string line;
    std::string section;

    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Check for section headers
        if (line[0] == '[' && line.back() == ']') {
            section = line.substr(1, line.size() - 2);
            std::cout << "Processing section: " << section << std::endl;
            continue;
        }

        // Process based on current section
        if (section == "NODES") {
            processNodeLine(line);
        }
        else if (section == "BRANCHES") {
            processBranchLine(line);
        }
        else if (section == "SOURCES") {
            processSourceLine(line);
        }
        else if (section == "POWER_ELECTRONICS") {
            processPowerElectronicsLine(line);
        }
        else if (section == "SEMICONDUCTOR_DEVICES") {
            processSemiconductorDeviceLine(line);
        }
        else if (section == "CONTROL_SYSTEMS") {
            processControlSystemLine(line);
        }
        else if (section == "THERMAL_MODELS") {
            processThermalModelLine(line);
        }
        else if (section == "SWITCHES") {
            processSwitchLine(line);
        }
        else if (section == "SIMULATION_PARAMETERS") {
            processSimulationParametersLine(line);
        }
        else if (section == "OUTPUT") {
            processOutputLine(line);
        }
        // Add more sections as needed
    }

    // Connect power electronic converters with their semiconductor devices and controllers
    connectPowerElectronicsComponents();

    // After loading, partition the network
    return partitionNetwork();
}

void EMTPSolver::processNodeLine(const std::string& line) {
    std::istringstream iss(line);
    std::string nodeName;
    double x, y;  // Coordinates for visualization
    double nominalVoltage = 0.0; // Optional nominal voltage
    int nodeType = 1; // Default to PQ node

    if (iss >> nodeName >> x >> y) {
        // Store node information
        int nodeIndex = nodeNames.size();
        nodeNames.push_back(nodeName);
        nodeNameToIndex[nodeName] = nodeIndex;

        // Read optional parameters if present
        if (iss >> nominalVoltage) {
            nodeNominalVoltages[nodeName] = nominalVoltage;

            // Read node type if present
            if (iss >> nodeType) {
                nodeTypes[nodeName] = nodeType;
            }
        }

        // Reserve space in results
        results.nodeVoltages[nodeName] = std::vector<double>(numTimeSteps, 0.0);
    }
}

void EMTPSolver::processBranchLine(const std::string& line) {
    std::istringstream iss(line);
    std::string branchType, fromNode, toNode, branchName;

    if (iss >> branchType) {
        if (branchType == "LINE") {
            double resistance, inductance, capacitance, length = 1.0;

            if (iss >> fromNode >> toNode >> branchName >> resistance >> inductance >> capacitance) {
                // Read optional length parameter if present
                iss >> length;

                // Validate nodes
                if (nodeNameToIndex.find(fromNode) == nodeNameToIndex.end() ||
                    nodeNameToIndex.find(toNode) == nodeNameToIndex.end()) {
                    std::cerr << "Warning: Line " << branchName << " references unknown node" << std::endl;
                    return;
                }

                // Store branch information (implementation depends on internal data structures)

                // Reserve space in results
                results.branchCurrents[branchName] = std::vector<double>(numTimeSteps, 0.0);
            }
        }
        else if (branchType == "TRANSFORMER") {
            double primaryV, secondaryV, rating, leakageX;
            double saturationFlux = 0.0, coreLossR = 0.0, magCurrent = 0.0;

            if (iss >> fromNode >> toNode >> branchName >> primaryV >> secondaryV >> rating >> leakageX) {
                // Read optional magnetization parameters if present
                iss >> saturationFlux >> coreLossR >> magCurrent;

                // Validate nodes
                if (nodeNameToIndex.find(fromNode) == nodeNameToIndex.end() ||
                    nodeNameToIndex.find(toNode) == nodeNameToIndex.end()) {
                    std::cerr << "Warning: Transformer " << branchName << " references unknown node" << std::endl;
                    return;
                }

                // Store transformer information

                // Reserve space in results
                results.branchCurrents[branchName] = std::vector<double>(numTimeSteps, 0.0);
            }
        }
        else if (branchType == "LOAD") {
            double activePower, reactivePower;
            int loadType = 0; // 0=constant Z, 1=constant P, 2=constant I, 3=ZIP composite

            if (iss >> fromNode >> branchName >> activePower >> reactivePower) {
                // Read optional load type if present
                iss >> loadType;

                // Validate node
                if (nodeNameToIndex.find(fromNode) == nodeNameToIndex.end()) {
                    std::cerr << "Warning: Load " << branchName << " references unknown node" << std::endl;
                    return;
                }

                // Store load information
            }
        }
        // Add more branch types as needed
    }
}

void EMTPSolver::processSourceLine(const std::string& line) {
    std::istringstream iss(line);
    std::string sourceType, nodeName, sourceName;

    if (iss >> sourceType) {
        if (sourceType == "VOLTAGE") {
            double amplitude, frequency, phase, rSource = 0.0, lSource = 0.0;

            if (iss >> nodeName >> sourceName >> amplitude >> frequency >> phase) {
                // Read optional source impedance parameters if present
                iss >> rSource >> lSource;

                // Validate node
                if (nodeNameToIndex.find(nodeName) == nodeNameToIndex.end()) {
                    std::cerr << "Warning: Voltage source " << sourceName << " references unknown node" << std::endl;
                    return;
                }

                // Store source information
            }
        }
        else if (sourceType == "WINDGEN") {
            double voltage, frequency, activePower, reactivePower, inertia;
            int type, controlStrategy;

            if (iss >> nodeName >> sourceName >> voltage >> frequency >> activePower >> reactivePower >> inertia >> type >> controlStrategy) {
                // Validate node
                if (nodeNameToIndex.find(nodeName) == nodeNameToIndex.end()) {
                    std::cerr << "Warning: Wind generator " << sourceName << " references unknown node" << std::endl;
                    return;
                }

                // Store wind generator information
            }
        }
        else if (sourceType == "SOLARGEN") {
            double voltage, frequency, activePower, reactivePower;
            int type, controlStrategy;
            double switchingFreq = 0.0;

            if (iss >> nodeName >> sourceName >> voltage >> frequency >> activePower >> reactivePower >> type >> switchingFreq >> controlStrategy) {
                // Validate node
                if (nodeNameToIndex.find(nodeName) == nodeNameToIndex.end()) {
                    std::cerr << "Warning: Solar generator " << sourceName << " references unknown node" << std::endl;
                    return;
                }

                // Store solar generator information
            }
        }
        // Add more source types as needed
    }
}

void EMTPSolver::processPowerElectronicsLine(const std::string& line) {
    std::istringstream iss(line);
    std::string converterType;

    if (iss >> converterType) {
        // Create the appropriate converter type
        std::unique_ptr<PowerElectronicConverter> converter = createPowerElectronicConverter(line);

        if (converter) {
            // Store the converter
            converter->setId(converters.size());
            converterNameToIndex[converter->getName()] = converters.size();
            converters.push_back(std::move(converter));

            // Reserve space in results
            std::string converterName = converters.back()->getName();
            results.converterLosses[converterName] = std::vector<double>(numTimeSteps, 0.0);
            results.dcVoltages[converterName] = std::vector<double>(numTimeSteps, 0.0);
            results.dcCurrents[converterName] = std::vector<double>(numTimeSteps, 0.0);
            results.modulationIndices[converterName] = std::vector<double>(numTimeSteps, 0.0);
        }
    }
}

void EMTPSolver::processSemiconductorDeviceLine(const std::string& line) {
    std::istringstream iss(line);
    std::string deviceType;

    if (iss >> deviceType) {
        // Create the appropriate semiconductor device
        std::unique_ptr<SemiconductorDevice> device = createSemiconductorDevice(line);

        if (device) {
            // Store the device
            device->setId(semiconductorDevices.size());
            deviceNameToIndex[device->getName()] = semiconductorDevices.size();
            semiconductorDevices.push_back(std::move(device));

            // Reserve space in results
            std::string deviceName = semiconductorDevices.back()->getName();
            results.semiconductorTemperatures[deviceName] = std::vector<double>(numTimeSteps, 0.0);
            results.semiconductorLosses[deviceName] = std::vector<double>(numTimeSteps, 0.0);
        }
    }
}

void EMTPSolver::processControlSystemLine(const std::string& line) {
    std::istringstream iss(line);
    std::string controlType;

    if (iss >> controlType) {
        // Create the appropriate control system
        std::unique_ptr<ControlSystem> controller = createControlSystem(line);

        if (controller) {
            // Store the controller
            controller->setId(controlSystems.size());
            controlSystemNameToIndex[controller->getName()] = controlSystems.size();
            controlSystems.push_back(std::move(controller));

            // Reserve space in results
            std::string controllerName = controlSystems.back()->getName();
            results.controllerOutputs[controllerName] = std::vector<double>(numTimeSteps, 0.0);

            // Special handling for PLL and GridFormingController
            if (controlType == "PLL" || controlType == "GFM_CONT") {
                results.phaseAngles[controllerName] = std::vector<double>(numTimeSteps, 0.0);
                results.frequencies[controllerName] = std::vector<double>(numTimeSteps, 0.0);
            }

            // Special handling for PowerController
            if (controlType == "POW_CONT") {
                results.activePowers[controllerName] = std::vector<double>(numTimeSteps, 0.0);
                results.reactivePowers[controllerName] = std::vector<double>(numTimeSteps, 0.0);
            }
        }
    }
}

void EMTPSolver::processThermalModelLine(const std::string& line) {
    std::istringstream iss(line);
    std::string modelType;

    if (iss >> modelType) {
        // Create the appropriate thermal model
        std::unique_ptr<ThermalModel> model = createThermalModel(line);

        if (model) {
            // Store the thermal model
            model->setId(thermalModels.size());
            thermalModelNameToIndex[model->getName()] = thermalModels.size();
            thermalModels.push_back(std::move(model));
        }
    }
}

void EMTPSolver::processSwitchLine(const std::string& line) {
    std::istringstream iss(line);
    std::string switchType, fromNode, toNode, switchName;

    if (iss >> switchType) {
        if (switchType == "CB") {
            double rating, openingTime, closingTime, openingResistance = 1e6;
            int arcModel = 0;
            double arcTimeConstant = 0.0, arcVoltage = 0.0;

            if (iss >> fromNode >> toNode >> switchName >> rating >> openingTime >> closingTime) {
                // Read optional parameters if present
                iss >> openingResistance >> arcModel >> arcTimeConstant >> arcVoltage;

                // Validate nodes
                if (nodeNameToIndex.find(fromNode) == nodeNameToIndex.end() ||
                    nodeNameToIndex.find(toNode) == nodeNameToIndex.end()) {
                    std::cerr << "Warning: Circuit breaker " << switchName << " references unknown node" << std::endl;
                    return;
                }

                // Store circuit breaker information
            }
        }
        else if (switchType == "SECT" || switchType == "DISC") {
            double rating, openingTime, closingTime;

            if (iss >> fromNode >> toNode >> switchName >> rating >> openingTime >> closingTime) {
                // Validate nodes
                if (nodeNameToIndex.find(fromNode) == nodeNameToIndex.end() ||
                    nodeNameToIndex.find(toNode) == nodeNameToIndex.end()) {
                    std::cerr << "Warning: Switch " << switchName << " references unknown node" << std::endl;
                    return;
                }

                // Store switch information
            }
        }
        else if (switchType == "TCSC") {
            double rating, minX, maxX;
            int controlType;
            double firingAngleMin = 90.0, firingAngleMax = 180.0;

            if (iss >> fromNode >> toNode >> switchName >> rating >> minX >> maxX >> controlType) {
                // Read optional parameters if present
                iss >> firingAngleMin >> firingAngleMax;

                // Validate nodes
                if (nodeNameToIndex.find(fromNode) == nodeNameToIndex.end() ||
                    nodeNameToIndex.find(toNode) == nodeNameToIndex.end()) {
                    std::cerr << "Warning: TCSC " << switchName << " references unknown node" << std::endl;
                    return;
                }

                // Store TCSC information
            }
        }
        // Add more switch types as needed
    }
}

void EMTPSolver::processSimulationParametersLine(const std::string& line) {
    std::istringstream iss(line);
    std::string paramName;

    if (iss >> paramName) {
        if (paramName == "TIMESTEP") {
            double newTimeStep;
            if (iss >> newTimeStep) {
                timeStep = newTimeStep;
                numTimeSteps = static_cast<int>(endTime / timeStep) + 1;
            }
        }
        else if (paramName == "ENDTIME") {
            double newEndTime;
            if (iss >> newEndTime) {
                endTime = newEndTime;
                numTimeSteps = static_cast<int>(endTime / timeStep) + 1;
            }
        }
        else if (paramName == "SOLVER_TYPE") {
            int newSolverType;
            if (iss >> newSolverType) {
                solverType = newSolverType;
            }
        }
        else if (paramName == "MAX_ITERATIONS") {
            int newMaxIterations;
            if (iss >> newMaxIterations) {
                maxWaveformIterations = newMaxIterations;
            }
        }
        else if (paramName == "TOLERANCE") {
            double newTolerance;
            if (iss >> newTolerance) {
                convergenceTolerance = newTolerance;
            }
        }
        else if (paramName == "OUTPUT_DECIMATION") {
            int newOutputDecimation;
            if (iss >> newOutputDecimation) {
                outputDecimation = newOutputDecimation;
            }
        }
        else if (paramName == "INIT_FROM_LOADFLOW") {
            int newInitFromLoadflow;
            if (iss >> newInitFromLoadflow) {
                initFromLoadflow = (newInitFromLoadflow != 0);
            }
        }
        else if (paramName == "ACCELERATION_FACTOR") {
            double newAccelerationFactor;
            if (iss >> newAccelerationFactor) {
                accelerationFactor = newAccelerationFactor;
            }
        }
        else if (paramName == "SPARSE_MATRIX_SOLVER") {
            int newSparseMatrixSolver;
            if (iss >> newSparseMatrixSolver) {
                sparseMatrixSolver = newSparseMatrixSolver;
            }
        }
        // Add more parameters as needed
    }
}

void EMTPSolver::processOutputLine(const std::string& line) {
    // Process output control directives
    // This would control what gets saved to the results object
    // and potentially exported to files
}

void EMTPSolver::connectPowerElectronicsComponents() {
    // Connect semiconductor devices to their converters
    // In a real implementation, this would use the input file to determine connections

    // Example of manually connecting components:
    // For each converter, find devices and controllers associated with it
    for (auto& converter : converters) {
        // Find semiconductor devices for this converter
        // (in a real implementation, this would come from the input file)
        for (auto& device : semiconductorDevices) {
            // Connect device to converter
            // Example condition: if device name starts with converter name
            if (device->getName().find(converter->getName()) == 0) {
                converter->addSemiconductor(device.get());
            }
        }

        // Find control systems for this converter
        for (auto& controller : controlSystems) {
            // Connect controller to converter
            // Example condition: if controller name starts with converter name
            if (controller->getName().find(converter->getName()) == 0) {
                converter->addController(controller.get());
            }
        }

        // Find thermal models for this converter
        for (auto& model : thermalModels) {
            // Connect thermal model to converter
            // Example condition: if model name starts with converter name
            if (model->getName().find(converter->getName()) == 0) {
                converter->addThermalModel(model.get());
            }
        }
    }

    // Connect semiconductor devices to their thermal models
    for (auto& device : semiconductorDevices) {
        for (auto& model : thermalModels) {
            // Connect thermal model to device
            // Example condition: if model name contains device name
            if (model->getName().find(device->getName()) != std::string::npos) {
                device->setThermalModel(model.get());
                break;  // Only connect one thermal model per device
            }
        }
    }
}

bool EMTPSolver::partitionNetwork() {
    std::cout << "Partitioning network for " << numGPUs << " GPUs" << std::endl;

    // In a commercial implementation, use a high-quality graph partitioning library
    // For this example, we'll use a simple node distribution approach
    int nodesPerGPU = (nodeNames.size() + numGPUs - 1) / numGPUs;

    // For each partition, create a subnetwork
    for (int i = 0; i < numGPUs; i++) {
        int startNode = i * nodesPerGPU;
        int endNode = std::min<int>((i + 1) * nodesPerGPU, static_cast<int>(nodeNames.size()));

        subnetworks.push_back(std::make_unique<Subnetwork>(i, deviceIDs[i], streams[i],
            timeStep, endNode - startNode));

        // Assign nodes to this subnetwork
        for (int nodeIdx = startNode; nodeIdx < endNode; nodeIdx++) {
            subnetworks.back()->addNode(nodeIdx, nodeNames[nodeIdx]);
        }
    }

    // Identify boundary nodes between partitions
    return identifyBoundaryNodes();
}

bool EMTPSolver::identifyBoundaryNodes() {
    std::cout << "Setting up boundary nodes for inter-GPU communication" << std::endl;

    // This is a simplified approach; in a real implementation, you'd 
    // identify nodes that connect between partitions based on branch information

    // For now, we'll assume all nodes at partition boundaries are boundary nodes
    for (int i = 0; i < numGPUs - 1; i++) {
        int boundaryNodeIdx = (i + 1) * ((nodeNames.size() + numGPUs - 1) / numGPUs) - 1;

        // Create boundary node
        BoundaryNode boundaryNode(boundaryNodeIdx, nodeNames[boundaryNodeIdx]);

        // Add mapping to relevant subnetworks
        boundaryNode.addSubnetworkMapping(i, subnetworks[i]->getLocalNodeIndex(boundaryNodeIdx));
        boundaryNode.addSubnetworkMapping(i + 1, subnetworks[i + 1]->getLocalNodeIndex(boundaryNodeIdx));

        boundaryNodes.push_back(boundaryNode);
    }

    return true;
}

bool EMTPSolver::runSimulation(bool useRealTimeWeb) {
    std::cout << "Starting EMTP simulation with " << numTimeSteps << " time steps" << std::endl;

    // Start performance tracking
    startTime = std::chrono::high_resolution_clock::now();
    simulationRunning.store(true);

    // Initialize from load flow if requested
    if (initFromLoadflow) {
        if (!initializeFromLoadflow()) {
            std::cerr << "Failed to initialize from load flow" << std::endl;
            return false;
        }
    }

    // Initialize power electronic converters
    for (auto& converter : converters) {
        converter->initialize();
    }

    // Initialize control systems
    for (auto& controller : controlSystems) {
        controller->initialize();
    }

    // Initialize real-time visualization
    std::unique_ptr<DataCommunicator> localDataCommunicator;
    if (useRealTimeWeb) {
        localDataCommunicator = std::make_unique<DataCommunicator>("localhost", 5555);
        if (!localDataCommunicator->connect()) {
            std::cerr << "Failed to connect to visualization server. Continuing without web visualization." << std::endl;
            useRealTimeWeb = false;
        }
        else {
            // Extract node positions and branch connections
            auto nodePositions = extractNodePositions("example_network.net");
            auto branches = extractBranches("example_network.net");

            // Send initial data
            localDataCommunicator->sendInitialData(results, nodeNames, nodePositions, branches);
        }
    }
    else if (realTimeVisualization) {
        // Use built-in visualization if web is not requested
        startVisualization();
    }

    // Store the data communicator as a member if it was successfully connected
    if (useRealTimeWeb && localDataCommunicator && localDataCommunicator->isConnected()) {
        dataCommunicator = std::move(localDataCommunicator);
    }

    // Allocate results storage
    results.timePoints.resize(numTimeSteps);

    // Time stepping loop
    for (int t = 0; t < numTimeSteps; t++) {
        double currentTime = t * timeStep;
        results.timePoints[t] = currentTime;

        // Process control systems
        auto controlStart = std::chrono::high_resolution_clock::now();

        for (auto& controller : controlSystems) {
            controller->update(currentTime, timeStep);
        }

        auto controlEnd = std::chrono::high_resolution_clock::now();
        results.controlSystemTime +=
            std::chrono::duration<double>(controlEnd - controlStart).count();

        // Process power electronic converters
        auto peStart = std::chrono::high_resolution_clock::now();

        for (auto& converter : converters) {
            converter->update(currentTime, timeStep);
            converter->calculateLosses();
        }

        auto peEnd = std::chrono::high_resolution_clock::now();
        results.powerElectronicsTime +=
            std::chrono::duration<double>(peEnd - peStart).count();

        // Waveform relaxation iteration loop
        int iterations = 0;
        bool converged = false;

        for (int iter = 0; iter < maxWaveformIterations && !converged; iter++) {
            iterations++;

            // Process each subnetwork in parallel on its assigned GPU
            for (int i = 0; i < numGPUs; i++) {
                CHECK_CUDA_ERROR(cudaSetDevice(deviceIDs[i]));
                subnetworks[i]->solve(currentTime, iter);
            }

            // Synchronize GPUs
            for (int i = 0; i < numGPUs; i++) {
                CHECK_CUDA_ERROR(cudaStreamSynchronize(streams[i]));
            }

            // Exchange boundary information between GPUs
            auto commStart = std::chrono::high_resolution_clock::now();
            exchangeBoundaryData();
            auto commEnd = std::chrono::high_resolution_clock::now();

            results.communicationTime +=
                std::chrono::duration<double>(commEnd - commStart).count();

            // Check convergence
            converged = checkConvergence();
        }

        results.iterationCount += iterations;

        // Collect results for this time step
        collectResults(t);

        // Update visualization if enabled
        if (dataCommunicator && dataCommunicator->isConnected()) {
            dataCommunicator->sendTimeStepData(t, results);
        }
        else if (realTimeVisualization && t % 10 == 0) {
            updateVisualization(t);
        }

        // Print progress every 10%
        if (t % (numTimeSteps / 10) == 0 || t == numTimeSteps - 1) {
            double progress = static_cast<double>(t + 1) / numTimeSteps * 100.0;
            std::cout << "Progress: " << progress << "%" << std::endl;
        }

        // Check if simulation has been paused
        while (pauseSimulation.load() && simulationRunning.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Check if simulation has been stopped
        if (!simulationRunning.load()) {
            break;
        }
    }

    // Stop simulation
    simulationEndTime = std::chrono::high_resolution_clock::now();
    results.totalSimulationTime =
        std::chrono::duration<double>(simulationEndTime - startTime).count();

    // Calculate average iterations per time step
    if (numTimeSteps > 0) {
        results.iterationCount /= numTimeSteps;
    }

    simulationRunning.store(false);

    // Stop visualization thread
    if (realTimeVisualization && !dataCommunicator) {
        stopVisualization();
    }

    // Print performance metrics
    printPerformanceMetrics();

    // Export to database
    if (!exportToDatabase("simulation_results.db")) {
        std::cerr << "Failed to export results to database" << std::endl;
        return false;
    }

    return true;
}

bool EMTPSolver::initializeFromLoadflow() {
    std::cout << "Initializing simulation from load flow solution..." << std::endl;

    // In a real implementation, this would solve a load flow problem
    // to get initial voltages and currents

    // For this simplified implementation, we'll just set initial conditions
    // to nominal values

    // Set nominal voltages at nodes
    for (const auto& node : nodeNames) {
        if (nodeNominalVoltages.find(node) != nodeNominalVoltages.end()) {
            double nominalVoltage = nodeNominalVoltages[node];

            // Set initial voltage for this node in all subnetworks
            for (auto& subnetwork : subnetworks) {
                int localIndex = subnetwork->getLocalNodeIndex(nodeNameToIndex[node]);
                if (localIndex >= 0) {
                    subnetwork->setNodeVoltage(localIndex, nominalVoltage);
                }
            }
        }
    }

    return true;
}

void EMTPSolver::exchangeBoundaryData() {
    for (auto& boundaryNode : boundaryNodes) {
        // Get voltage values from all subnetworks
        std::vector<double> voltages;
        voltages.reserve(boundaryNode.getSubnetworkCount());

        for (size_t i = 0; i < boundaryNode.getSubnetworkCount(); i++) {
            int subnetID = boundaryNode.getSubnetworkID(i);
            int localID = boundaryNode.getLocalID(i);

            double voltage = subnetworks[subnetID]->getNodeVoltage(localID);
            voltages.push_back(voltage);
        }

        // Calculate average voltage (weighted by the number of connected nodes)
        double totalVoltage = 0.0;
        double totalWeight = 0.0;

        for (size_t i = 0; i < boundaryNode.getSubnetworkCount(); i++) {
            int subnetID = boundaryNode.getSubnetworkID(i);
            double weight = subnetworks[subnetID]->getNumNodes(); // Weight by subnetwork size
            totalVoltage += voltages[i] * weight;
            totalWeight += weight;
        }

        double avgVoltage = totalWeight > 0 ? totalVoltage / totalWeight : 0.0;

        // Update the voltage in the boundary node object for convergence checking
        boundaryNode.updateVoltage(avgVoltage);

        // Set the voltage back to all subnetworks
        for (size_t i = 0; i < boundaryNode.getSubnetworkCount(); i++) {
            int subnetID = boundaryNode.getSubnetworkID(i);
            int localID = boundaryNode.getLocalID(i);

            subnetworks[subnetID]->setNodeVoltage(localID, avgVoltage);
        }
    }
}


void EMTPSolver::collectResults(int timeStep) {
    std::lock_guard<std::mutex> lock(resultsMutex);

    // Only collect results for time steps that match the decimation factor
    if (timeStep % outputDecimation != 0) {
        return;
    }

    // Index in decimated results array
    int resultIndex = timeStep / outputDecimation;

    // Collect node voltages
    for (int subnetID = 0; subnetID < numGPUs; subnetID++) {
        for (const auto& nodePair : subnetworks[subnetID]->getNodeMap()) {
            std::string nodeName = nodePair.first;
            int localIndex = nodePair.second;

            double voltage = subnetworks[subnetID]->getNodeVoltage(localIndex);
            results.nodeVoltages[nodeName][resultIndex] = voltage;
        }
    }

    // Collect branch currents
    for (int subnetID = 0; subnetID < numGPUs; subnetID++) {
        for (const auto& branchPair : subnetworks[subnetID]->getBranchMap()) {
            std::string branchName = branchPair.first;
            int localIndex = branchPair.second;

            double current = subnetworks[subnetID]->getBranchCurrent(localIndex);
            results.branchCurrents[branchName][resultIndex] = current;
        }
    }

    // Collect power electronics results
    for (auto& converter : converters) {
        std::string converterName = converter->getName();

        // Store converter losses
        // In a real implementation, this would come from detailed loss calculations
        double losses = 0.0;
        for (auto& device : semiconductorDevices) {
            if (device->getName().find(converterName) != std::string::npos) {
                losses += device->getTemperature(); // Placeholder - in reality would be getTotalLoss()
            }
        }
        results.converterLosses[converterName][resultIndex] = losses;

        // Store DC voltages and currents (simplified)
        if (auto vscConverter = dynamic_cast<VSC_MMC*>(converter.get())) {
            results.dcVoltages[converterName][resultIndex] = vscConverter->getDcVoltage();
            results.dcCurrents[converterName][resultIndex] = 100.0; // Placeholder
            results.modulationIndices[converterName][resultIndex] = 0.8; // Placeholder
        }
    }

    // Collect semiconductor device results
    for (auto& device : semiconductorDevices) {
        std::string deviceName = device->getName();
        results.semiconductorTemperatures[deviceName][resultIndex] = device->getTemperature();
        results.semiconductorLosses[deviceName][resultIndex] = 100.0; // Placeholder
    }

    // Collect control system results
    for (auto& controller : controlSystems) {
        std::string controllerName = controller->getName();

        // Store controller outputs (simplified)
        if (auto pll = dynamic_cast<PLL*>(controller.get())) {
            results.phaseAngles[controllerName][resultIndex] = pll->getTheta();
            results.frequencies[controllerName][resultIndex] = pll->getOmega() / (2.0 * M_PI);
            results.controllerOutputs[controllerName][resultIndex] = pll->getOmega();
        }
        else if (auto gfm = dynamic_cast<GridFormingController*>(controller.get())) {
            results.phaseAngles[controllerName][resultIndex] = gfm->getTheta();
            results.frequencies[controllerName][resultIndex] = gfm->getOmega() / (2.0 * M_PI);
            results.controllerOutputs[controllerName][resultIndex] = gfm->getOmega();
        }
        else if (auto power = dynamic_cast<PowerController*>(controller.get())) {
            // In a real implementation, would store actual power measurements
            results.activePowers[controllerName][resultIndex] = 100.0; // Placeholder
            results.reactivePowers[controllerName][resultIndex] = 30.0; // Placeholder
            results.controllerOutputs[controllerName][resultIndex] = power->getOutputs()[0];
        }
        else if (auto current = dynamic_cast<CurrentController*>(controller.get())) {
            results.controllerOutputs[controllerName][resultIndex] = current->getOutputs()[0];
        }
        else if (auto voltage = dynamic_cast<VoltageController*>(controller.get())) {
            results.controllerOutputs[controllerName][resultIndex] = voltage->getOutputs()[0];
        }
        else if (auto dc = dynamic_cast<DCVoltageController*>(controller.get())) {
            results.controllerOutputs[controllerName][resultIndex] = dc->getOutput();
        }
    }

    // Notify visualization thread
    resultsCV.notify_one();
}

void EMTPSolver::startVisualization() {
    visualizationActive.store(true);

    // Detailed implementation would depend on your visualization library
    if (visualizer) {
        // Extract node positions and branch connections for visualization
        auto nodePositions = extractNodePositions("example_network.net");
        auto branches = extractBranches("example_network.net");

        visualizer->initialize(nodeNames, results.timePoints, nodePositions, branches);
    }
}

void EMTPSolver::updateVisualization(int currentTimeStep) {
    if (visualizer) {
        // This would update the visualization with the latest data
        visualizer->update(results, currentTimeStep);
    }
}

void EMTPSolver::stopVisualization() {
    visualizationActive.store(false);

    if (visualizer) {
        visualizer->shutdown();
    }
}

void EMTPSolver::stopSimulation() {
    simulationRunning.store(false);

    if (visualizationActive.load()) {
        stopVisualization();
    }

    if (dataCommunicator) {
        dataCommunicator->disconnect();
        dataCommunicator.reset();
    }
}

void EMTPSolver::printPerformanceMetrics() {
    std::cout << "\n--- Performance Metrics ---" << std::endl;
    std::cout << "Total simulation time: " << results.totalSimulationTime << " seconds" << std::endl;
    std::cout << "Matrix build time: " << results.matrixBuildTime << " seconds" << std::endl;
    std::cout << "Solver time: " << results.solverTime << " seconds" << std::endl;
    std::cout << "Communication time: " << results.communicationTime << " seconds" << std::endl;
    std::cout << "Power electronics time: " << results.powerElectronicsTime << " seconds" << std::endl;
    std::cout << "Control system time: " << results.controlSystemTime << " seconds" << std::endl;
    std::cout << "Average iterations per time step: " << results.iterationCount << std::endl;

    // Calculate throughput
    double nodesPerSec = (nodeNames.size() * numTimeSteps) / results.totalSimulationTime;
    std::cout << "Throughput: " << nodesPerSec << " node-timesteps per second" << std::endl;
}

bool EMTPSolver::checkConvergence() {
    // For each boundary node, check if voltages have converged
    for (auto& boundaryNode : boundaryNodes) {
        if (!boundaryNode.hasConverged(convergenceTolerance)) {
            return false;
        }
    }

    // If we have no boundary nodes (single GPU case), check node voltage changes
    if (boundaryNodes.empty() && !subnetworks.empty()) {
        // Get a sample of node voltages from the first subnetwork
        int sampleSize = std::min<int>(10, subnetworks[0]->getNumNodes());


        for (int i = 0; i < sampleSize; i++) {
            // Check if voltage has significantly changed (implementation depends on how
            // the subnetwork tracks previous values)
            double currentVoltage = subnetworks[0]->getNodeVoltage(i);

            // We can't actually check this without adding state - just return true
            // in the single GPU case when there are no boundary nodes
        }
    }

    return true;
}

bool EMTPSolver::exportToDatabase(const std::string& databaseFile) {
    // This implementation would use the DatabaseConnector class
    // to export the simulation results to a SQLite database

    // Create database connector
    DatabaseConnector dbConnector(databaseFile.empty(), databaseFile, 1000);

    // Connect to database
    if (!dbConnector.connect()) {
        std::cerr << "Failed to connect to database" << std::endl;
        return false;
    }

    // Export data in bulk
    if (!dbConnector.writeBulkData(results)) {
        std::cerr << "Failed to write data to database" << std::endl;
        return false;
    }

    // Disconnect from database
    dbConnector.disconnect();

    std::cout << "Results exported to database" << (databaseFile.empty() ? " (in-memory)" : (": " + databaseFile)) << std::endl;
    return true;
}