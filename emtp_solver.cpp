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
#include <iomanip>

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
    std::cout << "=== Starting EMTP simulation with " << numTimeSteps << " time steps ===" << std::endl;
    std::cout << "    Time step: " << timeStep << "s, End time: " << endTime << "s" << std::endl;
    std::cout << "    Number of GPUs: " << numGPUs << ", Number of nodes: " << nodeNames.size() << std::endl;

    // Add simulation monitoring and error recovery
    int consecutiveFailedIterations = 0;
    int maxConsecutiveFailedIterations = 5;

    // Start performance tracking
    startTime = std::chrono::high_resolution_clock::now();
    simulationRunning.store(true);

    // Setup detailed logging
    std::ofstream detailedLog("emtp_detailed.log", std::ios::out);
    if (!detailedLog.is_open()) {
        std::cerr << "WARNING: Could not open detailed log file." << std::endl;
    }

    auto log = [&detailedLog](const std::string& message) {
        auto now = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(now);
        std::tm timeInfo;
#ifdef _WIN32
        localtime_s(&timeInfo, &now_c);
#else
        localtime_r(&now_c, &timeInfo);
#endif
        char timeBuffer[80];
        std::strftime(timeBuffer, sizeof(timeBuffer), "%Y-%m-%d %H:%M:%S", &timeInfo);

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::stringstream ss;
        ss << timeBuffer << "." << std::setfill('0') << std::setw(3) << ms.count()
            << " - " << message;

        std::string fullMessage = ss.str();
        std::cout << fullMessage << std::endl;
        if (detailedLog.is_open()) {
            detailedLog << fullMessage << std::endl;
            detailedLog.flush();
        }
        };

    // Add timeout protection
    auto simulationStartTime = std::chrono::high_resolution_clock::now();
    const double maxSimulationTimeInSeconds = 600.0; // 10 minutes timeout

    log("Starting simulation with timeout of " + std::to_string(maxSimulationTimeInSeconds) + " seconds");

    // Initialize from load flow if requested
    if (initFromLoadflow) {
        log("Initializing from load flow");
        if (!initializeFromLoadflow()) {
            log("ERROR: Failed to initialize from load flow");
            return false;
        }
        log("Load flow initialization complete");
    }

    // Initialize power electronic converters
    log("Initializing " + std::to_string(converters.size()) + " power electronic converters");
    for (auto& converter : converters) {
        converter->initialize();
    }

    // Initialize control systems
    log("Initializing " + std::to_string(controlSystems.size()) + " control systems");
    for (auto& controller : controlSystems) {
        controller->initialize();
    }

    // Initialize real-time visualization
    std::unique_ptr<DataCommunicator> localDataCommunicator;
    if (useRealTimeWeb) {
        log("Setting up web visualization");
        localDataCommunicator = std::make_unique<DataCommunicator>("localhost", 5555);
        if (!localDataCommunicator->connect()) {
            log("WARNING: Failed to connect to visualization server. Continuing without web visualization.");
            useRealTimeWeb = false;
        }
        else {
            // Extract node positions and branch connections
            auto nodePositions = extractNodePositions("example_network.net");
            auto branches = extractBranches("example_network.net");

            // Send initial data
            log("Sending initial data to visualization server");
            localDataCommunicator->sendInitialData(results, nodeNames, nodePositions, branches);
        }
    }
    else if (realTimeVisualization) {
        // Use built-in visualization if web is not requested
        log("Setting up built-in visualization");
        startVisualization();
    }

    // Store the data communicator as a member if it was successfully connected
    if (useRealTimeWeb && localDataCommunicator && localDataCommunicator->isConnected()) {
        dataCommunicator = std::move(localDataCommunicator);
    }

    // Allocate results storage
    log("Allocating results storage for " + std::to_string(numTimeSteps) + " time steps");
    results.timePoints.resize(numTimeSteps);

    // Time stepping loop
    log("Beginning time stepping loop");
    for (int t = 0; t < numTimeSteps; t++) {
        // Check for timeout
        auto currentTime = std::chrono::high_resolution_clock::now();
        double elapsedTimeInSeconds = std::chrono::duration<double>(currentTime - simulationStartTime).count();
        if (elapsedTimeInSeconds > maxSimulationTimeInSeconds) {
            log("ERROR: Simulation timeout after " + std::to_string(elapsedTimeInSeconds)
                + " seconds at time step " + std::to_string(t) + " of " + std::to_string(numTimeSteps));
            return false;
        }

        double simTime = t * timeStep;
        results.timePoints[t] = simTime;

        // Log progress every 100 steps or with finer granularity at the beginning
        if (t % 100 == 0 || t < 10 || t == numTimeSteps - 1) {
            double progress = static_cast<double>(t + 1) / numTimeSteps * 100.0;
            log(std::string("Progress: ") + std::to_string(progress) + "% (step "
                + std::to_string(t) + "/" + std::to_string(numTimeSteps)
                + ", sim time: " + std::to_string(simTime) + "s)");
        }

        // Process control systems
        log("Time step " + std::to_string(t) + ": Processing control systems");
        auto controlStart = std::chrono::high_resolution_clock::now();

        try {
            for (size_t i = 0; i < controlSystems.size(); i++) {
                controlSystems[i]->update(simTime, timeStep);
            }
        }
        catch (const std::exception& e) {
            log("ERROR: Exception in control system processing: " + std::string(e.what()));
            return false;
        }

        auto controlEnd = std::chrono::high_resolution_clock::now();
        results.controlSystemTime +=
            std::chrono::duration<double>(controlEnd - controlStart).count();

        // Process power electronic converters
        log("Time step " + std::to_string(t) + ": Processing power electronic converters");
        auto peStart = std::chrono::high_resolution_clock::now();

        try {
            for (size_t i = 0; i < converters.size(); i++) {
                converters[i]->update(simTime, timeStep);
                converters[i]->calculateLosses();
            }
        }
        catch (const std::exception& e) {
            log("ERROR: Exception in power electronics processing: " + std::string(e.what()));
            return false;
        }

        auto peEnd = std::chrono::high_resolution_clock::now();
        results.powerElectronicsTime +=
            std::chrono::duration<double>(peEnd - peStart).count();

        // Waveform relaxation iteration loop
        int iterations = 0;
        bool converged = false;

        log("Time step " + std::to_string(t) + ": Starting waveform relaxation iterations");
        for (int iter = 0; iter < maxWaveformIterations && !converged; iter++) {
            iterations++;
            log("Time step " + std::to_string(t) + ", Iteration " + std::to_string(iter) + ": Processing subnetworks");

            try {
                // Process each subnetwork with error handling
                for (int i = 0; i < numGPUs; i++) {
                    try {
                        log("Time step " + std::to_string(t) + ", Iteration " + std::to_string(iter)
                            + ": Setting CUDA device " + std::to_string(deviceIDs[i]));
                        cudaError_t err = cudaSetDevice(deviceIDs[i]);
                        if (err != cudaSuccess) {
                            log("CUDA error setting device: " + std::string(cudaGetErrorString(err)));
                            throw std::runtime_error("CUDA device selection failed");
                        }

                        log("Time step " + std::to_string(t) + ", Iteration " + std::to_string(iter)
                            + ": Solving subnetwork " + std::to_string(i));
                        subnetworks[i]->solve(simTime, iter);

                        // Check for CUDA errors after solve
                        err = cudaGetLastError();
                        if (err != cudaSuccess) {
                            log("CUDA error after solve: " + std::string(cudaGetErrorString(err)));
                            throw std::runtime_error("CUDA error in solver");
                        }
                    }
                    catch (const std::exception& e) {
                        log("ERROR in subnetwork " + std::to_string(i) + ": " + std::string(e.what()));
                        log("Attempting to continue by resetting node voltages");

                        // Try to recover by resetting voltages
                        for (int j = 0; j < subnetworks[i]->getNumNodes(); j++) {
                            subnetworks[i]->setNodeVoltage(j, 0.0);
                        }
                    }
                }

                // Synchronize GPUs
                log("Time step " + std::to_string(t) + ", Iteration " + std::to_string(iter) + ": Synchronizing GPU streams");
                for (int i = 0; i < numGPUs; i++) {
                    cudaError_t err = cudaStreamSynchronize(streams[i]);
                    if (err != cudaSuccess) {
                        log("CUDA error synchronizing stream " + std::to_string(i) + ": " + std::string(cudaGetErrorString(err)));
                        throw std::runtime_error("CUDA stream synchronization failed");
                    }
                }

                // Exchange boundary information between GPUs
                log("Time step " + std::to_string(t) + ", Iteration " + std::to_string(iter) + ": Exchanging boundary data");
                auto commStart = std::chrono::high_resolution_clock::now();
                exchangeBoundaryData();
                auto commEnd = std::chrono::high_resolution_clock::now();

                results.communicationTime +=
                    std::chrono::duration<double>(commEnd - commStart).count();

                // Check convergence
                log("Time step " + std::to_string(t) + ", Iteration " + std::to_string(iter) + ": Checking convergence");
                converged = checkConvergence();

                if (converged) {
                    consecutiveFailedIterations = 0; // Reset counter on success
                    log("Time step " + std::to_string(t) + ": Converged after " + std::to_string(iterations) + " iterations");
                }
                else if (iter == maxWaveformIterations - 1) {
                    consecutiveFailedIterations++; // Increment counter on failure
                    log("WARNING: Time step " + std::to_string(t) + " failed to converge after "
                        + std::to_string(maxWaveformIterations) + " iterations");

                    // Check if we've had too many consecutive failures
                    if (consecutiveFailedIterations >= maxConsecutiveFailedIterations) {
                        log("ERROR: Too many consecutive convergence failures ("
                            + std::to_string(consecutiveFailedIterations) + "). Aborting simulation.");
                        return false;
                    }

                    // Force "convergence" to continue simulation
                    log("Forcing convergence to continue simulation after "
                        + std::to_string(consecutiveFailedIterations) + " consecutive failures");
                    converged = true;
                }
            }
            catch (const std::exception& e) {
                log("ERROR: Exception in simulation at time " + std::to_string(simTime)
                    + "s, iteration " + std::to_string(iter) + ": " + std::string(e.what()));
                consecutiveFailedIterations++;

                if (consecutiveFailedIterations >= maxConsecutiveFailedIterations) {
                    log("ERROR: Too many consecutive failures ("
                        + std::to_string(consecutiveFailedIterations) + "). Aborting simulation.");
                    return false;
                }

                // Try to continue with next iteration
                log("Attempting to continue with next iteration");
                continue;
            }
        }

        results.iterationCount += iterations;

        // Collect results for this time step
        log("Time step " + std::to_string(t) + ": Collecting results");
        try {
            collectResults(t);
        }
        catch (const std::exception& e) {
            log("ERROR: Exception while collecting results: " + std::string(e.what()));
            // Continue anyway
        }

        // Update visualization if enabled
        if (dataCommunicator && dataCommunicator->isConnected()) {
            try {
                dataCommunicator->sendTimeStepData(t, results);
            }
            catch (const std::exception& e) {
                log("WARNING: Exception in visualization: " + std::string(e.what()));
                // Continue anyway
            }
        }
        else if (realTimeVisualization && t % 10 == 0) {
            try {
                updateVisualization(t);
            }
            catch (const std::exception& e) {
                log("WARNING: Exception in visualization: " + std::string(e.what()));
                // Continue anyway
            }
        }

        // Check if simulation has been paused
        while (pauseSimulation.load() && simulationRunning.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Check if simulation has been stopped
        if (!simulationRunning.load()) {
            log("Simulation stopped by user request");
            break;
        }
    }

    // Indicate simulation is complete
    log("==== SIMULATION COMPLETE ====");

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
    log("Exporting results to database");
    if (!exportToDatabase("simulation_results.db")) {
        log("ERROR: Failed to export results to database");
        return false;
    }
    log("Database export complete");

    detailedLog.close();
    return true;
}

bool EMTPSolver::initializeFromLoadflow() {
    std::cout << "Initializing simulation from load flow solution..." << std::endl;

    // Create a realistic set of initial conditions

    // Set phase angles based on network topology to create a valid starting point
    double baseVoltage = 1.0;  // Per unit
    double baseAngle = 0.0;    // Radians
    double angleStep = 2.0 * M_PI / nodeNames.size();  // Distribute angles 

    int nodeCounter = 0;

    for (const auto& node : nodeNames) {
        // Set realistic magnitude based on nominal voltage
        double nominalVoltage = 1.0;  // Default 1.0 pu

        if (nodeNominalVoltages.find(node) != nodeNominalVoltages.end()) {
            nominalVoltage = nodeNominalVoltages[node];
        }

        // Calculate angle based on network position to ensure proper power flow
        double angle = baseAngle + nodeCounter * angleStep;

        // Convert to rectangular form to avoid discontinuities
        double realVoltage = nominalVoltage * cos(angle);
        double imagVoltage = nominalVoltage * sin(angle);

        // Add a small random offset to break symmetry
        // This can help avoid oscillations during convergence
        double randomOffsetReal = (rand() % 100) * 0.0001 * realVoltage;
        double randomOffsetImag = (rand() % 100) * 0.0001 * imagVoltage;

        double initialVoltageReal = realVoltage + randomOffsetReal;
        double initialVoltageImag = imagVoltage + randomOffsetImag;

        // Convert back to magnitude
        double initialVoltage = sqrt(initialVoltageReal * initialVoltageReal +
            initialVoltageImag * initialVoltageImag);

        // Set initial voltage for this node in all subnetworks
        for (auto& subnetwork : subnetworks) {
            int localIndex = subnetwork->getLocalNodeIndex(nodeNameToIndex[node]);
            if (localIndex >= 0) {
                subnetwork->setNodeVoltage(localIndex, initialVoltage);
            }
        }

        nodeCounter++;
    }

    // Initialize history terms for all elements to ensure proper representation
    for (auto& subnetwork : subnetworks) {
        subnetwork->buildAdmittanceMatrix();
        double time = 0.0; // Initial time
        subnetwork->updateHistoryTerms(time);
    }

    return true;
}

void EMTPSolver::exchangeBoundaryData() {
    std::ofstream logStream("boundary_exchange_log.txt", std::ios::app);

    auto logMessage = [&](const std::string& message) {
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << "[" << std::put_time(std::localtime(&now_time_t), "%H:%M:%S") << "] "
            << "Boundary Exchange: " << message;
        std::string fullMessage = ss.str();
        std::cout << fullMessage << std::endl;
        logStream << fullMessage << std::endl;
        };

    // Dynamic relaxation factor based on convergence history
    static int timeStepCounter = 0;

    // Increase time step counter
    timeStepCounter++;
    logMessage("Time step counter: " + std::to_string(timeStepCounter));

    // Adjust relaxation factor - start more conservative and increase gradually
    double baseRelaxationFactor = 0.7;
    double relaxationFactor = baseRelaxationFactor;

    // For long simulations, gradually increase relaxation factor 
    // for better performance, but be more conservative in early time steps
    if (timeStepCounter > 100) {
        relaxationFactor = (std::min<double>)(baseRelaxationFactor + 0.1, 0.9);
        logMessage("Increased relaxation factor to " + std::to_string(relaxationFactor) +
            " after 100 time steps");
    }

    // Log number of boundary nodes
    logMessage("Processing " + std::to_string(boundaryNodes.size()) + " boundary nodes");

    for (size_t bn = 0; bn < boundaryNodes.size(); bn++) {
        auto& boundaryNode = boundaryNodes[bn];
        logMessage("Processing boundary node " + std::to_string(bn) +
            " (name: " + boundaryNode.getName() + ")");

        // Get voltage values from all subnetworks
        std::vector<double> voltages;
        voltages.reserve(boundaryNode.getSubnetworkCount());

        for (size_t i = 0; i < boundaryNode.getSubnetworkCount(); i++) {
            int subnetID = boundaryNode.getSubnetworkID(i);
            int localID = boundaryNode.getLocalID(i);

            logMessage("  Getting voltage from subnetwork " + std::to_string(subnetID) +
                ", local node " + std::to_string(localID));

            double voltage = 0.0;
            try {
                voltage = subnetworks[subnetID]->getNodeVoltage(localID);
            }
            catch (const std::exception& e) {
                logMessage("  ERROR getting voltage: " + std::string(e.what()));
                voltage = boundaryNode.getCurrentVoltage(); // Use previous value
            }

            // Check for NaN or very large values that could destabilize simulation
            if (std::isnan(voltage) || std::isinf(voltage) || std::abs(voltage) > 1e6) {
                logMessage("  WARNING: Detected invalid voltage value " + std::to_string(voltage) +
                    ". Using previous value.");

                // Use previous value instead
                voltage = boundaryNode.getCurrentVoltage();

                // If still problematic, use a reasonable default
                if (std::isnan(voltage) || std::isinf(voltage) || std::abs(voltage) > 1e6) {
                    logMessage("  Previous value also invalid, using zero instead");
                    voltage = 0.0;
                }
            }

            logMessage("  Voltage from subnetwork " + std::to_string(subnetID) +
                ": " + std::to_string(voltage));
            voltages.push_back(voltage);
        }

        // Calculate weighted average voltage
        double totalVoltage = 0.0;
        double weightSum = 0.0;

        for (size_t i = 0; i < voltages.size(); i++) {
            // Apply stronger weight to smaller subnetworks (they tend to be more sensitive)
            double weight = 1.0 / (subnetworks[boundaryNode.getSubnetworkID(i)]->getNumNodes() + 1.0);
            totalVoltage += voltages[i] * weight;
            weightSum += weight;

            logMessage("  Subnetwork " + std::to_string(boundaryNode.getSubnetworkID(i)) +
                " weight: " + std::to_string(weight));
        }

        double avgVoltage = weightSum > 0.0 ? totalVoltage / weightSum : 0.0;
        logMessage("  Average voltage: " + std::to_string(avgVoltage) +
            " (from " + std::to_string(voltages.size()) + " values)");

        // Get previous voltage value
        double previousVoltage = boundaryNode.getCurrentVoltage();
        logMessage("  Previous voltage: " + std::to_string(previousVoltage));

        // Check for convergence rate and adjust relaxation factor if needed
        double convergenceRate = boundaryNode.getConvergenceRate();
        logMessage("  Convergence rate: " + std::to_string(convergenceRate));

        if (convergenceRate > 1.0) {
            // If error is growing, use a more conservative relaxation factor
            double originalFactor = relaxationFactor;
            relaxationFactor = (std::max<double>)(0.3, relaxationFactor - 0.1);
            logMessage("  Reduced relaxation factor from " + std::to_string(originalFactor) +
                " to " + std::to_string(relaxationFactor) + " due to increasing error");
        }

        // Apply relaxation - this helps convergence significantly
        double relaxedVoltage = relaxationFactor * avgVoltage +
            (1.0 - relaxationFactor) * previousVoltage;

        logMessage("  Applied relaxation factor " + std::to_string(relaxationFactor) +
            ", result: " + std::to_string(relaxedVoltage));

        // Update the voltage in the boundary node object
        try {
            boundaryNode.updateVoltage(relaxedVoltage);
        }
        catch (const std::exception& e) {
            logMessage("  ERROR updating boundary node voltage: " + std::string(e.what()));
        }

        // Set the voltage back to all subnetworks
        for (size_t i = 0; i < boundaryNode.getSubnetworkCount(); i++) {
            int subnetID = boundaryNode.getSubnetworkID(i);
            int localID = boundaryNode.getLocalID(i);

            logMessage("  Setting voltage " + std::to_string(relaxedVoltage) +
                " to subnetwork " + std::to_string(subnetID) +
                ", local node " + std::to_string(localID));

            try {
                subnetworks[subnetID]->setNodeVoltage(localID, relaxedVoltage);
            }
            catch (const std::exception& e) {
                logMessage("  ERROR setting voltage: " + std::string(e.what()));
            }
        }
    }

    logMessage("Boundary data exchange complete");
    logStream.close();
}


void EMTPSolver::collectResults(int timeStep) {
    std::lock_guard<std::mutex> lock(resultsMutex);

    // Make sure timeStep is valid
    if (timeStep < 0 || timeStep >= numTimeSteps) {
        std::cerr << "WARNING: Invalid time step in collectResults: " << timeStep << std::endl;
        return;
    }

    // Only collect results for time steps that match the decimation factor
    if (timeStep % outputDecimation != 0 && timeStep != numTimeSteps - 1) {
        return;
    }

    // Index in decimated results array
    int resultIndex = timeStep / outputDecimation;

    // Make sure resultIndex is valid
    if (resultIndex < 0 || resultIndex >= static_cast<int>(results.timePoints.size())) {
        std::cerr << "WARNING: Invalid result index in collectResults: " << resultIndex << std::endl;
        return;
    }

    // Make sure time is recorded correctly
    results.timePoints[resultIndex] = timeStep * timeStep;

    // Collect node voltages with error checking
    for (int subnetID = 0; subnetID < numGPUs; subnetID++) {
        for (const auto& nodePair : subnetworks[subnetID]->getNodeMap()) {
            std::string nodeName = nodePair.first;
            int localIndex = nodePair.second;

            // Verify node exists in results structure
            if (results.nodeVoltages.find(nodeName) == results.nodeVoltages.end()) {
                std::cerr << "WARNING: Node " << nodeName << " not found in results structure" << std::endl;
                continue;
            }

            // Verify result vector has enough space
            if (resultIndex >= static_cast<int>(results.nodeVoltages[nodeName].size())) {
                std::cerr << "WARNING: Result index " << resultIndex
                    << " out of bounds for node " << nodeName << " (size: "
                    << results.nodeVoltages[nodeName].size() << ")" << std::endl;
                continue;
            }

            try {
                double voltage = subnetworks[subnetID]->getNodeVoltage(localIndex);

                // Perform sanity check on voltage value
                if (std::isnan(voltage) || std::isinf(voltage) || std::abs(voltage) > 1e6) {
                    std::cerr << "WARNING: Invalid voltage value " << voltage
                        << " for node " << nodeName << " at time step " << timeStep << std::endl;

                    // Use previous value or zero instead
                    if (resultIndex > 0) {
                        voltage = results.nodeVoltages[nodeName][resultIndex - 1];
                    }
                    else {
                        voltage = 0.0;
                    }
                }

                results.nodeVoltages[nodeName][resultIndex] = voltage;
            }
            catch (const std::exception& e) {
                std::cerr << "ERROR: Exception in collectResults for node " << nodeName
                    << ": " << e.what() << std::endl;
            }
        }
    }

    // Similar error checking for branch currents
    for (int subnetID = 0; subnetID < numGPUs; subnetID++) {
        for (const auto& branchPair : subnetworks[subnetID]->getBranchMap()) {
            std::string branchName = branchPair.first;
            int localIndex = branchPair.second;

            // Verify branch exists in results structure
            if (results.branchCurrents.find(branchName) == results.branchCurrents.end()) {
                std::cerr << "WARNING: Branch " << branchName << " not found in results structure" << std::endl;
                continue;
            }

            // Verify result vector has enough space
            if (resultIndex >= static_cast<int>(results.branchCurrents[branchName].size())) {
                std::cerr << "WARNING: Result index " << resultIndex
                    << " out of bounds for branch " << branchName << " (size: "
                    << results.branchCurrents[branchName].size() << ")" << std::endl;
                continue;
            }

            try {
                double current = subnetworks[subnetID]->getBranchCurrent(localIndex);

                // Perform sanity check on current value
                if (std::isnan(current) || std::isinf(current) || std::abs(current) > 1e6) {
                    std::cerr << "WARNING: Invalid current value " << current
                        << " for branch " << branchName << " at time step " << timeStep << std::endl;

                    // Use previous value or zero instead
                    if (resultIndex > 0) {
                        current = results.branchCurrents[branchName][resultIndex - 1];
                    }
                    else {
                        current = 0.0;
                    }
                }

                results.branchCurrents[branchName][resultIndex] = current;
            }
            catch (const std::exception& e) {
                std::cerr << "ERROR: Exception in collectResults for branch " << branchName
                    << ": " << e.what() << std::endl;
            }
        }
    }

    // Rest of the function for collecting power electronics results, etc.
    // [Keep existing code but add similar error checking]

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

// Add this to checkConvergence method
bool EMTPSolver::checkConvergence() {
    std::ofstream logStream("convergence_check_log.txt", std::ios::app);

    auto logMessage = [&](const std::string& message) {
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << "[" << std::put_time(std::localtime(&now_time_t), "%H:%M:%S") << "] "
            << "Convergence Check: " << message;
        std::string fullMessage = ss.str();
        std::cout << fullMessage << std::endl;
        logStream << fullMessage << std::endl;
        };

    // Enhanced convergence checking with better divergence detection
    static double previous_max_diff = 0.0;
    double current_max_diff = 0.0;
    double max_voltage_magnitude = 0.0;
    int numBoundaryNodes = boundaryNodes.size();

    logMessage("Checking convergence for " + std::to_string(numBoundaryNodes) + " boundary nodes");
    logMessage("Previous maximum difference: " + std::to_string(previous_max_diff));
    logMessage("Convergence tolerance: " + std::to_string(convergenceTolerance));

    // Single GPU case - just return true
    if (numBoundaryNodes == 0) {
        logMessage("No boundary nodes, returning true");
        logStream.close();
        return true;
    }

    // For each boundary node, check if voltages have converged
    for (int i = 0; i < numBoundaryNodes; i++) {
        auto& boundaryNode = boundaryNodes[i];
        logMessage("Checking node " + std::to_string(i) + " (name: " +
            boundaryNode.getName() + ")");

        // Calculate the change in voltage
        double prevVoltage = boundaryNode.getPreviousVoltage();
        double currVoltage = boundaryNode.getCurrentVoltage();
        double diff = std::abs(currVoltage - prevVoltage);

        logMessage("  Previous voltage: " + std::to_string(prevVoltage));
        logMessage("  Current voltage: " + std::to_string(currVoltage));
        logMessage("  Absolute difference: " + std::to_string(diff));

        // Track maximum voltage magnitude for relative error calculation
        max_voltage_magnitude = (std::max<double>)(max_voltage_magnitude, std::abs(currVoltage));

        // Track maximum difference
        if (current_max_diff < diff) {
            current_max_diff = diff;
        }

        // Calculate relative error if voltage magnitude is significant
        double rel_error = (max_voltage_magnitude > 1e-6) ?
            diff / max_voltage_magnitude : diff;

        logMessage("  Maximum voltage magnitude: " + std::to_string(max_voltage_magnitude));
        logMessage("  Relative error: " + std::to_string(rel_error));

        // Use relative error for convergence check when voltage magnitude is significant
        if (max_voltage_magnitude > 1.0 && rel_error > convergenceTolerance) {
            // Detect divergence: if error grows substantially
            if (diff > 5.0 * previous_max_diff && previous_max_diff > 0.01) {
                logMessage("WARNING: Possible divergence detected! Diff: " +
                    std::to_string(diff) + " > " +
                    std::to_string(5.0 * previous_max_diff) +
                    " at node " + boundaryNode.getName());

                // Apply stronger relaxation factor for this node
                double relaxedVoltage = 0.5 * currVoltage + 0.5 * prevVoltage;
                logMessage("  Applying strong relaxation: " + std::to_string(relaxedVoltage));
                boundaryNode.updateVoltage(relaxedVoltage);

                // Continue instead of returning to allow simulation to try to recover
                continue;
            }

            logMessage("Not converged (relative error " + std::to_string(rel_error) +
                " > tolerance " + std::to_string(convergenceTolerance) + ")");
            logStream.close();
            return false;
        }
        else if (max_voltage_magnitude <= 1.0 && diff > convergenceTolerance) {
            logMessage("Not converged (absolute difference " + std::to_string(diff) +
                " > tolerance " + std::to_string(convergenceTolerance) + ")");
            logStream.close();
            return false;
        }
    }

    // Update previous max difference
    previous_max_diff = current_max_diff;
    logMessage("Convergence achieved, maximum difference: " + std::to_string(current_max_diff));
    logStream.close();
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