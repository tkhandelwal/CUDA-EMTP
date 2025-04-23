#include "emtp_solver.h"
#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <thread>
#include <fstream>
#include <iomanip>

// Helper function to log to both console and file
void log(const std::string& message, std::ofstream& logFile) {
    auto now = std::chrono::system_clock::now();
    auto currentTime = std::chrono::system_clock::to_time_t(now);

    std::string timestamp = std::ctime(&currentTime);
    timestamp = timestamp.substr(0, timestamp.length() - 1);  // Remove trailing newline

    std::string formattedMessage = "[" + timestamp + "] " + message;
    std::cout << formattedMessage << std::endl;
    logFile << formattedMessage << std::endl;
    logFile.flush();  // Ensure it's written immediately, helpful in case of crashes
}

void printUsage(std::ofstream& logFile) {
    std::string usage = "Usage: emtp_solver [OPTIONS]\n"
        "Options:\n"
        "  -i, --input FILE       Input network file (default: example_network.net)\n"
        "  -t, --time-step VALUE  Simulation time step in seconds (default: 20e-6)\n"
        "  -e, --end-time VALUE   Simulation end time in seconds (default: 0.1)\n"
        "  -nv, --no-viz          Disable real-time visualization\n"
        "  -db, --database FILE   Output results to SQLite database file (default: simulation_results.db)\n"
        "  -web, --web-viz        Enable web-based visualization\n"
        "  -s, --server ADDRESS   Server address for web visualization (default: localhost)\n"
        "  -p, --port NUMBER      Port number for web visualization (default: 5555)\n"
        "  -h, --help             Display this help message\n"
        "  -check-cuda            Check CUDA environment and exit\n";

    log(usage, logFile);
}

bool checkCudaEnvironment(std::ofstream& logFile) {
    log("Checking CUDA environment...", logFile);

    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        log("ERROR: Failed to get CUDA device count: " + std::string(cudaGetErrorString(error)), logFile);
        return false;
    }

    log("Found " + std::to_string(deviceCount) + " CUDA device(s)", logFile);

    if (deviceCount == 0) {
        log("ERROR: No CUDA devices found. This application requires CUDA.", logFile);
        return false;
    }

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaError_t propError = cudaGetDeviceProperties(&prop, i);

        if (propError != cudaSuccess) {
            log("ERROR: Failed to get properties for device " + std::to_string(i) +
                ": " + std::string(cudaGetErrorString(propError)), logFile);
            continue;
        }

        // Log device capabilities
        std::stringstream ss;
        ss << "Device " << i << ": " << prop.name
            << " (Compute Capability " << prop.major << "." << prop.minor << ")";
        log(ss.str(), logFile);

        ss.str("");
        ss << "  - Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB";
        log(ss.str(), logFile);

        ss.str("");
        ss << "  - Multiprocessors: " << prop.multiProcessorCount;
        log(ss.str(), logFile);

        ss.str("");
        ss << "  - Max Threads Per Block: " << prop.maxThreadsPerBlock;
        log(ss.str(), logFile);

        ss.str("");
        ss << "  - Max Threads Dimensions: [" << prop.maxThreadsDim[0] << ", "
            << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]";
        log(ss.str(), logFile);

        ss.str("");
        ss << "  - Max Grid Size: [" << prop.maxGridSize[0] << ", "
            << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]";
        log(ss.str(), logFile);
    }

    // Test a simple CUDA operation
    log("Testing CUDA memory operations...", logFile);

    float* d_test = nullptr;
    cudaError_t allocError = cudaMalloc(&d_test, sizeof(float));

    if (allocError != cudaSuccess) {
        log("ERROR: CUDA memory allocation failed: " + std::string(cudaGetErrorString(allocError)), logFile);
        return false;
    }

    // Try to copy some data
    float h_test = 3.14159f;
    cudaError_t copyError = cudaMemcpy(d_test, &h_test, sizeof(float), cudaMemcpyHostToDevice);

    if (copyError != cudaSuccess) {
        log("ERROR: CUDA memory copy failed: " + std::string(cudaGetErrorString(copyError)), logFile);
        cudaFree(d_test);
        return false;
    }

    // Free the memory
    cudaError_t freeError = cudaFree(d_test);

    if (freeError != cudaSuccess) {
        log("ERROR: CUDA memory free failed: " + std::string(cudaGetErrorString(freeError)), logFile);
        return false;
    }

    log("CUDA environment check passed successfully", logFile);
    return true;
}

int main(int argc, char** argv) {
    // Open log file
    std::ofstream logFile("emtp_debug.log", std::ios::out | std::ios::app);

    if (!logFile.is_open()) {
        std::cerr << "WARNING: Could not open log file. Continuing without file logging." << std::endl;
    }

    log("====== EMTP-CUDA-V2 Session Started ======", logFile);

    try {
        // Default parameters
        std::string inputFile = "example_network.net";  // Default to example network file
        std::string databaseFile = "simulation_results.db"; // Default database file
        double timeStep = 20e-6;  // 20μs time step
        double endTime = 0.1;     // 0.1s simulation
        bool enableVisualization = true;
        bool useRealTimeWeb = false;
        std::string serverAddress = "localhost";
        int serverPort = 5555;
        bool checkCudaOnly = false;

        // Parse command line arguments
        log("Parsing command line arguments...", logFile);
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            log("Processing argument: " + arg, logFile);

            if (arg == "-i" || arg == "--input") {
                if (i + 1 < argc) {
                    inputFile = argv[++i];
                    log("Input file set to: " + inputFile, logFile);
                }
                else {
                    log("Error: -i/--input requires a filename", logFile);
                    printUsage(logFile);
                    return 1;
                }
            }
            else if (arg == "-db" || arg == "--database") {
                if (i + 1 < argc) {
                    databaseFile = argv[++i];
                    log("Database file set to: " + databaseFile, logFile);
                }
                else {
                    log("Error: -db/--database requires a filename", logFile);
                    printUsage(logFile);
                    return 1;
                }
            }
            else if (arg == "-t" || arg == "--time-step") {
                if (i + 1 < argc) {
                    try {
                        timeStep = std::stod(argv[++i]);
                        if (timeStep <= 0) {
                            throw std::out_of_range("Time step must be positive");
                        }
                        log("Time step set to: " + std::to_string(timeStep), logFile);
                    }
                    catch (const std::exception& e) {
                        log("Error parsing time step: " + std::string(e.what()), logFile);
                        printUsage(logFile);
                        return 1;
                    }
                }
                else {
                    log("Error: -t/--time-step requires a value", logFile);
                    printUsage(logFile);
                    return 1;
                }
            }
            else if (arg == "-e" || arg == "--end-time") {
                if (i + 1 < argc) {
                    try {
                        endTime = std::stod(argv[++i]);
                        if (endTime <= 0) {
                            throw std::out_of_range("End time must be positive");
                        }
                        log("End time set to: " + std::to_string(endTime), logFile);
                    }
                    catch (const std::exception& e) {
                        log("Error parsing end time: " + std::string(e.what()), logFile);
                        printUsage(logFile);
                        return 1;
                    }
                }
                else {
                    log("Error: -e/--end-time requires a value", logFile);
                    printUsage(logFile);
                    return 1;
                }
            }
            else if (arg == "-nv" || arg == "--no-viz") {
                enableVisualization = false;
                log("Visualization disabled", logFile);
            }
            else if (arg == "-web" || arg == "--web-viz") {
                useRealTimeWeb = true;
                log("Web visualization enabled", logFile);
            }
            else if (arg == "-s" || arg == "--server") {
                if (i + 1 < argc) {
                    serverAddress = argv[++i];
                    log("Server address set to: " + serverAddress, logFile);
                }
                else {
                    log("Error: -s/--server requires an address", logFile);
                    printUsage(logFile);
                    return 1;
                }
            }
            else if (arg == "-p" || arg == "--port") {
                if (i + 1 < argc) {
                    try {
                        serverPort = std::stoi(argv[++i]);
                        if (serverPort <= 0 || serverPort > 65535) {
                            throw std::out_of_range("Port must be between 1 and 65535");
                        }
                        log("Server port set to: " + std::to_string(serverPort), logFile);
                    }
                    catch (const std::exception& e) {
                        log("Error parsing port number: " + std::string(e.what()), logFile);
                        printUsage(logFile);
                        return 1;
                    }
                }
                else {
                    log("Error: -p/--port requires a value", logFile);
                    printUsage(logFile);
                    return 1;
                }
            }
            else if (arg == "-h" || arg == "--help") {
                printUsage(logFile);
                return 0;
            }
            else if (arg == "-check-cuda") {
                checkCudaOnly = true;
                log("CUDA check only mode enabled", logFile);
            }
            else {
                log("Error: Unknown option: " + arg, logFile);
                printUsage(logFile);
                return 1;
            }
        }

        // Check CUDA environment
        if (!checkCudaEnvironment(logFile)) {
            log("CUDA environment check failed", logFile);
            return 1;
        }

        if (checkCudaOnly) {
            log("CUDA check completed successfully. Exiting as requested.", logFile);
            return 0;
        }

        // Validate input file
        log("Validating input file: " + inputFile, logFile);
        if (!std::ifstream(inputFile).good()) {
            log("Error: Input file does not exist: " + inputFile, logFile);
            return 1;
        }
        log("Input file exists", logFile);

        // Print banner
        log("===============================================", logFile);
        log("   EMTP Solver with CUDA Multi-GPU Support    ", logFile);
        log("===============================================", logFile);
        log("Input file: " + inputFile, logFile);
        log("Database file: " + databaseFile, logFile);
        log("Time step: " + std::to_string(timeStep) + " seconds", logFile);
        log("End time: " + std::to_string(endTime) + " seconds", logFile);
        log("Visualization: " + std::string(enableVisualization ? "Enabled" : "Disabled"), logFile);
        log("Web visualization: " + std::string(useRealTimeWeb ? "Enabled" : "Disabled"), logFile);
        if (useRealTimeWeb) {
            log("Server address: " + serverAddress + ":" + std::to_string(serverPort), logFile);
        }
        log("===============================================", logFile);

        try {
            // Create EMTP solver
            log("Creating EMTP solver instance...", logFile);
            EMTPSolver solver(timeStep, endTime, enableVisualization);
            log("EMTP solver instance created successfully", logFile);

            // Load network from example_network.net
            log("Loading network from: " + inputFile, logFile);
            if (!solver.loadNetwork(inputFile)) {
                log("Failed to load network from: " + inputFile, logFile);
                return 1;
            }
            log("Network loaded successfully from " + inputFile, logFile);

            // Run simulation
            log("Starting simulation...", logFile);
            auto startTime = std::chrono::high_resolution_clock::now();

            log("Calling solver.runSimulation(useRealTimeWeb)...", logFile);
            if (!solver.runSimulation(useRealTimeWeb)) {
                log("Simulation failed", logFile);
                return 1;
            }

            auto endTimePoint = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTime);

            log("Simulation completed in " + std::to_string(duration.count() / 1000.0) + " seconds", logFile);

            // Export to database
            log("Exporting to database: " + databaseFile, logFile);
            if (!solver.exportToDatabase(databaseFile)) {
                log("Failed to export results to database: " + databaseFile, logFile);
                return 1;
            }
            log("Database export completed successfully", logFile);

            log("Simulation completed successfully", logFile);

            // Keep the visualization window open if enabled
            if (enableVisualization && !useRealTimeWeb) {
                log("Press Ctrl+C to exit...", logFile);
                while (true) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    if (!solver.isRunning()) {
                        break;
                    }
                }
            }
        }
        catch (const std::exception& e) {
            log("ERROR: Exception caught in main simulation block: " + std::string(e.what()), logFile);
            return 1;
        }
        catch (...) {
            log("ERROR: Unknown exception caught in main simulation block", logFile);
            return 1;
        }
    }
    catch (const std::exception& e) {
        log("CRITICAL ERROR: Exception caught in main function: " + std::string(e.what()), logFile);
        return 1;
    }
    catch (...) {
        log("CRITICAL ERROR: Unknown exception caught in main function", logFile);
        return 1;
    }

    log("====== EMTP-CUDA-V2 Session Completed Successfully ======", logFile);

    // Wait for user input before closing
    log("Press Enter to exit...", logFile);
    std::cin.get();

    return 0;
}