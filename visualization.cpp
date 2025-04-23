#include "visualization.h"
#include "simulation_results.h"
#include "data_communicator.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <thread>


Visualization::Visualization(int width, int height)
    : windowWidth(width),
    windowHeight(height),
    running(false),
    dataUpdated(false),
    currentTimeStep(0),
    totalTimeSteps(0),
    simulationProgress(0.0) {
}

Visualization::~Visualization() {
    shutdown();
}

bool Visualization::initialize(const std::vector<std::string>& nodes,
    const std::vector<double>& times,
    const std::unordered_map<std::string, std::pair<float, float>>& positions,
    const std::vector<std::pair<std::string, std::string>>& branchList) {

    std::cout << "Initializing visualization system..." << std::endl;

    // Store input data
    nodeNames = nodes;
    timePoints = times;
    nodePositions = positions;
    branches = branchList;
    totalTimeSteps = static_cast<int>(timePoints.size());

    // Extract branch names from branch list
    for (const auto& branch : branches) {
        std::string branchName = branch.first + "_" + branch.second;
        branchNames.push_back(branchName);
    }

    // Initialize data structures
    for (const auto& node : nodeNames) {
        nodeVoltages[node] = std::vector<double>(totalTimeSteps, 0.0);
        selectedNodes.push_back(node);  // Select all nodes by default
    }

    for (const auto& branch : branchNames) {
        branchCurrents[branch] = std::vector<double>(totalTimeSteps, 0.0);
        selectedBranches.push_back(branch);  // Select all branches by default
    }

    // Initialize data communicator
    try {
        dataCommunicator = std::make_unique<DataCommunicator>("localhost", 5555);
        if (!dataCommunicator->connect()) {
            std::cerr << "Warning: Could not connect to visualization server. Continuing without real-time visualization." << std::endl;
        }
        else {
            // Send initial data
            dataCommunicator->sendInitialData(
                SimulationResults{ timePoints, nodeVoltages, branchCurrents },
                nodeNames,
                nodePositions,
                branches
            );
            std::cout << "Connected to visualization server and sent initial data." << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error initializing data communicator: " << e.what() << std::endl;
    }

    // Start the visualization thread
    running = true;
    renderThread = std::thread(&Visualization::renderLoop, this);

    return true;
}

void Visualization::shutdown() {
    if (running.load()) {
        running.store(false);
        if (renderThread.joinable()) {
            renderThread.join();
        }
    }

    if (dataCommunicator) {
        dataCommunicator->disconnect();
        dataCommunicator.reset();
    }

    std::cout << "Visualization system shut down." << std::endl;
}

void Visualization::update(const SimulationResults& results, int timeStep) {
    std::lock_guard<std::mutex> lock(dataMutex);

    currentTimeStep = timeStep;
    simulationProgress = static_cast<double>(timeStep) / (totalTimeSteps > 0 ? totalTimeSteps : 1) * 100.0;

    // Copy data from results
    for (const auto& node : nodeNames) {
        if (results.nodeVoltages.find(node) != results.nodeVoltages.end()) {
            const auto& voltages = results.nodeVoltages.at(node);
            if (timeStep < static_cast<int>(voltages.size())) {
                nodeVoltages[node][timeStep] = voltages[timeStep];
            }
        }
    }

    for (const auto& branch : branchNames) {
        if (results.branchCurrents.find(branch) != results.branchCurrents.end()) {
            const auto& currents = results.branchCurrents.at(branch);
            if (timeStep < static_cast<int>(currents.size())) {
                branchCurrents[branch][timeStep] = currents[timeStep];
            }
        }
    }

    // Add power electronics data if available
    for (const auto& converter : results.converterLosses) {
        if (timeStep < static_cast<int>(converter.second.size())) {
            // Just store for now - would display this in a more complete visualization
        }
    }

    dataUpdated.store(true);

    // Send data to visualization server if connected
    if (dataCommunicator && dataCommunicator->isConnected()) {
        try {
            dataCommunicator->sendTimeStepData(timeStep, results);
        }
        catch (const std::exception& e) {
            std::cerr << "Error sending data to visualization server: " << e.what() << std::endl;
        }
    }
}

void Visualization::renderLoop() {
    auto lastUpdateTime = std::chrono::high_resolution_clock::now();
    int frameCount = 0;

    while (running.load()) {
        // Calculate time since last update
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastUpdateTime).count();

        // Update every 500ms (2 fps) to avoid excessive console output
        if (elapsed > 500) {
            std::lock_guard<std::mutex> lock(dataMutex);

            // Only print if data has been updated
            if (dataUpdated.load()) {
                frameCount++;

                // Clear screen (platform-specific)
#ifdef _WIN32
                system("cls");
#else
                system("clear");
#endif

                std::cout << "========== EMTP Simulation Visualization ==========" << std::endl;
                std::cout << "Frame: " << frameCount << " | Progress: "
                    << simulationProgress << "% (Time step: "
                    << currentTimeStep << " of " << totalTimeSteps << ")" << std::endl;

                // Generate progress bar
                std::string progressBar = ConsoleViz::generateProgressBar(simulationProgress / 100.0, 50);
                std::cout << progressBar << std::endl << std::endl;

                // Print some node voltages (up to 5 for clarity)
                std::cout << "Selected Node Voltages:" << std::endl;
                int count = 0;
                for (const auto& node : selectedNodes) {
                    if (count >= 5) break;
                    if (node.empty() || nodeVoltages.find(node) == nodeVoltages.end() ||
                        currentTimeStep >= nodeVoltages[node].size()) continue;

                    std::cout << "  " << node << ": " << nodeVoltages[node][currentTimeStep] << " kV" << std::endl;
                    count++;
                }
                std::cout << std::endl;

                // Print some branch currents (up to 5 for clarity)
                std::cout << "Selected Branch Currents:" << std::endl;
                count = 0;
                for (const auto& branch : selectedBranches) {
                    if (count >= 5) break;
                    if (branch.empty() || branchCurrents.find(branch) == branchCurrents.end() ||
                        currentTimeStep >= branchCurrents[branch].size()) continue;

                    std::cout << "  " << branch << ": " << branchCurrents[branch][currentTimeStep] << " A" << std::endl;
                    count++;
                }
                std::cout << std::endl;

                // Generate an ASCII chart of a selected node voltage over time
                if (!selectedNodes.empty() && nodeVoltages.find(selectedNodes[0]) != nodeVoltages.end()) {
                    std::vector<double> chartData;

                    // Calculate how many history points to show (up to 50)
                    int historyPoints = (50 < (currentTimeStep + 1)) ? 50 : (currentTimeStep + 1);

                    // Calculate start point
                    int startPoint = (0 > (currentTimeStep - historyPoints + 1)) ? 0 : (currentTimeStep - historyPoints + 1);

                    for (int i = startPoint; i <= currentTimeStep; i++) {
                        chartData.push_back(nodeVoltages[selectedNodes[0]][i]);
                    }

                    // Find min and max for scaling
                    double minValue = chartData[0];
                    double maxValue = chartData[0];

                    for (const auto& value : chartData) {
                        if (value < minValue) minValue = value;
                        if (value > maxValue) maxValue = value;
                    }

                    // Add some margin
                    double margin = (maxValue - minValue) * 0.1;
                    minValue -= margin;
                    maxValue += margin;

                    std::cout << "Voltage Plot for " << selectedNodes[0] << ":" << std::endl;
                    std::string chart = ConsoleViz::generateChart(chartData, 60, 10, minValue, maxValue);
                    std::cout << chart << std::endl;
                }

                // Generate simple network diagram if we have positions
                if (!nodePositions.empty() && !branches.empty()) {
                    // Create current voltages map for coloring
                    std::unordered_map<std::string, double> currentVoltages;
                    for (const auto& node : nodeNames) {
                        if (nodeVoltages.find(node) != nodeVoltages.end() &&
                            currentTimeStep < nodeVoltages[node].size()) {
                            currentVoltages[node] = nodeVoltages[node][currentTimeStep];
                        }
                    }

                    std::cout << "Network Diagram:" << std::endl;
                    std::string diagram = ConsoleViz::generateNetworkDiagram(
                        nodePositions, branches, 80, 20, &currentVoltages);
                    std::cout << diagram << std::endl;
                }

                dataUpdated.store(false);
                lastUpdateTime = currentTime;
            }
        }

        // Sleep to prevent high CPU usage
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

void Visualization::setWindowTitle(const std::string& title) {
    // This function is a placeholder since we're not using a GUI window
    std::cout << "Visualization title set to: " << title << std::endl;
}

// Utility functions implementation
std::unordered_map<std::string, std::pair<float, float>> extractNodePositions(const std::string& networkFile) {
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

std::vector<std::pair<std::string, std::string>> extractBranches(const std::string& networkFile) {
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


namespace ConsoleViz {
    std::string generateNetworkDiagram(
        const std::unordered_map<std::string, std::pair<float, float>>& nodePositions,
        const std::vector<std::pair<std::string, std::string>>& branches,
        int width, int height,
        const std::unordered_map<std::string, double>* currentVoltages) {

        // Create an empty grid
        std::vector<std::vector<char>> grid(height, std::vector<char>(width, ' '));
        std::unordered_map<std::string, std::pair<int, int>> screenPositions;

        // Find min/max coordinates using explicit values instead of numeric_limits
        // Use very large/small initial values
        float minX = 1.0e38f;  // Very large positive number
        float minY = 1.0e38f;  // Very large positive number
        float maxX = -1.0e38f; // Very large negative number
        float maxY = -1.0e38f; // Very large negative number

        // First pass: find min and max values
        for (const auto& node : nodePositions) {
            float x = node.second.first;
            float y = node.second.second;

            // Manual min/max comparisons
            if (x < minX) minX = x;
            if (y < minY) minY = y;
            if (x > maxX) maxX = x;
            if (y > maxY) maxY = y;
        }

        // Scale factors (protect against division by zero)
        float scaleX = (maxX > minX) ? ((width - 4) / (maxX - minX)) : 1.0f;
        float scaleY = (maxY > minY) ? ((height - 4) / (maxY - minY)) : 1.0f;

        // Position nodes
        for (const auto& node : nodePositions) {
            int x = static_cast<int>((node.second.first - minX) * scaleX) + 2;
            int y = static_cast<int>((node.second.second - minY) * scaleY) + 2;

            // Stay within bounds
            if (x < 0) x = 0;
            if (x >= width) x = width - 1;
            if (y < 0) y = 0;
            if (y >= height) y = height - 1;

            // Store screen position
            screenPositions[node.first] = std::make_pair(x, y);

            // Draw node
            if (y < height && x < width) {
                grid[y][x] = '+';

                // Add node name (truncated if too long)
                std::string nodeName = node.first;
                if (nodeName.length() > 8) {
                    nodeName = nodeName.substr(0, 8);
                }

                for (size_t i = 0; i < nodeName.length(); i++) {
                    if (x + i + 1 < width) {
                        grid[y][x + i + 1] = nodeName[i];
                    }
                }

                // Add voltage value if provided
                if (currentVoltages != nullptr && currentVoltages->find(node.first) != currentVoltages->end()) {
                    double voltage = currentVoltages->at(node.first);
                    std::string voltageStr = std::to_string(voltage);
                    // Truncate to reasonable length
                    if (voltageStr.length() > 6) {
                        voltageStr = voltageStr.substr(0, 6);
                    }

                    if (y + 1 < height && x < width) {
                        for (size_t i = 0; i < voltageStr.length(); i++) {
                            if (x + i < width) {
                                grid[y + 1][x + i] = voltageStr[i];
                            }
                        }
                    }
                }
            }
        }

        // Draw branches (simple lines)
        for (const auto& branch : branches) {
            if (screenPositions.find(branch.first) != screenPositions.end() &&
                screenPositions.find(branch.second) != screenPositions.end()) {

                auto fromPos = screenPositions[branch.first];
                auto toPos = screenPositions[branch.second];

                // Draw a simple Manhattan path
                int x1 = fromPos.first;
                int y1 = fromPos.second;
                int x2 = toPos.first;
                int y2 = toPos.second;

                // Draw horizontal line
                int xStart = (x1 < x2) ? x1 : x2;  // Instead of std::min
                int xEnd = (x1 > x2) ? x1 : x2;    // Instead of std::max

                for (int x = xStart; x <= xEnd; x++) {
                    if (y1 < height && x < width && grid[y1][x] == ' ') {
                        grid[y1][x] = '-';
                    }
                }

                // Draw vertical line
                int yStart = (y1 < y2) ? y1 : y2;  // Instead of std::min
                int yEnd = (y1 > y2) ? y1 : y2;    // Instead of std::max

                for (int y = yStart; y <= yEnd; y++) {
                    if (y < height && x2 < width && grid[y][x2] == ' ') {
                        grid[y][x2] = '|';
                    }
                }
            }
        }

        // Convert grid to string
        std::string result;
        for (const auto& row : grid) {
            for (const auto& cell : row) {
                result += cell;
            }
            result += '\n';
        }

        return result;
    }

    std::string generateChart(
        const std::vector<double>& values,
        int width, int height,
        double min, double max) {
        // Create an empty grid
        std::vector<std::vector<char>> grid(height, std::vector<char>(width, ' '));

        // Draw axes
        for (int y = 0; y < height; y++) {
            grid[y][0] = '|';
        }
        for (int x = 0; x < width; x++) {
            grid[height - 1][x] = '-';
        }
        grid[height - 1][0] = '+';

        // Draw data points
        if (!values.empty() && min < max) {
            // Scale factor
            double scale = (height - 2) / (max - min);

            // Plot each point
            // Use direct comparison instead of std::min
            int pointCount = (static_cast<int>(values.size()) < (width - 2)) ?
                static_cast<int>(values.size()) : (width - 2);

            double xStep = static_cast<double>(width - 2) / pointCount;

            for (size_t i = 0; i < values.size() && i * xStep < width - 2; i++) {
                double value = values[i];
                int x = static_cast<int>(i * xStep) + 1;
                int y = height - 2 - static_cast<int>((value - min) * scale);

                // Stay within bounds (replace std::max and std::min)
                if (y < 0) y = 0;
                if (y > height - 2) y = height - 2;

                // Draw point
                if (y < height && x < width) {
                    grid[y][x] = '*';
                }
            }
        }

        // Convert grid to string
        std::string result;
        for (const auto& row : grid) {
            for (const auto& cell : row) {
                result += cell;
            }
            result += '\n';
        }

        return result;
    }

    std::string generateProgressBar(double progress, int width) {
        if (width <= 2) return "";

        int filledWidth = static_cast<int>((width - 2) * progress);

        // Replace std::max and std::min with direct comparisons
        if (filledWidth < 0) filledWidth = 0;
        if (filledWidth > width - 2) filledWidth = width - 2;

        std::string result = "[";
        for (int i = 0; i < width - 2; i++) {
            result += (i < filledWidth) ? '=' : ' ';
        }
        result += "]";

        // Add percentage in the middle if there's enough space
        if (width > 7) {
            std::string percent = std::to_string(static_cast<int>(progress * 100)) + "%";
            int startPos = (width - percent.length()) / 2;
            for (size_t i = 0; i < percent.length() && startPos + i < result.length(); i++) {
                result[startPos + i] = percent[i];
            }
        }

        return result;
    } // This closing brace might be missing

} // This closing brace for the namespace might be missing