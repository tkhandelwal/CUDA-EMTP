#pragma once

#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <functional>
#include <chrono>
#include <cmath>

// Forward declarations
class DataCommunicator;
struct SimulationResults;

/**
 * @brief Visualization class for real-time monitoring of EMTP simulation
 */
class Visualization {
private:
    // Configuration
    int windowWidth;
    int windowHeight;

    // Thread synchronization
    std::mutex dataMutex;
    std::atomic<bool> running;
    std::atomic<bool> dataUpdated;
    std::thread renderThread;

    // Data communicator for web-based visualization (already defined elsewhere)
    std::unique_ptr<DataCommunicator> dataCommunicator;

    // Simulation data
    std::vector<std::string> nodeNames;
    std::vector<std::string> branchNames;
    std::vector<double> timePoints;

    // Results data
    std::unordered_map<std::string, std::vector<double>> nodeVoltages;
    std::unordered_map<std::string, std::vector<double>> branchCurrents;
    int currentTimeStep;
    int totalTimeSteps;

    // Network topology for visualization
    std::unordered_map<std::string, std::pair<float, float>> nodePositions;
    std::vector<std::pair<std::string, std::string>> branches;

    // Selection state
    std::vector<std::string> selectedNodes;
    std::vector<std::string> selectedBranches;

    // Performance metrics
    double simulationProgress;

    // Rendering method that runs in a separate thread
    void renderLoop();

    // Helper methods
    void displayStatus();

public:
    /**
     * @brief Constructor
     * @param width Display width (for future terminal-based UI)
     * @param height Display height (for future terminal-based UI)
     */
    Visualization(int width = 80, int height = 24);

    /**
     * @brief Destructor
     */
    ~Visualization();

    /**
     * @brief Initialize the visualization system
     * @param nodes List of node names
     * @param times List of simulation time points
     * @param positions Map of node name to (x,y) position
     * @param branchList List of branch connections (fromNode, toNode)
     * @return True if initialization successful
     */
    bool initialize(const std::vector<std::string>& nodes,
        const std::vector<double>& times,
        const std::unordered_map<std::string, std::pair<float, float>>& positions,
        const std::vector<std::pair<std::string, std::string>>& branchList);

    /**
     * @brief Shut down the visualization system
     */
    void shutdown();

    /**
     * @brief Update visualization with the latest simulation results
     * @param results Current simulation results
     * @param timeStep Current time step index
     */
    void update(const SimulationResults& results, int timeStep);

    /**
     * @brief Set window title - placeholder for future GUI implementation
     * @param title Window title
     */
    void setWindowTitle(const std::string& title);

    /**
     * @brief Check if visualization is still running
     * @return True if running
     */
    bool isRunning() const { return running.load(); }

    /**
     * @brief Select specific nodes for detailed display
     * @param nodes Vector of node names to display
     */
    void selectNodes(const std::vector<std::string>& nodes) {
        std::lock_guard<std::mutex> lock(dataMutex);
        selectedNodes = nodes;
    }

    /**
     * @brief Select specific branches for detailed display
     * @param branches Vector of branch names to display
     */
    void selectBranches(const std::vector<std::string>& branches) {
        std::lock_guard<std::mutex> lock(dataMutex);
        selectedBranches = branches;
    }
};

// Note: These utility functions are already defined in visualization.cpp
// Declaration repeated here for reference
std::unordered_map<std::string, std::pair<float, float>> extractNodePositions(const std::string& networkFile);
std::vector<std::pair<std::string, std::string>> extractBranches(const std::string& networkFile);

// New utility functions for console-based visualization
namespace ConsoleViz {
    /**
     * @brief Generate an ASCII network diagram
     * @param nodePositions Map of node positions
     * @param branches List of branch connections
     * @param width Diagram width
     * @param height Diagram height
     * @param currentVoltages Optional current voltages for coloring
     * @return String containing ASCII diagram
     */
    std::string generateNetworkDiagram(
        const std::unordered_map<std::string, std::pair<float, float>>& nodePositions,
        const std::vector<std::pair<std::string, std::string>>& branches,
        int width, int height,
        const std::unordered_map<std::string, double>* currentVoltages = nullptr);

    /**
     * @brief Generate simple ASCII chart for data visualization
     * @param values Values to display
     * @param width Chart width
     * @param height Chart height
     * @param min Min value for scaling
     * @param max Max value for scaling
     * @return String containing ASCII chart
     */
    std::string generateChart(
        const std::vector<double>& values,
        int width, int height,
        double min, double max);

    /**
     * @brief Generate a progress bar
     * @param progress Progress value (0.0 to 1.0)
     * @param width Width of progress bar
     * @return String containing ASCII progress bar
     */
    std::string generateProgressBar(double progress, int width);
}