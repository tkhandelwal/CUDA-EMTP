// simulation_results.h
#pragma once

#include <vector>
#include <string>
#include <unordered_map>

/**
 * @brief Structure to store simulation results and performance metrics
 */
struct SimulationResults {
    std::vector<double> timePoints;
    std::unordered_map<std::string, std::vector<double>> nodeVoltages;
    std::unordered_map<std::string, std::vector<double>> branchCurrents;

    // Power electronics results
    std::unordered_map<std::string, std::vector<double>> semiconductorTemperatures;
    std::unordered_map<std::string, std::vector<double>> semiconductorLosses;
    std::unordered_map<std::string, std::vector<double>> converterLosses;
    std::unordered_map<std::string, std::vector<double>> dcVoltages;
    std::unordered_map<std::string, std::vector<double>> dcCurrents;
    std::unordered_map<std::string, std::vector<double>> modulationIndices;

    // Control system results
    std::unordered_map<std::string, std::vector<double>> controllerOutputs;
    std::unordered_map<std::string, std::vector<double>> phaseAngles;
    std::unordered_map<std::string, std::vector<double>> frequencies;
    std::unordered_map<std::string, std::vector<double>> activePowers;
    std::unordered_map<std::string, std::vector<double>> reactivePowers;

    // Harmonic analysis results
    std::unordered_map<std::string, std::vector<double>> harmonicMagnitudes;
    std::unordered_map<std::string, std::vector<double>> harmonicPhases;
    std::unordered_map<std::string, double> thd;

    // Performance metrics
    double totalSimulationTime;
    double matrixBuildTime;
    double solverTime;
    double communicationTime;
    double powerElectronicsTime;
    double controlSystemTime;
    int iterationCount;

    // Default constructor
    SimulationResults() :
        totalSimulationTime(0.0),
        matrixBuildTime(0.0),
        solverTime(0.0),
        communicationTime(0.0),
        powerElectronicsTime(0.0),
        controlSystemTime(0.0),
        iterationCount(0) {
    }

    // Constructor with data
    SimulationResults(
        const std::vector<double>& tp,
        const std::unordered_map<std::string, std::vector<double>>& nv,
        const std::unordered_map<std::string, std::vector<double>>& bc) :
        timePoints(tp),
        nodeVoltages(nv),
        branchCurrents(bc),
        totalSimulationTime(0.0),
        matrixBuildTime(0.0),
        solverTime(0.0),
        communicationTime(0.0),
        powerElectronicsTime(0.0),
        controlSystemTime(0.0),
        iterationCount(0) {
    }

    // Helper to clear all results
    void clear() {
        timePoints.clear();
        nodeVoltages.clear();
        branchCurrents.clear();
        semiconductorTemperatures.clear();
        semiconductorLosses.clear();
        converterLosses.clear();
        dcVoltages.clear();
        dcCurrents.clear();
        modulationIndices.clear();
        controllerOutputs.clear();
        phaseAngles.clear();
        frequencies.clear();
        activePowers.clear();
        reactivePowers.clear();
        harmonicMagnitudes.clear();
        harmonicPhases.clear();
        thd.clear();
        totalSimulationTime = 0.0;
        matrixBuildTime = 0.0;
        solverTime = 0.0;
        communicationTime = 0.0;
        powerElectronicsTime = 0.0;
        controlSystemTime = 0.0;
        iterationCount = 0;
    }
};