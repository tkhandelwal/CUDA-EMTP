#pragma once

#include <string>
#include <vector>
#include <cmath>
#include <mutex>

/**
 * @brief Represents a node at the boundary between subnetworks
 */
class BoundaryNode {
private:
    int globalID;
    std::string name;
    std::vector<int> subnetworkIDs;
    std::vector<int> localIDs;

    // Current and previous voltage values for convergence checking
    double currentVoltage;
    double previousVoltage;


    // Mutex for thread safety when updating values - marked mutable for const methods
    mutable std::mutex nodeMutex;

    // Convergence history for debugging
    std::vector<double> convergenceHistory;
    bool trackConvergence;

public:
    BoundaryNode(int globalID, const std::string& name, bool trackConvergenceHistory = false);

    // Copy constructor
    BoundaryNode(const BoundaryNode& other);

    void addSubnetworkMapping(int subnetworkID, int localID);

    size_t getSubnetworkCount() const;
    int getSubnetworkID(size_t index) const;
    int getLocalID(size_t index) const;

    int getGlobalID() const { return globalID; }
    const std::string& getName() const { return name; }

    void updateVoltage(double newVoltage);
    double getCurrentVoltage() const;
    double getPreviousVoltage() const;

    // Get convergence metrics
    double getConvergenceRate() const;

    // Enable/disable convergence tracking
    void setTrackConvergence(bool track);

    // Get convergence history
    const std::vector<double>& getConvergenceHistory() const;

    // Clear convergence history
    void clearConvergenceHistory();

    // Check if voltages have converged to within tolerance
    bool hasConverged(double tolerance) const;
};