#include "boundary_node.h"
#include <algorithm>

BoundaryNode::BoundaryNode(int globalID, const std::string& name, bool trackConvergenceHistory)
    : globalID(globalID), name(name), currentVoltage(0.0), previousVoltage(0.0),
    trackConvergence(trackConvergenceHistory) {
}

// Copy constructor implementation
BoundaryNode::BoundaryNode(const BoundaryNode& other)
    : globalID(other.globalID),
    name(other.name),
    subnetworkIDs(other.subnetworkIDs),
    localIDs(other.localIDs),
    currentVoltage(other.currentVoltage),
    previousVoltage(other.previousVoltage),
    convergenceHistory(other.convergenceHistory),
    trackConvergence(other.trackConvergence) {
}

void BoundaryNode::addSubnetworkMapping(int subnetworkID, int localID) {
    subnetworkIDs.push_back(subnetworkID);
    localIDs.push_back(localID);
}

size_t BoundaryNode::getSubnetworkCount() const {
    return subnetworkIDs.size();
}

int BoundaryNode::getSubnetworkID(size_t index) const {
    if (index < subnetworkIDs.size()) {
        return subnetworkIDs[index];
    }
    return -1;
}

int BoundaryNode::getLocalID(size_t index) const {
    if (index < localIDs.size()) {
        return localIDs[index];
    }
    return -1;
}

void BoundaryNode::updateVoltage(double newVoltage) {
    std::lock_guard<std::mutex> lock(nodeMutex);
    previousVoltage = currentVoltage;
    currentVoltage = newVoltage;

    if (trackConvergence) {
        double difference = std::abs(currentVoltage - previousVoltage);
        convergenceHistory.push_back(difference);
    }
}

double BoundaryNode::getCurrentVoltage() const {
    std::lock_guard<std::mutex> lock(nodeMutex);
    return currentVoltage;
}

double BoundaryNode::getPreviousVoltage() const {
    std::lock_guard<std::mutex> lock(nodeMutex);
    return previousVoltage;
}

double BoundaryNode::getConvergenceRate() const {
    std::lock_guard<std::mutex> lock(nodeMutex);
    if (convergenceHistory.size() < 2) {
        return 0.0;
    }

    // Calculate average convergence rate over the last few iterations
    int numPoints = std::min(5, static_cast<int>(convergenceHistory.size()));
    double sum = 0.0;

    for (int i = convergenceHistory.size() - numPoints; i < convergenceHistory.size() - 1; i++) {
        if (convergenceHistory[i] > 1e-10) {  // Avoid division by zero
            sum += convergenceHistory[i + 1] / convergenceHistory[i];
        }
    }

    return sum / (numPoints - 1);
}

void BoundaryNode::setTrackConvergence(bool track) {
    std::lock_guard<std::mutex> lock(nodeMutex);
    trackConvergence = track;
    if (!trackConvergence) {
        convergenceHistory.clear();
    }
}

const std::vector<double>& BoundaryNode::getConvergenceHistory() const {
    // Note: This doesn't lock because we're returning a reference
    // The caller should ensure thread safety if needed
    return convergenceHistory;
}

void BoundaryNode::clearConvergenceHistory() {
    std::lock_guard<std::mutex> lock(nodeMutex);
    convergenceHistory.clear();
}

bool BoundaryNode::hasConverged(double tolerance) const {
    std::lock_guard<std::mutex> lock(nodeMutex);
    return std::abs(currentVoltage - previousVoltage) < tolerance;
}