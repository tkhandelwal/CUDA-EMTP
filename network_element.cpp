#include "network_element.h"
#include "cuda_kernels.cuh"
#include "cuda_helpers.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ------------------- TransmissionLine Implementation -------------------

TransmissionLine::TransmissionLine(const std::string& name, int fromNode, int toNode,
    double resistance, double inductance, double capacitance, double length)
    : NetworkElement(name), fromNode(fromNode), toNode(toNode),
    resistance(resistance), inductance(inductance), capacitance(capacitance),
    length(length), historyTermFrom(0.0), historyTermTo(0.0),
    d_parameters(nullptr), d_historyTerms(nullptr) {
}

TransmissionLine::~TransmissionLine() {
    // Free device memory if allocated
    if (d_parameters) {
        cudaFree(d_parameters);
        d_parameters = nullptr;
    }
    if (d_historyTerms) {
        cudaFree(d_historyTerms);
        d_historyTerms = nullptr;
    }
}

void TransmissionLine::updateHistoryTerms(double timeStep, double time) {
    // Compute characteristic impedance and propagation velocity
    double Z0 = sqrt(inductance / capacitance);
    double v = 1.0 / sqrt(inductance * capacitance);

    // Compute travel time
    double travelTime = length / v;

    // In a full implementation, these terms would depend on past values 
    // with delay equal to the travel time
    // This is a simplified implementation
    historyTermFrom = historyTermFrom * exp(-resistance * length / (2.0 * Z0));
    historyTermTo = historyTermTo * exp(-resistance * length / (2.0 * Z0));
}

void TransmissionLine::calculateContributions() {
    // In a full implementation, this would update the admittance matrix
    // and current vector contributions for this transmission line
}

void TransmissionLine::prepareDeviceData() {
    // Allocate device memory for parameters
    if (!d_parameters) {
        cudaMalloc(&d_parameters, 6 * sizeof(double));
        double params[6] = {
            resistance, inductance, capacitance, length,
            static_cast<double>(fromNode), static_cast<double>(toNode)
        };
        cudaMemcpy(d_parameters, params, 6 * sizeof(double), cudaMemcpyHostToDevice);
    }

    // Allocate device memory for history terms
    if (!d_historyTerms) {
        cudaMalloc(&d_historyTerms, 2 * sizeof(double));
    }
}

void TransmissionLine::updateDeviceData() {
    // Copy latest history terms to device
    if (d_historyTerms) {
        double terms[2] = { historyTermFrom, historyTermTo };
        cudaMemcpy(d_historyTerms, terms, 2 * sizeof(double), cudaMemcpyHostToDevice);
    }
}

void TransmissionLine::retrieveDeviceData() {
    // Copy updated history terms from device to host
    if (d_historyTerms) {
        double terms[2];
        cudaMemcpy(terms, d_historyTerms, 2 * sizeof(double), cudaMemcpyDeviceToHost);
        historyTermFrom = terms[0];
        historyTermTo = terms[1];
    }
}

// ------------------- Transformer Implementation -------------------

Transformer::Transformer(const std::string& name, int primaryNode, int secondaryNode,
    double primaryVoltage, double secondaryVoltage, double rating, double leakageReactance)
    : NetworkElement(name), primaryNode(primaryNode), secondaryNode(secondaryNode),
    primaryVoltage(primaryVoltage), secondaryVoltage(secondaryVoltage),
    rating(rating), leakageReactance(leakageReactance),
    historyTerm(0.0), d_parameters(nullptr), d_historyTerms(nullptr) {
}

Transformer::~Transformer() {
    // Free device memory if allocated
    if (d_parameters) {
        cudaFree(d_parameters);
        d_parameters = nullptr;
    }
    if (d_historyTerms) {
        cudaFree(d_historyTerms);
        d_historyTerms = nullptr;
    }
}

void Transformer::updateHistoryTerms(double timeStep, double time) {
    // In a full implementation, this would update the transformer's history terms
    // based on the current state and past values
    // This is a simplified implementation
    double ratio = primaryVoltage / secondaryVoltage;
    double baseZ = (primaryVoltage * primaryVoltage) / (rating * 1e6);
    double leakageZ = leakageReactance * baseZ / 100.0;

    // Apply a simple decay to the history term
    historyTerm = historyTerm * 0.9;
}

void Transformer::calculateContributions() {
    // In a full implementation, this would update the admittance matrix
    // and current vector contributions for this transformer
}

void Transformer::prepareDeviceData() {
    // Allocate device memory for parameters
    if (!d_parameters) {
        cudaMalloc(&d_parameters, 7 * sizeof(double));
        double params[7] = {
            primaryVoltage, secondaryVoltage, rating, leakageReactance,
            static_cast<double>(primaryNode), static_cast<double>(secondaryNode),
            primaryVoltage / secondaryVoltage // Add the turns ratio as well
        };
        cudaMemcpy(d_parameters, params, 7 * sizeof(double), cudaMemcpyHostToDevice);
    }

    // Allocate device memory for history term
    if (!d_historyTerms) {
        cudaMalloc(&d_historyTerms, sizeof(double));
    }
}

void Transformer::updateDeviceData() {
    // Copy latest history term to device
    if (d_historyTerms) {
        cudaMemcpy(d_historyTerms, &historyTerm, sizeof(double), cudaMemcpyHostToDevice);
    }
}

void Transformer::retrieveDeviceData() {
    // Copy updated history term from device to host
    if (d_historyTerms) {
        cudaMemcpy(&historyTerm, d_historyTerms, sizeof(double), cudaMemcpyDeviceToHost);
    }
}

// ------------------- Load Implementation -------------------

Load::Load(const std::string& name, int node, double activePower, double reactivePower, bool isConstantImpedance)
    : NetworkElement(name), node(node), activePower(activePower), reactivePower(reactivePower),
    isConstantImpedance(isConstantImpedance), historyTerm(0.0),
    d_parameters(nullptr), d_historyTerm(nullptr) {
}

Load::~Load() {
    // Free device memory if allocated
    if (d_parameters) {
        cudaFree(d_parameters);
        d_parameters = nullptr;
    }
    if (d_historyTerm) {
        cudaFree(d_historyTerm);
        d_historyTerm = nullptr;
    }
}

void Load::updateHistoryTerms(double timeStep, double time) {
    // In a full implementation, this would update the load's history terms
    // based on the load model (constant impedance, constant power, etc.)

    // For constant impedance model:
    if (isConstantImpedance) {
        // No history terms needed
        historyTerm = 0.0;
    }
    else {
        // For constant power model, history term would depend on voltage
        // This is a simplified implementation
        historyTerm = sqrt(activePower * activePower + reactivePower * reactivePower) / 1e6;
    }
}

void Load::calculateContributions() {
    // In a full implementation, this would update the admittance matrix
    // and current vector contributions for this load
}

void Load::prepareDeviceData() {
    // Allocate device memory for parameters
    if (!d_parameters) {
        cudaMalloc(&d_parameters, 4 * sizeof(double));
        double params[4] = {
            activePower, reactivePower,
            isConstantImpedance ? 1.0 : 0.0,
            static_cast<double>(node)
        };
        cudaMemcpy(d_parameters, params, 4 * sizeof(double), cudaMemcpyHostToDevice);
    }

    // Allocate device memory for history term
    if (!d_historyTerm) {
        cudaMalloc(&d_historyTerm, sizeof(double));
    }
}

void Load::updateDeviceData() {
    // Copy latest history term to device
    if (d_historyTerm) {
        cudaMemcpy(d_historyTerm, &historyTerm, sizeof(double), cudaMemcpyHostToDevice);
    }
}

void Load::retrieveDeviceData() {
    // Copy updated history term from device to host
    if (d_historyTerm) {
        cudaMemcpy(&historyTerm, d_historyTerm, sizeof(double), cudaMemcpyDeviceToHost);
    }
}

// ------------------- VoltageSource Implementation -------------------

VoltageSource::VoltageSource(const std::string& name, int node, double amplitude, double frequency, double phase)
    : NetworkElement(name), node(node), amplitude(amplitude), frequency(frequency), phase(phase), d_parameters(nullptr) {
}

VoltageSource::~VoltageSource() {
    // Free device memory if allocated
    if (d_parameters) {
        cudaFree(d_parameters);
        d_parameters = nullptr;
    }
}

void VoltageSource::updateHistoryTerms(double timeStep, double time) {
    // Voltage sources don't typically require history terms
    // since they impose a voltage rather than requiring integration
}

void VoltageSource::calculateContributions() {
    // In a full implementation, this would update the admittance matrix
    // and current vector contributions for this voltage source
}

double VoltageSource::getValue(double time) const {
    // Calculate the instantaneous voltage value
    return amplitude * sin(2.0 * M_PI * frequency * time + phase * M_PI / 180.0);
}

void VoltageSource::prepareDeviceData() {
    // Allocate device memory for parameters
    if (!d_parameters) {
        cudaMalloc(&d_parameters, 4 * sizeof(double));
        double params[4] = {
            amplitude, frequency, phase,
            static_cast<double>(node)
        };
        cudaMemcpy(d_parameters, params, 4 * sizeof(double), cudaMemcpyHostToDevice);
    }
}

void VoltageSource::updateDeviceData() {
    // No dynamic data to update for a voltage source
}

void VoltageSource::retrieveDeviceData() {
    // No data to retrieve for a voltage source
}

// ------------------- Fault Implementation -------------------

Fault::Fault(const std::string& name, int node, double startTime, double duration, double resistance)
    : NetworkElement(name), node(node), startTime(startTime), duration(duration), resistance(resistance),
    active(false), d_parameters(nullptr), d_active(nullptr) {
}

Fault::~Fault() {
    // Free device memory if allocated
    if (d_parameters) {
        cudaFree(d_parameters);
        d_parameters = nullptr;
    }
    if (d_active) {
        cudaFree(d_active);
        d_active = nullptr;
    }
}

void Fault::updateHistoryTerms(double timeStep, double time) {
    // Update fault status based on time
    active = isActive(time);
}

void Fault::calculateContributions() {
    // In a full implementation, this would update the admittance matrix
    // and current vector contributions for this fault if active
}

bool Fault::isActive(double time) const {
    return (time >= startTime) && (time <= startTime + duration);
}

void Fault::prepareDeviceData() {
    // Allocate device memory for parameters
    if (!d_parameters) {
        cudaMalloc(&d_parameters, 4 * sizeof(double));
        double params[4] = {
            startTime, duration, resistance,
            static_cast<double>(node)
        };
        cudaMemcpy(d_parameters, params, 4 * sizeof(double), cudaMemcpyHostToDevice);
    }

    // Allocate device memory for active state
    if (!d_active) {
        cudaMalloc(&d_active, sizeof(bool));
    }
}

void Fault::updateDeviceData() {
    // Copy latest active state to device
    if (d_active) {
        cudaMemcpy(d_active, &active, sizeof(bool), cudaMemcpyHostToDevice);
    }
}

void Fault::retrieveDeviceData() {
    // No data to retrieve for a fault
}

// ------------------- Factory Function Implementation -------------------

std::unique_ptr<NetworkElement> createNetworkElement(const std::string& elementSpec) {
    std::istringstream iss(elementSpec);
    std::string elementType;
    iss >> elementType;

    if (elementType == "LINE") {
        std::string name, fromNode, toNode;
        double r, l, c, length;

        iss >> fromNode >> toNode >> name >> r >> l >> c >> length;

        // In a full implementation, node indices would be looked up by name
        // For simplicity, we'll use a placeholder value here
        int fromNodeIdx = 0;
        int toNodeIdx = 1;

        return std::make_unique<TransmissionLine>(name, fromNodeIdx, toNodeIdx, r, l, c, length);
    }
    else if (elementType == "TRANSFORMER") {
        std::string name, primaryNode, secondaryNode;
        double primaryV, secondaryV, rating, leakageX;

        iss >> primaryNode >> secondaryNode >> name >> primaryV >> secondaryV >> rating >> leakageX;

        // Placeholder node indices
        int primaryNodeIdx = 0;
        int secondaryNodeIdx = 1;

        return std::make_unique<Transformer>(name, primaryNodeIdx, secondaryNodeIdx,
            primaryV, secondaryV, rating, leakageX);
    }
    else if (elementType == "LOAD") {
        std::string name, nodeName;
        double activePower, reactivePower;

        iss >> nodeName >> name >> activePower >> reactivePower;

        // Placeholder node index
        int nodeIdx = 0;

        return std::make_unique<Load>(name, nodeIdx, activePower, reactivePower);
    }
    else if (elementType == "VOLTAGE") {
        std::string name, nodeName;
        double amplitude, frequency, phase;

        iss >> nodeName >> name >> amplitude >> frequency >> phase;

        // Placeholder node index
        int nodeIdx = 0;

        return std::make_unique<VoltageSource>(name, nodeIdx, amplitude, frequency, phase);
    }
    else if (elementType == "FAULT") {
        std::string name, nodeName;
        double startTime, duration, resistance;

        iss >> nodeName >> name >> startTime >> duration >> resistance;

        // Placeholder node index
        int nodeIdx = 0;

        return std::make_unique<Fault>(name, nodeIdx, startTime, duration, resistance);
    }

    // Unknown element type
    std::cerr << "Unknown network element type: " << elementType << std::endl;
    return nullptr;
}