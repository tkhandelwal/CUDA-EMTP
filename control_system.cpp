#include "control_system.h"
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// PLL implementation
PLL::PLL(const std::string& name, double bandwidth, double dampingRatio,
    double kp, double ki, double settlingTime, double overshoot, int type)
    : ControlSystem(name, -1), bandwidth(bandwidth), dampingRatio(dampingRatio),
    kp(kp), ki(ki), settlingTime(settlingTime), overshoot(overshoot), type(type),
    theta(0.0), omega(2.0 * M_PI * 60.0), integralTerm(0.0), vq(0.0),
    d_stateVars(nullptr) {
}

PLL::~PLL() {
    // Free device memory if allocated
    if (d_stateVars) {
        cudaFree(d_stateVars);
        d_stateVars = nullptr;
    }
}

void PLL::initialize() {
    // Reset state variables
    theta = 0.0;
    omega = 2.0 * M_PI * 60.0; // Start at nominal frequency (60 Hz)
    integralTerm = 0.0;
    vq = 0.0;
}

void PLL::update(double time, double timeStep) {
    // For simplicity, we'll use a dummy input voltage here
    // In a real application, this would come from measurements
    std::vector<double> dummyVoltages = {
        sin(2.0 * M_PI * 60.0 * time),
        sin(2.0 * M_PI * 60.0 * time - 2.0 * M_PI / 3.0),
        sin(2.0 * M_PI * 60.0 * time - 4.0 * M_PI / 3.0)
    };

    processVoltages(dummyVoltages);

    // PI controller to drive vq to zero
    double errorSignal = -vq; // Want vq to be zero

    // PI controller
    double proportionalTerm = kp * errorSignal;
    integralTerm += ki * errorSignal * timeStep;

    // Anti-windup for integrator
    double maxIntegralTerm = 2.0 * M_PI * 5.0; // Limit to +/- 5 Hz deviation
    integralTerm = std::max(-maxIntegralTerm, std::min(maxIntegralTerm, integralTerm));

    // Update frequency
    omega = 2.0 * M_PI * 60.0 + proportionalTerm + integralTerm;

    // Update phase angle
    theta += omega * timeStep;

    // Normalize theta to [0, 2*pi)
    theta = fmod(theta, 2.0 * M_PI);
    if (theta < 0) {
        theta += 2.0 * M_PI;
    }
}

void PLL::processVoltages(const std::vector<double>& voltages) {
    // Clarke transform (abc to alpha-beta)
    double valpha = (2.0 / 3.0) * (voltages[0] - 0.5 * voltages[1] - 0.5 * voltages[2]);
    double vbeta = (2.0 / 3.0) * (0.866 * voltages[1] - 0.866 * voltages[2]);

    // Park transform (alpha-beta to dq)
    double vd = valpha * cos(theta) + vbeta * sin(theta);
    vq = -valpha * sin(theta) + vbeta * cos(theta);

    // Specialized processing based on PLL type
    if (type == 2) { // DDSRF (Decoupled Double Synchronous Reference Frame)
        // In a real implementation, this would handle unbalanced grid conditions
        // by separating positive and negative sequence components

        // For this example, just using the basic SRF approach
    }
    else if (type == 3) { // DSOGI (Dual Second Order Generalized Integrator)
        // In a real implementation, this would use frequency-adaptive filters
        // to extract positive and negative sequence components

        // For this example, just using the basic SRF approach
    }
}

void PLL::prepareDeviceData() {
    // Allocate and initialize device memory for CUDA
    if (!d_stateVars) {
        cudaMalloc(&d_stateVars, 4 * sizeof(double));
    }

    // Copy initial data to device
    updateDeviceData();
}

void PLL::updateDeviceData() {
    if (d_stateVars) {
        double stateVars[4] = {
            theta,
            omega,
            integralTerm,
            vq
        };

        cudaMemcpy(d_stateVars, stateVars, 4 * sizeof(double),
            cudaMemcpyHostToDevice);
    }
}

void PLL::retrieveDeviceData() {
    if (d_stateVars) {
        double stateVars[4];

        cudaMemcpy(stateVars, d_stateVars, 4 * sizeof(double),
            cudaMemcpyDeviceToHost);

        theta = stateVars[0];
        omega = stateVars[1];
        integralTerm = stateVars[2];
        vq = stateVars[3];
    }
}

// CurrentController implementation
CurrentController::CurrentController(const std::string& name, int type, double bandwidth,
    double dampingRatio, double kp, double ki,
    double samplingFrequency, int predictionHorizon)
    : ControlSystem(name, -1), type(type), bandwidth(bandwidth), dampingRatio(dampingRatio),
    kp(kp), ki(ki), samplingFrequency(samplingFrequency), predictionHorizon(predictionHorizon),
    d_stateVars(nullptr) {

    // Initialize vectors
    references.resize(2, 0.0);    // d and q references
    feedbacks.resize(2, 0.0);     // d and q feedbacks
    outputs.resize(2, 0.0);       // d and q outputs
    integralTerms.resize(2, 0.0); // d and q integral terms
}

CurrentController::~CurrentController() {
    // Free device memory if allocated
    if (d_stateVars) {
        cudaFree(d_stateVars);
        d_stateVars = nullptr;
    }
}

void CurrentController::initialize() {
    // Reset all state variables
    references[0] = 0.0;
    references[1] = 0.0;
    feedbacks[0] = 0.0;
    feedbacks[1] = 0.0;
    outputs[0] = 0.0;
    outputs[1] = 0.0;
    integralTerms[0] = 0.0;
    integralTerms[1] = 0.0;
}

void CurrentController::update(double time, double timeStep) {
    // Calculate error for both d and q axes
    double errorD = references[0] - feedbacks[0];
    double errorQ = references[1] - feedbacks[1];

    // Handle different controller types
    if (type == 1) { // PI controller
        // PI control for d-axis
        double proportionalTermD = kp * errorD;
        integralTerms[0] += ki * errorD * timeStep;

        // PI control for q-axis
        double proportionalTermQ = kp * errorQ;
        integralTerms[1] += ki * errorQ * timeStep;

        // Anti-windup for integrators
        double maxIntegralTerm = 500.0; // Example limit
        integralTerms[0] = std::max(-maxIntegralTerm, std::min(maxIntegralTerm, integralTerms[0]));
        integralTerms[1] = std::max(-maxIntegralTerm, std::min(maxIntegralTerm, integralTerms[1]));

        // Compute controller outputs
        outputs[0] = proportionalTermD + integralTerms[0];
        outputs[1] = proportionalTermQ + integralTerms[1];
    }
    else if (type == 2) { // PR (Proportional Resonant) controller
        // Proportional Resonant controllers are typically used in the stationary frame
        // but here we're assuming dq frame and using a simplified implementation

        double omega0 = 2.0 * M_PI * 60.0; // Fundamental frequency

        // For demonstration - this is not a true PR controller implementation
        // A real PR controller would involve more complex dynamics
        outputs[0] = kp * errorD + integralTerms[0];
        outputs[1] = kp * errorQ + integralTerms[1];

        // Update integral terms with resonant component
        integralTerms[0] += ki * errorD * timeStep * cos(omega0 * time);
        integralTerms[1] += ki * errorQ * timeStep * sin(omega0 * time);
    }
    else if (type == 3) { // Deadbeat controller
        // Deadbeat controllers aim to reach the reference in minimal time
        // For a simplified model, we'll assume deadbeat response in one sample period

        // For first-order system, deadbeat gain is 1/(plant gain)
        // Assuming plant gain of 1 for simplicity
        outputs[0] = references[0];
        outputs[1] = references[1];

        // In a real implementation, this would account for plant dynamics
    }
    else if (type == 4) { // MPC (Model Predictive Control)
        // MPC uses a model to predict future behavior and optimize control
        // This is a highly simplified placeholder

        // For demonstration purposes only - not a real MPC implementation
        // A real MPC would solve an optimization problem

        // Simplistic approach - assume linear model and compute inputs
        // that would reach reference over prediction horizon
        double predictedStepsToReach = predictionHorizon;
        outputs[0] = feedbacks[0] + (references[0] - feedbacks[0]) / predictedStepsToReach;
        outputs[1] = feedbacks[1] + (references[1] - feedbacks[1]) / predictedStepsToReach;
    }

    // Apply output limits (saturation)
    double maxOutput = 1000.0; // Example limit
    outputs[0] = std::max(-maxOutput, std::min(maxOutput, outputs[0]));
    outputs[1] = std::max(-maxOutput, std::min(maxOutput, outputs[1]));
}

void CurrentController::setReferences(const std::vector<double>& refs) {
    if (refs.size() >= 2) {
        references[0] = refs[0];
        references[1] = refs[1];
    }
}

void CurrentController::setFeedbacks(const std::vector<double>& fbs) {
    if (fbs.size() >= 2) {
        feedbacks[0] = fbs[0];
        feedbacks[1] = fbs[1];
    }
}

void CurrentController::prepareDeviceData() {
    // Allocate and initialize device memory for CUDA
    if (!d_stateVars) {
        // 8 values: 2 references, 2 feedbacks, 2 outputs, 2 integral terms
        cudaMalloc(&d_stateVars, 8 * sizeof(double));
    }

    // Copy initial data to device
    updateDeviceData();
}

void CurrentController::updateDeviceData() {
    if (d_stateVars) {
        double stateVars[8] = {
            references[0], references[1],
            feedbacks[0], feedbacks[1],
            outputs[0], outputs[1],
            integralTerms[0], integralTerms[1]
        };

        cudaMemcpy(d_stateVars, stateVars, 8 * sizeof(double),
            cudaMemcpyHostToDevice);
    }
}

void CurrentController::retrieveDeviceData() {
    if (d_stateVars) {
        double stateVars[8];

        cudaMemcpy(stateVars, d_stateVars, 8 * sizeof(double),
            cudaMemcpyDeviceToHost);

        references[0] = stateVars[0];
        references[1] = stateVars[1];
        feedbacks[0] = stateVars[2];
        feedbacks[1] = stateVars[3];
        outputs[0] = stateVars[4];
        outputs[1] = stateVars[5];
        integralTerms[0] = stateVars[6];
        integralTerms[1] = stateVars[7];
    }
}

// VoltageController implementation
VoltageController::VoltageController(const std::string& name, int type, double bandwidth,
    double dampingRatio, double kp, double ki,
    double droopCoefficient, bool antiWindup)
    : ControlSystem(name, -1), type(type), bandwidth(bandwidth), dampingRatio(dampingRatio),
    kp(kp), ki(ki), droopCoefficient(droopCoefficient), antiWindup(antiWindup),
    d_stateVars(nullptr) {

    // Initialize vectors
    references.resize(2, 0.0);    // d and q references
    feedbacks.resize(2, 0.0);     // d and q feedbacks
    outputs.resize(2, 0.0);       // d and q outputs
    integralTerms.resize(2, 0.0); // d and q integral terms
}

VoltageController::~VoltageController() {
    // Free device memory if allocated
    if (d_stateVars) {
        cudaFree(d_stateVars);
        d_stateVars = nullptr;
    }
}

void VoltageController::initialize() {
    // Reset all state variables
    references[0] = 0.0;
    references[1] = 0.0;
    feedbacks[0] = 0.0;
    feedbacks[1] = 0.0;
    outputs[0] = 0.0;
    outputs[1] = 0.0;
    integralTerms[0] = 0.0;
    integralTerms[1] = 0.0;
}

void VoltageController::update(double time, double timeStep) {
    // Calculate error for both d and q axes
    double errorD = references[0] - feedbacks[0];
    double errorQ = references[1] - feedbacks[1];

    // Handle different controller types
    if (type == 1) { // PI controller
        // PI control for d-axis
        double proportionalTermD = kp * errorD;
        integralTerms[0] += ki * errorD * timeStep;

        // PI control for q-axis
        double proportionalTermQ = kp * errorQ;
        integralTerms[1] += ki * errorQ * timeStep;

        // Apply anti-windup if enabled
        if (antiWindup) {
            double maxIntegralTerm = 500.0; // Example limit
            integralTerms[0] = std::max(-maxIntegralTerm, std::min(maxIntegralTerm, integralTerms[0]));
            integralTerms[1] = std::max(-maxIntegralTerm, std::min(maxIntegralTerm, integralTerms[1]));
        }

        // Compute controller outputs
        outputs[0] = proportionalTermD + integralTerms[0];
        outputs[1] = proportionalTermQ + integralTerms[1];
    }
    else if (type == 2) { // Droop controller
        // Droop control for d-axis (voltage magnitude droop)
        // In a droop controller, output power affects voltage reference

        // For demonstration - this is a simplified implementation
        // Assuming outputs[0] is active power, outputs[1] is reactive power
        double nominalVoltage = 1.0; // per unit

        // Apply droop: V = V0 - k * Q
        double droopVoltage = nominalVoltage - (droopCoefficient / 100.0) * outputs[1];

        // PI control for voltage magnitude
        double vRef = droopVoltage;
        double vFb = sqrt(feedbacks[0] * feedbacks[0] + feedbacks[1] * feedbacks[1]);
        double vError = vRef - vFb;

        // Use PI controller to track voltage magnitude
        double proportionalTermV = kp * vError;
        integralTerms[0] += ki * vError * timeStep;

        // Anti-windup if enabled
        if (antiWindup) {
            double maxIntegralTerm = 500.0; // Example limit
            integralTerms[0] = std::max(-maxIntegralTerm, std::min(maxIntegralTerm, integralTerms[0]));
        }

        // Compute voltage controller output
        double vControlOutput = proportionalTermV + integralTerms[0];

        // Update d and q components of output
        // For simplicity, keeping voltage in d-axis only
        outputs[0] = vControlOutput;
        outputs[1] = 0.0;
    }
    else if (type == 3) { // Resonant controller
        double omega0 = 2.0 * M_PI * 60.0; // Fundamental frequency

        // Resonant controller - effective at tracking sinusoidal references
        // For demonstration purposes - simplified version

        // Update integral terms with resonant component
        integralTerms[0] += 2.0 * ki * errorD * timeStep;
        integralTerms[1] += 2.0 * ki * errorQ * timeStep;

        // Damping term to avoid instability
        integralTerms[0] -= omega0 * omega0 * timeStep * integralTerms[0];
        integralTerms[1] -= omega0 * omega0 * timeStep * integralTerms[1];

        // Compute controller outputs
        outputs[0] = kp * errorD + integralTerms[0];
        outputs[1] = kp * errorQ + integralTerms[1];
    }

    // Apply output limits (saturation)
    double maxOutput = 1000.0; // Example limit
    outputs[0] = std::max(-maxOutput, std::min(maxOutput, outputs[0]));
    outputs[1] = std::max(-maxOutput, std::min(maxOutput, outputs[1]));
}

void VoltageController::setReferences(const std::vector<double>& refs) {
    if (refs.size() >= 2) {
        references[0] = refs[0];
        references[1] = refs[1];
    }
}

void VoltageController::setFeedbacks(const std::vector<double>& fbs) {
    if (fbs.size() >= 2) {
        feedbacks[0] = fbs[0];
        feedbacks[1] = fbs[1];
    }
}

void VoltageController::prepareDeviceData() {
    // Allocate and initialize device memory for CUDA
    if (!d_stateVars) {
        // 8 values: 2 references, 2 feedbacks, 2 outputs, 2 integral terms
        cudaMalloc(&d_stateVars, 8 * sizeof(double));
    }

    // Copy initial data to device
    updateDeviceData();
}

void VoltageController::updateDeviceData() {
    if (d_stateVars) {
        double stateVars[8] = {
            references[0], references[1],
            feedbacks[0], feedbacks[1],
            outputs[0], outputs[1],
            integralTerms[0], integralTerms[1]
        };

        cudaMemcpy(d_stateVars, stateVars, 8 * sizeof(double),
            cudaMemcpyHostToDevice);
    }
}

void VoltageController::retrieveDeviceData() {
    if (d_stateVars) {
        double stateVars[8];

        cudaMemcpy(stateVars, d_stateVars, 8 * sizeof(double),
            cudaMemcpyDeviceToHost);

        references[0] = stateVars[0];
        references[1] = stateVars[1];
        feedbacks[0] = stateVars[2];
        feedbacks[1] = stateVars[3];
        outputs[0] = stateVars[4];
        outputs[1] = stateVars[5];
        integralTerms[0] = stateVars[6];
        integralTerms[1] = stateVars[7];
    }
}

// PowerController implementation
PowerController::PowerController(const std::string& name, int type, double bandwidth,
    double dampingRatio, double kp, double ki,
    double pDroop, double qDroop)
    : ControlSystem(name, -1), type(type), bandwidth(bandwidth), dampingRatio(dampingRatio),
    kp(kp), ki(ki), pDroop(pDroop), qDroop(qDroop),
    pRef(0.0), qRef(0.0), pFeedback(0.0), qFeedback(0.0),
    d_stateVars(nullptr) {

    // Initialize vectors
    outputs.resize(2, 0.0);       // d and q outputs
    integralTerms.resize(2, 0.0); // d and q integral terms
}

PowerController::~PowerController() {
    // Free device memory if allocated
    if (d_stateVars) {
        cudaFree(d_stateVars);
        d_stateVars = nullptr;
    }
}

void PowerController::initialize() {
    // Reset all state variables
    pRef = 0.0;
    qRef = 0.0;
    pFeedback = 0.0;
    qFeedback = 0.0;
    outputs[0] = 0.0;
    outputs[1] = 0.0;
    integralTerms[0] = 0.0;
    integralTerms[1] = 0.0;
}

void PowerController::update(double time, double timeStep) {
    // Calculate errors for active and reactive power
    double pError = pRef - pFeedback;
    double qError = qRef - qFeedback;

    // Handle different controller types
    if (type == 1) { // PI controller
        // PI control for active power
        double proportionalTermP = kp * pError;
        integralTerms[0] += ki * pError * timeStep;

        // PI control for reactive power
        double proportionalTermQ = kp * qError;
        integralTerms[1] += ki * qError * timeStep;

        // Anti-windup for integrators
        double maxIntegralTerm = 100.0; // Example limit
        integralTerms[0] = std::max(-maxIntegralTerm, std::min(maxIntegralTerm, integralTerms[0]));
        integralTerms[1] = std::max(-maxIntegralTerm, std::min(maxIntegralTerm, integralTerms[1]));

        // Compute controller outputs
        // For a grid-connected inverter, these would typically be current references
        outputs[0] = proportionalTermP + integralTerms[0]; // id reference
        outputs[1] = proportionalTermQ + integralTerms[1]; // iq reference
    }
    else if (type == 2) { // Droop controller
        // Droop control: frequency and voltage droop with active and reactive power
        double nominalFrequency = 60.0; // Hz
        double nominalVoltage = 1.0;    // per unit

        // Calculate droop response
        // In droop control: f = f0 - kp * P, V = V0 - kq * Q
        double frequency = nominalFrequency - (pDroop / 100.0) * pFeedback;
        double voltage = nominalVoltage - (qDroop / 100.0) * qFeedback;

        // Convert to controller outputs
        // In real implementation, these would feed into other controllers
        // For demonstration purposes, mapping directly to d-q outputs
        outputs[0] = voltage;             // Vd reference (voltage magnitude)
        outputs[1] = 2.0 * M_PI * frequency; // omega reference (frequency in rad/s)
    }

    // Apply output limits (saturation)
    double maxOutput = 1000.0; // Example limit
    outputs[0] = std::max(-maxOutput, std::min(maxOutput, outputs[0]));
    outputs[1] = std::max(-maxOutput, std::min(maxOutput, outputs[1]));
}

void PowerController::setReferences(double pRef, double qRef) {
    this->pRef = pRef;
    this->qRef = qRef;
}

void PowerController::setFeedbacks(double pFeedback, double qFeedback) {
    this->pFeedback = pFeedback;
    this->qFeedback = qFeedback;
}

void PowerController::prepareDeviceData() {
    // Allocate and initialize device memory for CUDA
    if (!d_stateVars) {
        // 8 values: pRef, qRef, pFeedback, qFeedback, 2 outputs, 2 integral terms
        cudaMalloc(&d_stateVars, 8 * sizeof(double));
    }

    // Copy initial data to device
    updateDeviceData();
}

void PowerController::updateDeviceData() {
    if (d_stateVars) {
        double stateVars[8] = {
            pRef, qRef,
            pFeedback, qFeedback,
            outputs[0], outputs[1],
            integralTerms[0], integralTerms[1]
        };

        cudaMemcpy(d_stateVars, stateVars, 8 * sizeof(double),
            cudaMemcpyHostToDevice);
    }
}

void PowerController::retrieveDeviceData() {
    if (d_stateVars) {
        double stateVars[8];

        cudaMemcpy(stateVars, d_stateVars, 8 * sizeof(double),
            cudaMemcpyDeviceToHost);

        pRef = stateVars[0];
        qRef = stateVars[1];
        pFeedback = stateVars[2];
        qFeedback = stateVars[3];
        outputs[0] = stateVars[4];
        outputs[1] = stateVars[5];
        integralTerms[0] = stateVars[6];
        integralTerms[1] = stateVars[7];
    }
}

// DCVoltageController implementation
DCVoltageController::DCVoltageController(const std::string& name, int type, double bandwidth,
    double dampingRatio, double kp, double ki, double droop)
    : ControlSystem(name, -1), type(type), bandwidth(bandwidth), dampingRatio(dampingRatio),
    kp(kp), ki(ki), droop(droop),
    reference(0.0), feedback(0.0), output(0.0), integralTerm(0.0),
    d_stateVars(nullptr) {
}

DCVoltageController::~DCVoltageController() {
    // Free device memory if allocated
    if (d_stateVars) {
        cudaFree(d_stateVars);
        d_stateVars = nullptr;
    }
}

void DCVoltageController::initialize() {
    // Reset all state variables
    reference = 0.0;
    feedback = 0.0;
    output = 0.0;
    integralTerm = 0.0;
}

void DCVoltageController::update(double time, double timeStep) {
    // Calculate DC voltage error
    double error = reference - feedback;

    // Handle different controller types
    if (type == 1) { // PI controller
        // PI control for DC voltage
        double proportionalTerm = kp * error;
        integralTerm += ki * error * timeStep;

        // Anti-windup for integrator
        double maxIntegralTerm = 100.0; // Example limit
        integralTerm = std::max(-maxIntegralTerm, std::min(maxIntegralTerm, integralTerm));

        // Compute controller output
        // For DC voltage control, this is typically a current or power reference
        output = proportionalTerm + integralTerm;
    }
    else if (type == 2) { // Droop controller
        // DC voltage droop control: Vdc = Vdc0 - k * P
        double nominalDcVoltage = reference;

        // In a real implementation, would measure power flow and apply droop
        // For demonstration, using feedback as power measurement
        double powerMeasurement = output; // Using previous output as a proxy

        // Apply droop
        double droopVoltage = nominalDcVoltage - (droop / 100.0) * powerMeasurement;

        // PI control for DC voltage with droop reference
        double vdcError = droopVoltage - feedback;
        double proportionalTerm = kp * vdcError;
        integralTerm += ki * vdcError * timeStep;

        // Anti-windup for integrator
        double maxIntegralTerm = 100.0; // Example limit
        integralTerm = std::max(-maxIntegralTerm, std::min(maxIntegralTerm, integralTerm));

        // Compute controller output
        output = proportionalTerm + integralTerm;
    }

    // Apply output limits (saturation)
    double maxOutput = 1000.0; // Example limit
    output = std::max(-maxOutput, std::min(maxOutput, output));
}

void DCVoltageController::setReference(double ref) {
    this->reference = ref;
}

void DCVoltageController::setFeedback(double fb) {
    this->feedback = fb;
}

void DCVoltageController::prepareDeviceData() {
    // Allocate and initialize device memory for CUDA
    if (!d_stateVars) {
        // 4 values: reference, feedback, output, integralTerm
        cudaMalloc(&d_stateVars, 4 * sizeof(double));
    }

    // Copy initial data to device
    updateDeviceData();
}

void DCVoltageController::updateDeviceData() {
    if (d_stateVars) {
        double stateVars[4] = {
            reference,
            feedback,
            output,
            integralTerm
        };

        cudaMemcpy(d_stateVars, stateVars, 4 * sizeof(double),
            cudaMemcpyHostToDevice);
    }
}

void DCVoltageController::retrieveDeviceData() {
    if (d_stateVars) {
        double stateVars[4];

        cudaMemcpy(stateVars, d_stateVars, 4 * sizeof(double),
            cudaMemcpyDeviceToHost);

        reference = stateVars[0];
        feedback = stateVars[1];
        output = stateVars[2];
        integralTerm = stateVars[3];
    }
}

// GridFormingController implementation
GridFormingController::GridFormingController(const std::string& name, int type, double virtualInertia,
    double droop, double damping, double synchronizingCoeff,
    double lowPassFilterFreq)
    : ControlSystem(name, -1), type(type), virtualInertia(virtualInertia),
    droop(droop), damping(damping), synchronizingCoeff(synchronizingCoeff),
    lowPassFilterFreq(lowPassFilterFreq),
    theta(0.0), omega(2.0 * M_PI * 60.0), p(0.0), q(0.0),
    d_stateVars(nullptr) {

    // Initialize vectors
    voltageRefs.resize(2, 0.0); // d and q voltage references
}

GridFormingController::~GridFormingController() {
    // Free device memory if allocated
    if (d_stateVars) {
        cudaFree(d_stateVars);
        d_stateVars = nullptr;
    }
}

void GridFormingController::initialize() {
    // Reset all state variables
    theta = 0.0;
    omega = 2.0 * M_PI * 60.0; // Start at nominal frequency (60 Hz)
    p = 0.0;
    q = 0.0;
    voltageRefs[0] = 1.0; // Default to nominal voltage in d-axis
    voltageRefs[1] = 0.0; // No voltage in q-axis (aligned with d-axis)
}

void GridFormingController::update(double time, double timeStep) {
    // Low-pass filter for power measurements
    // In real implementation, p and q would come from measurements
    double pFiltered = p;
    double qFiltered = q;

    // Handle different controller types
    if (type == 1) { // VSM (Virtual Synchronous Machine)
        // VSM emulates the behavior of a synchronous machine

        // Frequency droop with inertia
        double nominalFrequency = 60.0; // Hz
        double omegaNominal = 2.0 * M_PI * nominalFrequency; // rad/s

        // Calculate frequency deviation based on droop and inertia
        double omegaRef = omegaNominal - (droop / 100.0) * pFiltered; // Droop response

        // Apply virtual inertia dynamics (swing equation)
        // dω/dt = (Tm - Te) / (2H)
        // where Tm = mechanical torque, Te = electrical torque, H = inertia constant

        // In a simplified form:
        // dω/dt = (ωref - ω) / (2H) - D * (ω - ω0)
        // where D is damping constant

        // Numerical integration (Euler method)
        double deltaOmega = (omegaRef - omega) / (2.0 * virtualInertia) - damping * (omega - omegaNominal);
        omega += deltaOmega * timeStep;

        // Update phase angle
        theta += omega * timeStep;

        // Normalize theta to [0, 2*pi)
        theta = fmod(theta, 2.0 * M_PI);
        if (theta < 0) {
            theta += 2.0 * M_PI;
        }

        // Voltage magnitude control with reactive power droop
        double nominalVoltage = 1.0; // per unit
        double voltage = nominalVoltage - (droop / 100.0) * qFiltered;

        // Set voltage references in dq frame
        voltageRefs[0] = voltage; // d-axis aligned with converter reference frame
        voltageRefs[1] = 0.0;     // q-axis component is zero
    }
    else if (type == 2) { // Droop control (simplified version)
        // Droop control is a simpler version that just applies droop response
        // without explicit inertia emulation

        double nominalFrequency = 60.0; // Hz
        double omegaNominal = 2.0 * M_PI * nominalFrequency; // rad/s

        // Apply P-f droop
        omega = omegaNominal - (droop / 100.0) * pFiltered;

        // Update phase angle
        theta += omega * timeStep;

        // Normalize theta to [0, 2*pi)
        theta = fmod(theta, 2.0 * M_PI);
        if (theta < 0) {
            theta += 2.0 * M_PI;
        }

        // Apply Q-V droop
        double nominalVoltage = 1.0; // per unit
        double voltage = nominalVoltage - (droop / 100.0) * qFiltered;

        // Set voltage references in dq frame
        voltageRefs[0] = voltage;
        voltageRefs[1] = 0.0;
    }
    else if (type == 3) { // dVOC (dispatchable Virtual Oscillator Control)
        // dVOC is a more advanced control approach that enables
        // synchronization without PLL

        // Simplified implementation for demonstration
        double nominalFrequency = 60.0; // Hz
        double omegaNominal = 2.0 * M_PI * nominalFrequency; // rad/s

        // Apply dVOC dynamics
        // In real implementation, this would involve more complex calculations
        // involving synchronizing power and voltage dynamics

        // Update frequency based on power imbalance
        omega = omegaNominal - (droop / 100.0) * pFiltered;

        // Apply synchronizing dynamics
        // The synchronizing coefficient affects how strongly the converter
        // synchronizes with the grid
        omega += synchronizingCoeff * sin(theta) * timeStep;

        // Update phase angle
        theta += omega * timeStep;

        // Normalize theta to [0, 2*pi)
        theta = fmod(theta, 2.0 * M_PI);
        if (theta < 0) {
            theta += 2.0 * M_PI;
        }

        // Update voltage references
        double nominalVoltage = 1.0; // per unit
        double voltage = nominalVoltage - (droop / 100.0) * qFiltered;

        voltageRefs[0] = voltage * cos(0.1 * sin(theta)); // Example of modulation
        voltageRefs[1] = voltage * sin(0.1 * sin(theta));
    }
}

void GridFormingController::processPowerInputs(double pInput, double qInput) {
    // Apply low-pass filtering to power measurements
    double tau = 1.0 / (2.0 * M_PI * lowPassFilterFreq);
    double timeStep = 1.0 / (20.0 * lowPassFilterFreq); // Assuming 20 points per cycle

    // First-order low-pass filter: y(n) = a*y(n-1) + (1-a)*x(n)
    double a = exp(-timeStep / tau);

    p = a * p + (1.0 - a) * pInput;
    q = a * q + (1.0 - a) * qInput;
}

void GridFormingController::prepareDeviceData() {
    // Allocate and initialize device memory for CUDA
    if (!d_stateVars) {
        // 6 values: theta, omega, p, q, vd, vq
        cudaMalloc(&d_stateVars, 6 * sizeof(double));
    }

    // Copy initial data to device
    updateDeviceData();
}

void GridFormingController::updateDeviceData() {
    if (d_stateVars) {
        double stateVars[6] = {
            theta,
            omega,
            p,
            q,
            voltageRefs[0],
            voltageRefs[1]
        };

        cudaMemcpy(d_stateVars, stateVars, 6 * sizeof(double),
            cudaMemcpyHostToDevice);
    }
}

void GridFormingController::retrieveDeviceData() {
    if (d_stateVars) {
        double stateVars[6];

        cudaMemcpy(stateVars, d_stateVars, 6 * sizeof(double),
            cudaMemcpyDeviceToHost);

        theta = stateVars[0];
        omega = stateVars[1];
        p = stateVars[2];
        q = stateVars[3];
        voltageRefs[0] = stateVars[4];
        voltageRefs[1] = stateVars[5];
    }
}

// CapacitorBalancing implementation
CapacitorBalancing::CapacitorBalancing(const std::string& name, int type, double toleranceBand,
    double switchingFreqLimit, double updateRate)
    : ControlSystem(name, -1), type(type), toleranceBand(toleranceBand),
    switchingFreqLimit(switchingFreqLimit), updateRate(updateRate),
    d_capVoltages(nullptr), d_switchStates(nullptr), d_switchTimes(nullptr) {
}

CapacitorBalancing::~CapacitorBalancing() {
    // Free device memory if allocated
    if (d_capVoltages) {
        cudaFree(d_capVoltages);
        d_capVoltages = nullptr;
    }

    if (d_switchStates) {
        cudaFree(d_switchStates);
        d_switchStates = nullptr;
    }

    if (d_switchTimes) {
        cudaFree(d_switchTimes);
        d_switchTimes = nullptr;
    }
}

void CapacitorBalancing::initialize() {
    // Reset state variables
    capacitorVoltages.clear();
    switchingStates.clear();
    switchingTimes.clear();
}

void CapacitorBalancing::update(double time, double timeStep) {
    // Make sure we have capacitor voltages to work with
    if (capacitorVoltages.empty()) {
        return;
    }

    // Initialize switching states and times if they don't match capacitor voltages
    if (switchingStates.size() != capacitorVoltages.size()) {
        switchingStates.resize(capacitorVoltages.size(), 0);
        switchingTimes.resize(capacitorVoltages.size(), 0.0);
    }

    // Handle different balancing algorithm types
    if (type == 1) { // Sorting-based balancing
        sortingBalancing(time);
    }
    else if (type == 2) { // Tolerance band balancing
        toleranceBandBalancing(time);
    }
    else if (type == 3) { // Predictive balancing
        predictiveBalancing(time, timeStep);
    }
}

void CapacitorBalancing::sortingBalancing(double time) {
    // Sorting-based algorithm:
    // 1. Sort capacitors by voltage
    // 2. Determine which ones to insert or bypass based on arm current direction

    // For demonstration, let's assume a simplified case with one arm
    // and all capacitors in that arm

    // Example arm current (in real implementation, would come from measurements)
    double armCurrent = sin(2.0 * M_PI * 60.0 * time);

    // Create a vector of indices for sorting
    std::vector<int> indices;
    for (size_t i = 0; i < capacitorVoltages.size(); ++i) {
        indices.push_back(static_cast<int>(i));
    }

    // Sort indices based on capacitor voltages
    if (armCurrent > 0) {
        // Sort descending if arm current is positive (capacitors are charging)
        std::sort(indices.begin(), indices.end(),
            [&](int a, int b) {
                return capacitorVoltages[a] > capacitorVoltages[b];
            });
    }
    else {
        // Sort ascending if arm current is negative (capacitors are discharging)
        std::sort(indices.begin(), indices.end(),
            [&](int a, int b) {
                return capacitorVoltages[a] < capacitorVoltages[b];
            });
    }

    // Determine number of submodules to insert
    // In real implementation, this would come from modulation algorithm
    int numToInsert = static_cast<int>(capacitorVoltages.size() / 2);

    // Update switching states based on sorting
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];

        // Check if we can switch based on frequency limit
        bool canSwitch = (time - switchingTimes[idx]) > (1.0 / switchingFreqLimit);

        if (canSwitch) {
            // Insert or bypass based on position in sorted list
            int newState = (i < static_cast<size_t>(numToInsert)) ? 1 : 0;

            // Only update if state changes
            if (switchingStates[idx] != newState) {
                switchingStates[idx] = newState;
                switchingTimes[idx] = time;
            }
        }
    }
}

void CapacitorBalancing::toleranceBandBalancing(double time) {
    // Tolerance band algorithm:
    // 1. Calculate average capacitor voltage
    // 2. Insert capacitors that deviate too much from average

    // Calculate average voltage
    double avgVoltage = 0.0;
    for (double voltage : capacitorVoltages) {
        avgVoltage += voltage;
    }
    avgVoltage /= capacitorVoltages.size();

    // Example arm current (in real implementation, would come from measurements)
    double armCurrent = sin(2.0 * M_PI * 60.0 * time);

    // Update switching states based on tolerance band
    for (size_t i = 0; i < capacitorVoltages.size(); ++i) {
        // Check deviation from average
        double deviation = capacitorVoltages[i] - avgVoltage;

        // Check if we can switch based on frequency limit
        bool canSwitch = (time - switchingTimes[i]) > (1.0 / switchingFreqLimit);

        if (canSwitch) {
            int newState = switchingStates[i]; // Default to no change

            if (armCurrent > 0) {
                // Charging - insert capacitors with low voltage
                if (deviation < -toleranceBand) {
                    newState = 1; // Insert
                }
                else if (deviation > toleranceBand) {
                    newState = 0; // Bypass
                }
            }
            else {
                // Discharging - insert capacitors with high voltage
                if (deviation > toleranceBand) {
                    newState = 1; // Insert
                }
                else if (deviation < -toleranceBand) {
                    newState = 0; // Bypass
                }
            }

            // Only update if state changes
            if (switchingStates[i] != newState) {
                switchingStates[i] = newState;
                switchingTimes[i] = time;
            }
        }
    }
}

void CapacitorBalancing::predictiveBalancing(double time, double timeStep) {
    // Predictive balancing algorithm:
    // 1. Predict future capacitor voltages based on current states
    // 2. Optimize switching decisions to minimize future voltage deviations

    // Example arm current (in real implementation, would come from measurements)
    double armCurrent = sin(2.0 * M_PI * 60.0 * time);

    // Calculate capacitance (assumed the same for all capacitors)
    double capacitance = 10e-3; // Example: 10 mF

    // Predict future voltages (simplified)
    std::vector<double> predictedVoltages = capacitorVoltages;
    for (size_t i = 0; i < predictedVoltages.size(); ++i) {
        if (switchingStates[i] == 1) { // If inserted
            // Predict voltage change: dV = I * dt / C
            predictedVoltages[i] += armCurrent * timeStep / capacitance;
        }
    }

    // Calculate future average voltage
    double futureAvgVoltage = 0.0;
    for (double voltage : predictedVoltages) {
        futureAvgVoltage += voltage;
    }
    futureAvgVoltage /= predictedVoltages.size();

    // Calculate predicted deviations
    std::vector<double> predictedDeviations(predictedVoltages.size());
    for (size_t i = 0; i < predictedVoltages.size(); ++i) {
        predictedDeviations[i] = predictedVoltages[i] - futureAvgVoltage;
    }

    // Calculate cost for each possible switching state
    struct SwitchingOption {
        size_t index;
        int state;
        double cost;
    };

    std::vector<SwitchingOption> options;

    // Determine number of submodules to insert
    // In real implementation, this would come from modulation algorithm
    int numToInsert = static_cast<int>(capacitorVoltages.size() / 2);

    // Evaluate cost for each submodule
    for (size_t i = 0; i < capacitorVoltages.size(); ++i) {
        // Check if we can switch based on frequency limit
        bool canSwitch = (time - switchingTimes[i]) > (1.0 / switchingFreqLimit);

        if (canSwitch) {
            // Calculate cost for inserting this submodule
            double insertCost = abs(predictedDeviations[i] + armCurrent * timeStep / capacitance);

            // Calculate cost for bypassing this submodule
            double bypassCost = abs(predictedDeviations[i]);

            // Add both options to list
            options.push_back({ i, 1, insertCost });
            options.push_back({ i, 0, bypassCost });
        }
        else {
            // Can't switch, so keep current state
            options.push_back({ i, switchingStates[i], abs(predictedDeviations[i]) });
        }
    }

    // Sort options by cost
    std::sort(options.begin(), options.end(),
        [](const SwitchingOption& a, const SwitchingOption& b) {
            return a.cost < b.cost;
        });

    // Select best options to meet numToInsert requirement
    int currentInsertCount = 0;
    std::vector<int> newStates(capacitorVoltages.size(), -1);

    // First pass: handle fixed options (can't switch)
    for (const auto& option : options) {
        if (time - switchingTimes[option.index] <= 1.0 / switchingFreqLimit) {
            newStates[option.index] = switchingStates[option.index];
            if (switchingStates[option.index] == 1) {
                currentInsertCount++;
            }
        }
    }

    // Second pass: handle switchable options
    for (const auto& option : options) {
        if (newStates[option.index] == -1) { // Not yet decided
            if (option.state == 1 && currentInsertCount < numToInsert) {
                newStates[option.index] = 1;
                currentInsertCount++;
            }
            else if (option.state == 0) {
                newStates[option.index] = 0;
            }
        }
    }

    // Final pass: force remaining decisions to meet numToInsert
    for (size_t i = 0; i < newStates.size(); ++i) {
        if (newStates[i] == -1) {
            if (currentInsertCount < numToInsert) {
                newStates[i] = 1;
                currentInsertCount++;
            }
            else {
                newStates[i] = 0;
            }
        }
    }

    // Update switching states
    for (size_t i = 0; i < switchingStates.size(); ++i) {
        if (switchingStates[i] != newStates[i]) {
            switchingStates[i] = newStates[i];
            switchingTimes[i] = time;
        }
    }
}

void CapacitorBalancing::setCapacitorVoltages(const std::vector<double>& voltages) {
    capacitorVoltages = voltages;
}

void CapacitorBalancing::prepareDeviceData() {
    // Make sure we have capacitor voltages to work with
    if (capacitorVoltages.empty()) {
        return;
    }

    // Initialize switching states and times if they don't match capacitor voltages
    if (switchingStates.size() != capacitorVoltages.size()) {
        switchingStates.resize(capacitorVoltages.size(), 0);
        switchingTimes.resize(capacitorVoltages.size(), 0.0);
    }

    // Allocate and initialize device memory for CUDA
    size_t numElements = capacitorVoltages.size();

    if (!d_capVoltages) {
        cudaMalloc(&d_capVoltages, numElements * sizeof(double));
    }

    if (!d_switchStates) {
        cudaMalloc(&d_switchStates, numElements * sizeof(int));
    }

    if (!d_switchTimes) {
        cudaMalloc(&d_switchTimes, numElements * sizeof(double));
    }

    // Copy initial data to device
    updateDeviceData();
}

void CapacitorBalancing::updateDeviceData() {
    size_t numElements = capacitorVoltages.size();

    if (d_capVoltages && d_switchStates && d_switchTimes && numElements > 0) {
        cudaMemcpy(d_capVoltages, capacitorVoltages.data(),
            numElements * sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_switchStates, switchingStates.data(),
            numElements * sizeof(int), cudaMemcpyHostToDevice);

        cudaMemcpy(d_switchTimes, switchingTimes.data(),
            numElements * sizeof(double), cudaMemcpyHostToDevice);
    }
}

void CapacitorBalancing::retrieveDeviceData() {
    size_t numElements = capacitorVoltages.size();

    if (d_capVoltages && d_switchStates && d_switchTimes && numElements > 0) {
        cudaMemcpy(capacitorVoltages.data(), d_capVoltages,
            numElements * sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(switchingStates.data(), d_switchStates,
            numElements * sizeof(int), cudaMemcpyDeviceToHost);

        cudaMemcpy(switchingTimes.data(), d_switchTimes,
            numElements * sizeof(double), cudaMemcpyDeviceToHost);
    }
}

// Factory function implementation
std::unique_ptr<ControlSystem> createControlSystem(const std::string& controlSystemSpec) {
    std::istringstream iss(controlSystemSpec);
    std::string controlType;
    iss >> controlType;

    if (controlType == "PLL") {
        std::string name;
        double bandwidth, dampingRatio, kp, ki, settlingTime, overshoot;
        int type;

        iss >> name >> bandwidth >> dampingRatio >> kp >> ki >> settlingTime >> overshoot >> type;

        return std::make_unique<PLL>(name, bandwidth, dampingRatio, kp, ki,
            settlingTime, overshoot, type);
    }
    else if (controlType == "CURR_CONT") {
        std::string name;
        int type;
        double bandwidth, dampingRatio, kp, ki, samplingFrequency;
        int predictionHorizon;

        iss >> name >> type >> bandwidth >> dampingRatio >> kp >> ki
            >> samplingFrequency >> predictionHorizon;

        return std::make_unique<CurrentController>(name, type, bandwidth, dampingRatio,
            kp, ki, samplingFrequency, predictionHorizon);
    }
    else if (controlType == "VOLT_CONT") {
        std::string name;
        int type;
        double bandwidth, dampingRatio, kp, ki, droopCoefficient;
        bool antiWindup;

        iss >> name >> type >> bandwidth >> dampingRatio >> kp >> ki
            >> droopCoefficient >> antiWindup;

        return std::make_unique<VoltageController>(name, type, bandwidth, dampingRatio,
            kp, ki, droopCoefficient, antiWindup);
    }
    else if (controlType == "POW_CONT") {
        std::string name;
        int type;
        double bandwidth, dampingRatio, kp, ki, pDroop, qDroop;

        iss >> name >> type >> bandwidth >> dampingRatio >> kp >> ki >> pDroop >> qDroop;

        return std::make_unique<PowerController>(name, type, bandwidth, dampingRatio,
            kp, ki, pDroop, qDroop);
    }
    else if (controlType == "DC_CONT") {
        std::string name;
        int type;
        double bandwidth, dampingRatio, kp, ki, droop;

        iss >> name >> type >> bandwidth >> dampingRatio >> kp >> ki >> droop;

        return std::make_unique<DCVoltageController>(name, type, bandwidth, dampingRatio,
            kp, ki, droop);
    }
    else if (controlType == "GFM_CONT") {
        std::string name;
        int type;
        double virtualInertia, droop, damping, synchronizingCoeff, lowPassFilterFreq;

        iss >> name >> type >> virtualInertia >> droop >> damping
            >> synchronizingCoeff >> lowPassFilterFreq;

        return std::make_unique<GridFormingController>(name, type, virtualInertia, droop,
            damping, synchronizingCoeff, lowPassFilterFreq);
    }
    else if (controlType == "CAP_BAL") {
        std::string name;
        int type;
        double toleranceBand, switchingFreqLimit, updateRate;

        iss >> name >> type >> toleranceBand >> switchingFreqLimit >> updateRate;

        return std::make_unique<CapacitorBalancing>(name, type, toleranceBand,
            switchingFreqLimit, updateRate);
    }

    // Unknown control system type
    std::cerr << "Unknown control system type: " << controlType << std::endl;
    return nullptr;
}