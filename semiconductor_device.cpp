#include "semiconductor_device.h"
#include "thermal_model.h"
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

// SemiconductorDevice base class implementation
SemiconductorDevice::SemiconductorDevice(const std::string& name, int id)
    : name(name), id(id), thermalModel(nullptr), temperature(25.0) {
}

// IGBT implementation
IGBT::IGBT(const std::string& name, double vce, double ic, double ron, double roff,
    double vf, double ton, double toff, double eon, double eoff, double tjMax)
    : SemiconductorDevice(name, -1), vce(vce), ic(ic), ron(ron), roff(roff),
    vf(vf), ton(ton), toff(toff), eon(eon), eoff(eoff), tjMax(tjMax),
    isOn(false), conductionLoss(0.0), switchingLoss(0.0), totalLoss(0.0),
    d_stateVars(nullptr) {
}

IGBT::~IGBT() {
    // Free device memory if allocated
    if (d_stateVars) {
        cudaFree(d_stateVars);
        d_stateVars = nullptr;
    }
}

void IGBT::calculateLosses(double voltage, double current, double timeStep) {
    // Update temperature from thermal model if available
    if (thermalModel) {
        temperature = thermalModel->getJunctionTemperature();
    }

    // Limit temperature to maximum allowed
    temperature = std::min(temperature, tjMax);

    // Apply temperature dependence to parameters
    // Typical temperature coefficients for IGBT parameters
    double vceTemp = vce * (1.0 + 0.003 * (temperature - 25.0)); // +0.3%/°C
    double ronTemp = ron * (1.0 + 0.01 * (temperature - 25.0));  // +1%/°C

    // Calculate conduction losses
    if (isOn) {
        // When ON, use saturation voltage and on-state resistance
        // P_cond = Vce0 * Ic + ron * Ic^2
        conductionLoss = vceTemp * current + ronTemp * current * current;
    }
    else {
        // When OFF, leakage losses are typically negligible
        conductionLoss = 0.0;
    }

    // Calculate switching losses
    // In a real implementation, this would depend on detailed switching waveforms
    // For simplicity, using a scaled energy-based approach

    // Switching losses only occur during transitions
    static double lastState = isOn;
    if (isOn != lastState) {
        if (isOn) {
            // Turn-on loss
            double actualEon = eon * (voltage / vce) * (current / ic);

            // Temperature adjustment for switching energy
            actualEon *= (1.0 + 0.003 * (temperature - 25.0)); // +0.3%/°C

            // Instantaneous switching loss
            switchingLoss = actualEon / timeStep;
        }
        else {
            // Turn-off loss
            double actualEoff = eoff * (voltage / vce) * (current / ic);

            // Temperature adjustment for switching energy
            actualEoff *= (1.0 + 0.005 * (temperature - 25.0)); // +0.5%/°C

            // Instantaneous switching loss
            switchingLoss = actualEoff / timeStep;
        }
    }
    else {
        // Apply decay to switching loss (it's transient)
        switchingLoss *= exp(-timeStep / 1.0e-6); // Decay with 1 μs time constant
    }

    lastState = isOn;

    // Calculate total losses
    totalLoss = conductionLoss + switchingLoss;

    // Update thermal model if available
    if (thermalModel) {
        thermalModel->update(totalLoss, timeStep);
    }
}

double IGBT::calculateVoltage(double current) const {
    if (isOn) {
        // When ON, use saturation model
        // V_ce = Vce0 + ron * Ic
        return vf + ron * current;
    }
    else {
        // When OFF, act as very high resistance
        return roff * current;
    }
}

double IGBT::calculateCurrent(double voltage) const {
    if (isOn) {
        // When ON, use saturation model
        // Ic = (V_ce - Vce0) / ron
        if (ron > 1.0e-12) { // Avoid division by zero
            return (voltage - vf) / ron;
        }
        else {
            return voltage > vf ? 1.0e6 : 0.0; // Very high current if Vce > Vf
        }
    }
    else {
        // When OFF, very low leakage current
        if (roff > 1.0e-12) { // Avoid division by zero
            return voltage / roff;
        }
        else {
            return 0.0;
        }
    }
}

void IGBT::prepareDeviceData() {
    // Allocate memory for device variables
    if (!d_stateVars) {
        cudaMalloc(&d_stateVars, 6 * sizeof(double));
    }

    // Copy initial data to device
    updateDeviceData();
}

void IGBT::updateDeviceData() {
    if (d_stateVars) {
        double stateVars[6] = {
            static_cast<double>(isOn),
            temperature,
            conductionLoss,
            switchingLoss,
            totalLoss,
            0.0 // Reserved
        };

        cudaMemcpy(d_stateVars, stateVars, 6 * sizeof(double),
            cudaMemcpyHostToDevice);
    }
}

void IGBT::retrieveDeviceData() {
    if (d_stateVars) {
        double stateVars[6];

        cudaMemcpy(stateVars, d_stateVars, 6 * sizeof(double),
            cudaMemcpyDeviceToHost);

        isOn = stateVars[0] > 0.5;
        temperature = stateVars[1];
        conductionLoss = stateVars[2];
        switchingLoss = stateVars[3];
        totalLoss = stateVars[4];
        // stateVars[5] is reserved
    }
}

// Diode implementation
Diode::Diode(const std::string& name, double vf, double ifRated, double ron, double roff,
    double trr, double err, double qrr, double tjMax)
    : SemiconductorDevice(name, -1), vf(vf), ifRated(ifRated), ron(ron), roff(roff),
    trr(trr), err(err), qrr(qrr), tjMax(tjMax),
    isOn(false), conductionLoss(0.0), recoveryLoss(0.0), totalLoss(0.0),
    d_stateVars(nullptr) {
}

Diode::~Diode() {
    // Free device memory if allocated
    if (d_stateVars) {
        cudaFree(d_stateVars);
        d_stateVars = nullptr;
    }
}

void Diode::calculateLosses(double voltage, double current, double timeStep) {
    // Update temperature from thermal model if available
    if (thermalModel) {
        temperature = thermalModel->getJunctionTemperature();
    }

    // Limit temperature to maximum allowed
    temperature = std::min(temperature, tjMax);

    // Apply temperature dependence to parameters
    // Typical temperature coefficients for diode parameters
    double vfTemp = vf * (1.0 - 0.002 * (temperature - 25.0)); // -0.2%/°C
    double ronTemp = ron * (1.0 + 0.01 * (temperature - 25.0)); // +1%/°C

    // Determine conduction state based on voltage and current
    // A diode conducts when forward biased (V > 0)
    bool wasOn = isOn;
    isOn = (voltage > 0.0);

    // Calculate conduction losses
    if (isOn) {
        // When conducting, use forward voltage drop and on-state resistance
        // P_cond = Vf * If + ron * If^2
        conductionLoss = vfTemp * current + ronTemp * current * current;
    }
    else {
        // When blocking, leakage losses are typically negligible
        conductionLoss = 0.0;
    }

    // Calculate recovery losses
    // In a real implementation, this would depend on detailed switching waveforms
    // For simplicity, using a scaled energy-based approach

    // Recovery losses only occur during turn-off
    if (wasOn && !isOn) {
        // Turn-off (reverse recovery) loss
        // Scale with rated conditions
        double actualErr = err * (voltage / (2.0 * vf)) * (current / ifRated);

        // Temperature adjustment for recovery energy
        actualErr *= (1.0 + 0.006 * (temperature - 25.0)); // +0.6%/°C

        // Instantaneous recovery loss
        recoveryLoss = actualErr / timeStep;
    }
    else {
        // Apply decay to recovery loss (it's transient)
        recoveryLoss *= exp(-timeStep / (2.0 * trr)); // Decay based on recovery time
    }

    // Calculate total losses
    totalLoss = conductionLoss + recoveryLoss;

    // Update thermal model if available
    if (thermalModel) {
        thermalModel->update(totalLoss, timeStep);
    }
}

double Diode::calculateVoltage(double current) const {
    if (current > 0.0) {
        // Forward conduction
        // V_f = Vf0 + ron * If
        return vf + ron * current;
    }
    else {
        // Reverse blocking
        return roff * current; // Very high resistance in reverse
    }
}

double Diode::calculateCurrent(double voltage) const {
    if (voltage > vf) {
        // Forward conduction
        // If = (V - Vf) / ron
        if (ron > 1.0e-12) { // Avoid division by zero
            return (voltage - vf) / ron;
        }
        else {
            return 1.0e6; // Very high current if no resistance
        }
    }
    else {
        // Reverse blocking or below threshold
        if (roff > 1.0e-12) { // Avoid division by zero
            return voltage / roff; // Small leakage current
        }
        else {
            return 0.0;
        }
    }
}

void Diode::prepareDeviceData() {
    // Allocate memory for device variables
    if (!d_stateVars) {
        cudaMalloc(&d_stateVars, 6 * sizeof(double));
    }

    // Copy initial data to device
    updateDeviceData();
}

void Diode::updateDeviceData() {
    if (d_stateVars) {
        double stateVars[6] = {
            static_cast<double>(isOn),
            temperature,
            conductionLoss,
            recoveryLoss,
            totalLoss,
            0.0 // Reserved
        };

        cudaMemcpy(d_stateVars, stateVars, 6 * sizeof(double),
            cudaMemcpyHostToDevice);
    }
}

void Diode::retrieveDeviceData() {
    if (d_stateVars) {
        double stateVars[6];

        cudaMemcpy(stateVars, d_stateVars, 6 * sizeof(double),
            cudaMemcpyDeviceToHost);

        isOn = stateVars[0] > 0.5;
        temperature = stateVars[1];
        conductionLoss = stateVars[2];
        recoveryLoss = stateVars[3];
        totalLoss = stateVars[4];
        // stateVars[5] is reserved
    }
}

// Thyristor implementation
Thyristor::Thyristor(const std::string& name, double vdrm, double itRated, double ron, double roff,
    double diDtCritical, double dvDtCritical, double turnOnDelay,
    double switchingEnergy, double holdingCurrent, double tjMax)
    : SemiconductorDevice(name, -1), vdrm(vdrm), itRated(itRated), ron(ron), roff(roff),
    diDtCritical(diDtCritical), dvDtCritical(dvDtCritical), turnOnDelay(turnOnDelay),
    switchingEnergy(switchingEnergy), holdingCurrent(holdingCurrent), tjMax(tjMax),
    isOn(false), isTriggered(false), conductionLoss(0.0), switchingLoss(0.0), totalLoss(0.0),
    d_stateVars(nullptr) {
}

Thyristor::~Thyristor() {
    // Free device memory if allocated
    if (d_stateVars) {
        cudaFree(d_stateVars);
        d_stateVars = nullptr;
    }
}

void Thyristor::calculateLosses(double voltage, double current, double timeStep) {
    // Update temperature from thermal model if available
    if (thermalModel) {
        temperature = thermalModel->getJunctionTemperature();
    }

    // Limit temperature to maximum allowed
    temperature = std::min(temperature, tjMax);

    // Apply temperature dependence to parameters
    // Typical temperature coefficients for thyristor parameters
    double ronTemp = ron * (1.0 + 0.01 * (temperature - 25.0)); // +1%/°C

    // Track previous state for switching detection
    bool wasOn = isOn;

    // Update state based on trigger, voltage, and current
    if (!isOn) {
        // Thyristor is OFF - check if it should turn ON

        // Check gate trigger
        if (isTriggered && voltage > 0.0) {
            // Triggered and forward biased - turn ON after delay
            static double triggerTime = 0.0;
            triggerTime += timeStep;

            if (triggerTime >= turnOnDelay) {
                isOn = true;
                triggerTime = 0.0;
            }
        }
        else {
            // Without gate trigger, check for dv/dt turn-on
            static double lastVoltage = voltage;
            double dvdt = (voltage - lastVoltage) / timeStep;

            if (voltage > 0.0 && dvdt > dvDtCritical) {
                isOn = true; // dv/dt triggered
            }

            lastVoltage = voltage;
        }
    }
    else {
        // Thyristor is ON - check if it should turn OFF

        // Thyristor turns off when current falls below holding current
        // AND voltage is reversed
        if (current < holdingCurrent && voltage < 0.0) {
            isOn = false;
        }
    }

    // Reset gate trigger
    isTriggered = false;

    // Calculate conduction losses
    if (isOn) {
        // When conducting, use forward voltage drop and on-state resistance
        // Similar to a diode: P_cond = Vf * It + ron * It^2
        double vt = 1.0; // Typical forward drop for thyristor
        conductionLoss = vt * current + ronTemp * current * current;
    }
    else {
        // When blocking, leakage losses are typically negligible
        conductionLoss = 0.0;
    }

    // Calculate switching losses
    // In a real implementation, this would depend on detailed switching waveforms

    // Switching losses only occur during turn-on
    if (!wasOn && isOn) {
        // Turn-on loss
        // Scale with rated conditions
        double actualEsw = switchingEnergy * (voltage / vdrm) * (current / itRated);

        // Temperature adjustment for switching energy
        actualEsw *= (1.0 + 0.004 * (temperature - 25.0)); // +0.4%/°C

        // Instantaneous switching loss
        switchingLoss = actualEsw / timeStep;
    }
    else {
        // Apply decay to switching loss (it's transient)
        switchingLoss *= exp(-timeStep / 1.0e-3); // Decay with 1 ms time constant
    }

    // Calculate total losses
    totalLoss = conductionLoss + switchingLoss;

    // Update thermal model if available
    if (thermalModel) {
        thermalModel->update(totalLoss, timeStep);
    }
}

double Thyristor::calculateVoltage(double current) const {
    if (isOn && current > 0.0) {
        // Forward conduction
        // V_t = Vt0 + ron * It
        double vt = 1.0; // Typical forward drop for thyristor
        return vt + ron * current;
    }
    else if (current <= 0.0) {
        // Reverse blocking or forward blocking with zero/negative current
        return roff * current; // Very high resistance when blocking
    }
    else {
        // Forward blocking with positive current
        // (unlikely in normal operation, but possible during simulation)
        return vdrm; // Limited by breakdown voltage
    }
}

double Thyristor::calculateCurrent(double voltage) const {
    if (isOn && voltage > 0.0) {
        // Forward conduction
        // It = (V - Vt) / ron
        double vt = 1.0; // Typical forward drop for thyristor
        if (ron > 1.0e-12) { // Avoid division by zero
            return (voltage - vt) / ron;
        }
        else {
            return 1.0e6; // Very high current if no resistance
        }
    }
    else {
        // Blocking (forward or reverse)
        if (roff > 1.0e-12) { // Avoid division by zero
            return voltage / roff; // Small leakage current
        }
        else {
            return 0.0;
        }
    }
}

void Thyristor::prepareDeviceData() {
    // Allocate memory for device variables
    if (!d_stateVars) {
        cudaMalloc(&d_stateVars, 6 * sizeof(double));
    }

    // Copy initial data to device
    updateDeviceData();
}

void Thyristor::updateDeviceData() {
    if (d_stateVars) {
        double stateVars[6] = {
            static_cast<double>(isOn),
            static_cast<double>(isTriggered),
            temperature,
            conductionLoss,
            switchingLoss,
            totalLoss
        };

        cudaMemcpy(d_stateVars, stateVars, 6 * sizeof(double),
            cudaMemcpyHostToDevice);
    }
}

void Thyristor::retrieveDeviceData() {
    if (d_stateVars) {
        double stateVars[6];

        cudaMemcpy(stateVars, d_stateVars, 6 * sizeof(double),
            cudaMemcpyDeviceToHost);

        isOn = stateVars[0] > 0.5;
        isTriggered = stateVars[1] > 0.5;
        temperature = stateVars[2];
        conductionLoss = stateVars[3];
        switchingLoss = stateVars[4];
        totalLoss = stateVars[5];
    }
}

// Factory function implementation
std::unique_ptr<SemiconductorDevice> createSemiconductorDevice(const std::string& deviceSpec) {
    std::istringstream iss(deviceSpec);
    std::string deviceType;
    iss >> deviceType;

    if (deviceType == "IGBT") {
        std::string name;
        double vce, ic, ron, roff, vf, ton, toff, eon, eoff, tjMax;

        iss >> name >> vce >> ic >> ron >> roff >> vf >> ton >> toff >> eon >> eoff >> tjMax;

        return std::make_unique<IGBT>(name, vce, ic, ron, roff, vf, ton, toff, eon, eoff, tjMax);
    }
    else if (deviceType == "DIODE") {
        std::string name;
        double vf, ifRated, ron, roff, trr, err, qrr, tjMax;

        iss >> name >> vf >> ifRated >> ron >> roff >> trr >> err >> qrr >> tjMax;

        return std::make_unique<Diode>(name, vf, ifRated, ron, roff, trr, err, qrr, tjMax);
    }
    else if (deviceType == "THYRISTOR") {
        std::string name;
        double vdrm, itRated, ron, roff, diDtCritical, dvDtCritical;
        double turnOnDelay, switchingEnergy, holdingCurrent, tjMax;

        iss >> name >> vdrm >> itRated >> ron >> roff >> diDtCritical >> dvDtCritical
            >> turnOnDelay >> switchingEnergy >> holdingCurrent >> tjMax;

        return std::make_unique<Thyristor>(name, vdrm, itRated, ron, roff,
            diDtCritical, dvDtCritical, turnOnDelay,
            switchingEnergy, holdingCurrent, tjMax);
    }

    // Unknown device type
    std::cerr << "Unknown semiconductor device type: " << deviceType << std::endl;
    return nullptr;
}