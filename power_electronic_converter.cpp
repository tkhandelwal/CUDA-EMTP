#include "power_electronic_converter.h"
#include "semiconductor_device.h"
#include "control_system.h"
#include "thermal_model.h"
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Base PowerElectronicConverter implementation
PowerElectronicConverter::PowerElectronicConverter(const std::string& name, int id)
    : name(name), id(id), ratedPower(0.0), ratedVoltage(0.0), switchingFrequency(0.0) {
}

void PowerElectronicConverter::addSemiconductor(SemiconductorDevice* device) {
    if (device) {
        semiconductors.push_back(device);
    }
}

void PowerElectronicConverter::addController(ControlSystem* controller) {
    if (controller) {
        controllers.push_back(controller);
    }
}

void PowerElectronicConverter::addThermalModel(ThermalModel* model) {
    if (model) {
        thermalModels.push_back(model);
    }
}

// VSC_MMC implementation
VSC_MMC::VSC_MMC(const std::string& name, int type, int numSubModules,
    double capacitancePerSM, double armInductance, double armResistance,
    int controlType, double dcLinkCapacitance, double dcVoltage,
    double switchingFreq, double carrierPhase, int pwmType)
    : PowerElectronicConverter(name, -1), type(type), numSubModules(numSubModules),
    capacitancePerSM(capacitancePerSM), armInductance(armInductance),
    armResistance(armResistance), controlType(controlType),
    dcLinkCapacitance(dcLinkCapacitance), dcVoltage(dcVoltage),
    carrierPhase(carrierPhase), pwmType(pwmType),
    d_submoduleVoltages(nullptr), d_armCurrents(nullptr), d_switchingStates(nullptr) {

    // Set switching frequency
    this->switchingFrequency = switchingFreq;

    // Set rated parameters based on DC voltage and submodules
    this->ratedVoltage = dcVoltage;
    this->ratedPower = dcVoltage * 1000.0; // Approximate rating based on DC voltage

    // Initialize submodule voltages to equal sharing of DC voltage
    double initialVoltage = dcVoltage * 1000.0 / numSubModules; // Convert to volts
    submoduleVoltages.resize(6 * numSubModules, initialVoltage);

    // Initialize arm currents to zero
    armCurrents.resize(6, 0.0); // 6 arms in a three-phase MMC

    // Initialize switching states to all off
    switchingStates.resize(6 * numSubModules, 0);
}

VSC_MMC::~VSC_MMC() {
    // Free device memory if allocated
    if (d_submoduleVoltages) {
        cudaFree(d_submoduleVoltages);
        d_submoduleVoltages = nullptr;
    }

    if (d_armCurrents) {
        cudaFree(d_armCurrents);
        d_armCurrents = nullptr;
    }

    if (d_switchingStates) {
        cudaFree(d_switchingStates);
        d_switchingStates = nullptr;
    }
}

void VSC_MMC::initialize() {
    // Reset submodule voltages to equal sharing of DC voltage
    double initialVoltage = dcVoltage * 1000.0 / numSubModules; // Convert to volts
    std::fill(submoduleVoltages.begin(), submoduleVoltages.end(), initialVoltage);

    // Reset arm currents to zero
    std::fill(armCurrents.begin(), armCurrents.end(), 0.0);

    // Reset switching states to all off
    std::fill(switchingStates.begin(), switchingStates.end(), 0);

    // Initialize all connected controllers
    for (auto controller : controllers) {
        controller->initialize();
    }
}

void VSC_MMC::update(double time, double timeStep) {
    // Update controllers first
    for (auto controller : controllers) {
        controller->update(time, timeStep);
    }

    // Determine required number of inserted submodules for each arm
    // based on controller outputs (in a real implementation, would
    // process modulation indices and reference voltages)

    // Update capacitor voltages of each submodule
    // This would involve complex equations involving arm currents
    // and switching states. For brevity, using a simplified model here:
    for (int arm = 0; arm < 6; arm++) {
        for (int sm = 0; sm < numSubModules; sm++) {
            int idx = arm * numSubModules + sm;
            if (switchingStates[idx] == 1) { // If SM is inserted
                double current = armCurrents[arm];
                // Simplified capacitor voltage update
                double dv = (current * timeStep) / capacitancePerSM;
                submoduleVoltages[idx] += dv;
            }
        }
    }

    // Balance capacitor voltages if requested
    balanceCapacitorVoltages();

    // Update circulating current control if applicable
    updateCirculatingCurrentsControl();
}

void VSC_MMC::calculateLosses() {
    // Calculate losses in each semiconductor device
    for (auto device : semiconductors) {
        // This is a simplified approach. In reality, we would:
        // 1. Determine voltage across and current through each device
        // 2. Check if device is on or off based on switching states
        // 3. Calculate conduction and switching losses

        // Example for now (not actually valid):
        double voltage = 1000.0; // Placeholder
        double current = 100.0;  // Placeholder
        device->calculateLosses(voltage, current, 1.0e-6);
    }
}

void VSC_MMC::prepareDeviceData() {
    // Allocate and initialize device memory for CUDA
    if (!d_submoduleVoltages) {
        cudaMalloc(&d_submoduleVoltages, 6 * numSubModules * sizeof(double));
    }

    if (!d_armCurrents) {
        cudaMalloc(&d_armCurrents, 6 * sizeof(double));
    }

    if (!d_switchingStates) {
        cudaMalloc(&d_switchingStates, 6 * numSubModules * sizeof(int));
    }

    // Copy initial data to device
    updateDeviceData();
}

void VSC_MMC::updateDeviceData() {
    if (d_submoduleVoltages && d_armCurrents && d_switchingStates) {
        cudaMemcpy(d_submoduleVoltages, submoduleVoltages.data(),
            6 * numSubModules * sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_armCurrents, armCurrents.data(),
            6 * sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_switchingStates, switchingStates.data(),
            6 * numSubModules * sizeof(int), cudaMemcpyHostToDevice);
    }
}

void VSC_MMC::retrieveDeviceData() {
    if (d_submoduleVoltages && d_armCurrents && d_switchingStates) {
        cudaMemcpy(submoduleVoltages.data(), d_submoduleVoltages,
            6 * numSubModules * sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(armCurrents.data(), d_armCurrents,
            6 * sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(switchingStates.data(), d_switchingStates,
            6 * numSubModules * sizeof(int), cudaMemcpyDeviceToHost);
    }
}

void VSC_MMC::balanceCapacitorVoltages() {
    // This is a simplified implementation of capacitor voltage balancing
    // In real MMCs, this would involve sophisticated algorithms like sorting
    // or tolerance band methods

    // Example: Simple sorting-based balancing
    for (int arm = 0; arm < 6; arm++) {
        // Extract capacitor voltages for this arm
        std::vector<std::pair<double, int>> sortedVoltages;
        for (int sm = 0; sm < numSubModules; sm++) {
            int idx = arm * numSubModules + sm;
            sortedVoltages.push_back({ submoduleVoltages[idx], sm });
        }

        // Sort based on voltage
        if (armCurrents[arm] > 0) {
            // Sort descending if arm current is positive
            std::sort(sortedVoltages.begin(), sortedVoltages.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
        }
        else {
            // Sort ascending if arm current is negative
            std::sort(sortedVoltages.begin(), sortedVoltages.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
        }

        // Calculate number of SMs to be inserted based on reference
        int nInsert = numSubModules / 2; // Simplified example

        // Update switching states
        for (int sm = 0; sm < numSubModules; sm++) {
            int submodule = sortedVoltages[sm].second;
            int idx = arm * numSubModules + submodule;

            // Insert submodules based on sorting
            if (sm < nInsert) {
                switchingStates[idx] = 1; // ON
            }
            else {
                switchingStates[idx] = 0; // OFF
            }
        }
    }
}

void VSC_MMC::updateCirculatingCurrentsControl() {
    // Circulating current is the common-mode current between the upper
    // and lower arms of the same phase

    // For each phase (a, b, c)
    for (int phase = 0; phase < 3; phase++) {
        int upperArm = phase;
        int lowerArm = phase + 3;

        // Calculate circulating current
        double iCirc = (armCurrents[upperArm] + armCurrents[lowerArm]) / 2.0;

        // In a real implementation, would apply control to suppress
        // the circulating current here
    }
}

// BESS_VSC implementation
BESS_VSC::BESS_VSC(const std::string& name, double capacity, double power, int batteryType,
    int cellsSeries, int cellsParallel, double cellVoltage, double cellCapacity,
    int converterType, double responseTime, int controlMode)
    : PowerElectronicConverter(name, -1), capacity(capacity), power(power), batteryType(batteryType),
    cellsSeries(cellsSeries), cellsParallel(cellsParallel), cellVoltage(cellVoltage),
    cellCapacity(cellCapacity), converterType(converterType), responseTime(responseTime),
    controlMode(controlMode), stateOfCharge(85.0), batteryVoltage(0.0), batteryCurrent(0.0),
    batteryTemperature(25.0), d_batteryStateVars(nullptr) {

    // Set rated parameters
    this->ratedPower = power;
    this->ratedVoltage = cellVoltage * cellsSeries;
    this->switchingFrequency = 5000.0; // Default 5 kHz

    // Calculate initial battery voltage based on SoC
    batteryVoltage = calculateBatteryVoltage(stateOfCharge);
}

BESS_VSC::~BESS_VSC() {
    // Free device memory if allocated
    if (d_batteryStateVars) {
        cudaFree(d_batteryStateVars);
        d_batteryStateVars = nullptr;
    }
}

void BESS_VSC::initialize() {
    // Reset battery state
    stateOfCharge = 85.0; // Default initial SoC
    batteryVoltage = calculateBatteryVoltage(stateOfCharge);
    batteryCurrent = 0.0;
    batteryTemperature = 25.0;

    // Initialize all connected controllers
    for (auto controller : controllers) {
        controller->initialize();
    }
}

double BESS_VSC::calculateBatteryVoltage(double soc) {
    // Simplified battery voltage model based on SoC
    // In a real model, this would be more sophisticated and depend on
    // battery chemistry, temperature, and actual SoC-OCV curves

    double nominalVoltage = cellVoltage * cellsSeries;

    if (batteryType == 1) {
        // Li-Ion model (simplified)
        if (soc > 90.0) {
            return nominalVoltage * (1.0 + 0.05 * (soc - 90.0) / 10.0);
        }
        else if (soc > 20.0) {
            return nominalVoltage;
        }
        else {
            return nominalVoltage * (1.0 - 0.15 * (20.0 - soc) / 20.0);
        }
    }
    else if (batteryType == 2) {
        // Flow battery model (simplified)
        return nominalVoltage * (0.8 + 0.4 * soc / 100.0);
    }
    else {
        // Default
        return nominalVoltage;
    }
}

void BESS_VSC::update(double time, double timeStep) {
    // Update controllers first
    for (auto controller : controllers) {
        controller->update(time, timeStep);
    }

    // Get power command from controller
    double powerCommand = 0.0; // Default

    // In a real implementation, we would get the power command from the appropriate controller
    // Here, using a simplified approach with a sine wave for demonstration
    double period = 600.0; // 10-minute cycle
    powerCommand = power * 0.5 * sin(2.0 * M_PI * time / period);

    // Calculate battery current based on power and voltage
    if (batteryVoltage > 1.0) { // Avoid division by zero
        batteryCurrent = powerCommand * 1.0e6 / batteryVoltage; // P = V * I
    }
    else {
        batteryCurrent = 0.0;
    }

    // Update state of charge
    double totalCapacity = cellCapacity * cellsParallel * 3600.0; // Convert to Coulombs
    double dSoC = (batteryCurrent * timeStep) / totalCapacity * 100.0; // Percent change
    stateOfCharge -= dSoC; // Discharge is positive current, reduces SoC

    // Limit SoC to valid range
    stateOfCharge = std::max(0.0, std::min(100.0, stateOfCharge));

    // Update battery voltage based on new SoC
    batteryVoltage = calculateBatteryVoltage(stateOfCharge);

    // Update battery temperature (simplified)
    double heatGenerated = batteryCurrent * batteryCurrent * 0.01; // I²R losses
    batteryTemperature += heatGenerated * timeStep / (capacity * 1.0e6); // Simple thermal model

    // Simulate cooling
    batteryTemperature += (25.0 - batteryTemperature) * 0.001 * timeStep;
}

void BESS_VSC::calculateLosses() {
    // Calculate losses in each semiconductor device
    for (auto device : semiconductors) {
        // This is a simplified approach.
        double voltage = batteryVoltage / 2.0; // Approximate DC link voltage across device
        double current = batteryCurrent / 2.0; // Approximate current through device
        device->calculateLosses(voltage, current, 1.0e-6);
    }

    // Battery losses (simplified)
    double batteryLosses = batteryCurrent * batteryCurrent * 0.01; // I²R losses in battery
}

void BESS_VSC::prepareDeviceData() {
    // Allocate and initialize device memory for CUDA
    if (!d_batteryStateVars) {
        cudaMalloc(&d_batteryStateVars, 4 * sizeof(double)); // 4 state variables
    }

    // Copy initial data to device
    updateDeviceData();
}

void BESS_VSC::updateDeviceData() {
    if (d_batteryStateVars) {
        double stateVars[4] = {
            stateOfCharge,
            batteryVoltage,
            batteryCurrent,
            batteryTemperature
        };

        cudaMemcpy(d_batteryStateVars, stateVars, 4 * sizeof(double),
            cudaMemcpyHostToDevice);
    }
}

void BESS_VSC::retrieveDeviceData() {
    if (d_batteryStateVars) {
        double stateVars[4];

        cudaMemcpy(stateVars, d_batteryStateVars, 4 * sizeof(double),
            cudaMemcpyDeviceToHost);

        stateOfCharge = stateVars[0];
        batteryVoltage = stateVars[1];
        batteryCurrent = stateVars[2];
        batteryTemperature = stateVars[3];
    }
}

// SOLAR_INV implementation
SOLAR_INV::SOLAR_INV(const std::string& name, int type, double rating, int controlType,
    int topology, double dcLinkCapacitance, double switchingFreq,
    double filterInductance, double filterCapacitance, double filterDampingR)
    : PowerElectronicConverter(name, -1), type(type), rating(rating), controlType(controlType),
    topology(topology), dcLinkCapacitance(dcLinkCapacitance),
    filterInductance(filterInductance), filterCapacitance(filterCapacitance),
    filterDampingR(filterDampingR), dcVoltage(0.0), dcCurrent(0.0), d_stateVars(nullptr) {

    // Set rated parameters
    this->ratedPower = rating;
    this->ratedVoltage = 480.0; // Default AC voltage for solar inverter
    this->switchingFrequency = switchingFreq;

    // Initialize AC currents and voltages
    acCurrents.resize(3, 0.0); // Three-phase currents
    acVoltages.resize(3, 0.0); // Three-phase voltages

    // Initialize DC voltage
    if (type == 1) { // String inverter
        dcVoltage = 600.0; // Typical string voltage
    }
    else { // Central inverter
        dcVoltage = 1000.0; // Typical central inverter DC voltage
    }
}

SOLAR_INV::~SOLAR_INV() {
    // Free device memory if allocated
    if (d_stateVars) {
        cudaFree(d_stateVars);
        d_stateVars = nullptr;
    }
}

void SOLAR_INV::initialize() {
    // Reset state variables
    dcVoltage = (type == 1) ? 600.0 : 1000.0;
    dcCurrent = 0.0;
    std::fill(acCurrents.begin(), acCurrents.end(), 0.0);
    std::fill(acVoltages.begin(), acVoltages.end(), 0.0);

    // Initialize all connected controllers
    for (auto controller : controllers) {
        controller->initialize();
    }
}

void SOLAR_INV::update(double time, double timeStep) {
    // Update controllers first
    for (auto controller : controllers) {
        controller->update(time, timeStep);
    }

    // Simulate solar irradiance based on time of day
    // This is a simplified model that assumes the simulation is running
    // in a 24-hour cycle with peak at noon
    double dayFraction = fmod(time, 86400.0) / 86400.0; // Fraction of day (0-1)
    double hourAngle = (dayFraction * 24.0 - 12.0) * 15.0; // Hour angle in degrees

    // Solar irradiance model (simplified)
    double irradiance = 0.0;
    if (dayFraction > 0.25 && dayFraction < 0.75) { // Between 6 AM and 6 PM
        irradiance = 1000.0 * cos(hourAngle * M_PI / 180.0);
        irradiance = std::max(0.0, irradiance);
    }

    // Add some cloud cover effects (random)
    double cloudEffect = 1.0;
    if (fmod(time, 600.0) < 300.0) { // Cloud passes by every 10 minutes
        cloudEffect = 0.3 + 0.7 * (fmod(time, 300.0) / 300.0);
    }
    irradiance *= cloudEffect;

    // Calculate DC current based on irradiance (simplified PV model)
    double maxPower = rating * 1.0e6; // Maximum power in W
    double mpptEfficiency = 0.98; // MPPT efficiency

    // Calculate available DC power based on irradiance
    double availableDcPower = maxPower * (irradiance / 1000.0) * mpptEfficiency;

    // Calculate DC current based on power and voltage
    if (dcVoltage > 10.0) {
        dcCurrent = availableDcPower / dcVoltage;
    }
    else {
        dcCurrent = 0.0;
    }

    // Update DC voltage based on capacitance and current
    // In a real implementation, this would involve complex interaction
    // with the MPPT algorithm

    // Update AC currents and voltages based on control mode
    if (controlType == 1) { // Grid-following
        // In grid-following mode, the inverter synchronizes with grid
        // and injects current based on available power

        // Example AC currents in balanced three-phase system
        double peakCurrent = sqrt(2.0) * availableDcPower / (sqrt(3.0) * ratedVoltage);

        for (int phase = 0; phase < 3; phase++) {
            double angle = 2.0 * M_PI * 60.0 * time + phase * (2.0 * M_PI / 3.0);
            acCurrents[phase] = peakCurrent * sin(angle);
        }

        // Grid voltage (assumed stiff)
        double peakVoltage = sqrt(2.0) * ratedVoltage / sqrt(3.0);
        for (int phase = 0; phase < 3; phase++) {
            double angle = 2.0 * M_PI * 60.0 * time + phase * (2.0 * M_PI / 3.0);
            acVoltages[phase] = peakVoltage * sin(angle);
        }
    }
    else { // Grid-forming
        // In grid-forming mode, the inverter sets the voltage and frequency

        // Example implementation - virtual oscillator
        double omega = 2.0 * M_PI * 60.0; // Nominal frequency

        // Get frequency adjustment from controllers (if any)
        double freqAdjust = 0.0;
        // In real implementation, would get this from a grid-forming controller

        // Calculate phase angle with adjusted frequency
        double angle = omega * (1.0 + freqAdjust) * time;

        // Set voltage amplitude based on available power
        double voltageSetpoint = ratedVoltage;

        // In real implementation, would get voltage setpoint from controller

        // Calculate three-phase voltages
        double peakVoltage = sqrt(2.0) * voltageSetpoint / sqrt(3.0);
        for (int phase = 0; phase < 3; phase++) {
            double phaseAngle = angle + phase * (2.0 * M_PI / 3.0);
            acVoltages[phase] = peakVoltage * sin(phaseAngle);
        }

        // Calculate currents based on local load (simplified)
        double loadImpedance = (voltageSetpoint * voltageSetpoint) / availableDcPower;
        double peakCurrent = peakVoltage / loadImpedance;

        for (int phase = 0; phase < 3; phase++) {
            double phaseAngle = angle + phase * (2.0 * M_PI / 3.0);
            acCurrents[phase] = peakCurrent * sin(phaseAngle);
        }
    }
}

void SOLAR_INV::calculateLosses() {
    // Calculate losses in each semiconductor device
    for (auto device : semiconductors) {
        // For IGBTs in the inverter
        double voltage = dcVoltage / 2.0; // Approximate voltage across device
        double current = acCurrents[0]; // Approximate current (using phase A as example)
        device->calculateLosses(voltage, current, 1.0e-6);
    }

    // Additional losses in filters and other components (simplified)
    double filterLosses = 0.0;
    for (int phase = 0; phase < 3; phase++) {
        // I²R losses in filter inductor
        filterLosses += acCurrents[phase] * acCurrents[phase] * 0.01;

        // Losses in damping resistor
        double capacitorCurrent = acVoltages[phase] * 2.0 * M_PI * 60.0 * filterCapacitance;
        filterLosses += capacitorCurrent * capacitorCurrent * filterDampingR;
    }
}

void SOLAR_INV::prepareDeviceData() {
    // Allocate and initialize device memory for CUDA
    if (!d_stateVars) {
        // 2 DC variables + 6 AC variables (3 currents, 3 voltages)
        cudaMalloc(&d_stateVars, 8 * sizeof(double));
    }

    // Copy initial data to device
    updateDeviceData();
}

void SOLAR_INV::updateDeviceData() {
    if (d_stateVars) {
        double stateVars[8];

        // DC variables
        stateVars[0] = dcVoltage;
        stateVars[1] = dcCurrent;

        // AC variables
        for (int i = 0; i < 3; i++) {
            stateVars[2 + i] = acCurrents[i];
            stateVars[5 + i] = acVoltages[i];
        }

        cudaMemcpy(d_stateVars, stateVars, 8 * sizeof(double),
            cudaMemcpyHostToDevice);
    }
}

void SOLAR_INV::retrieveDeviceData() {
    if (d_stateVars) {
        double stateVars[8];

        cudaMemcpy(stateVars, d_stateVars, 8 * sizeof(double),
            cudaMemcpyDeviceToHost);

        // DC variables
        dcVoltage = stateVars[0];
        dcCurrent = stateVars[1];

        // AC variables
        for (int i = 0; i < 3; i++) {
            acCurrents[i] = stateVars[2 + i];
            acVoltages[i] = stateVars[5 + i];
        }
    }
}

// STATCOM_MMC implementation
STATCOM_MMC::STATCOM_MMC(const std::string& name, int numSubModules, double capacitancePerSM,
    double armInductance, double armResistance, double rating,
    int controlType, double responseTime, double switchingFreq)
    : PowerElectronicConverter(name, -1), numSubModules(numSubModules),
    capacitancePerSM(capacitancePerSM), armInductance(armInductance),
    armResistance(armResistance), rating(rating), controlType(controlType),
    responseTime(responseTime),
    d_submoduleVoltages(nullptr), d_armCurrents(nullptr), d_switchingStates(nullptr) {

    // Set switching frequency
    this->switchingFrequency = switchingFreq;

    // Set rated parameters
    this->ratedPower = rating;
    this->ratedVoltage = 13.8; // Default for medium voltage STATCOM

    // Initialize submodule voltages to nominal value
    double initialVoltage = 1500.0; // Typical submodule voltage
    submoduleVoltages.resize(6 * numSubModules, initialVoltage);

    // Initialize arm currents to zero
    armCurrents.resize(6, 0.0); // 6 arms in a three-phase MMC

    // Initialize switching states to all off
    switchingStates.resize(6 * numSubModules, 0);
}

STATCOM_MMC::~STATCOM_MMC() {
    // Free device memory if allocated
    if (d_submoduleVoltages) {
        cudaFree(d_submoduleVoltages);
        d_submoduleVoltages = nullptr;
    }

    if (d_armCurrents) {
        cudaFree(d_armCurrents);
        d_armCurrents = nullptr;
    }

    if (d_switchingStates) {
        cudaFree(d_switchingStates);
        d_switchingStates = nullptr;
    }
}

void STATCOM_MMC::initialize() {
    // Reset submodule voltages to nominal value
    double initialVoltage = 1500.0; // Typical submodule voltage
    std::fill(submoduleVoltages.begin(), submoduleVoltages.end(), initialVoltage);

    // Reset arm currents to zero
    std::fill(armCurrents.begin(), armCurrents.end(), 0.0);

    // Reset switching states to all off
    std::fill(switchingStates.begin(), switchingStates.end(), 0);

    // Initialize all connected controllers
    for (auto controller : controllers) {
        controller->initialize();
    }
}

void STATCOM_MMC::update(double time, double timeStep) {
    // Update controllers first
    for (auto controller : controllers) {
        controller->update(time, timeStep);
    }

    // Get reactive power command from controller based on control type
    double qCommand = 0.0;

    if (controlType == 1) { // Voltage control
        // In voltage control mode, the STATCOM regulates the voltage
        // at its point of connection by injecting/absorbing reactive power

        // Example: Measure voltage and compare to setpoint
        double measuredVoltage = 1.0; // per unit (dummy value)
        double voltageSetpoint = 1.0; // per unit

        // In a real implementation, would get these from measurements and controller

        // Proportional control for Q based on voltage error
        double voltageError = voltageSetpoint - measuredVoltage;
        qCommand = 100.0 * voltageError * rating; // Simple proportional control
    }
    else if (controlType == 2) { // Reactive power control
        // In Q control mode, the STATCOM follows a reactive power setpoint

        // Example: Sinusoidal variation of Q for demonstration
        qCommand = rating * 0.5 * sin(2.0 * M_PI * time / 600.0); // 10-minute cycle
    }
    else if (controlType == 3) { // Power factor control
        // In PF control mode, the STATCOM maintains a target power factor
        // at its point of connection

        // Example: Assume we need to correct a measured power factor
        double measuredPF = 0.85; // Dummy value
        double targetPF = 0.98;

        // Calculate Q needed to correct power factor
        double activeLoad = 100.0; // MW (dummy value)
        double currentQ = activeLoad * tan(acos(measuredPF));
        double targetQ = activeLoad * tan(acos(targetPF));

        qCommand = currentQ - targetQ;
    }

    // Limit Q command to STATCOM rating
    qCommand = std::max(-rating, std::min(rating, qCommand));

    // Calculate required currents for desired reactive power
    double lineVoltage = ratedVoltage * 1000.0; // Convert to V
    double phaseVoltage = lineVoltage / sqrt(3.0);

    // Calculate required phase current for reactive power
    double currentMagnitude = qCommand * 1.0e6 / (3.0 * phaseVoltage);

    // Generate three-phase currents
    std::vector<double> phaseCurrents(3);
    for (int phase = 0; phase < 3; phase++) {
        double angle = 2.0 * M_PI * 60.0 * time + phase * (2.0 * M_PI / 3.0) + (M_PI / 2.0); // +90° for purely reactive
        phaseCurrents[phase] = currentMagnitude * sin(angle);
    }

    // Convert phase currents to arm currents
    // In a real implementation, this would involve complex calculations
    // that account for the MMC topology and control strategy

    // Simplified model for demonstration
    armCurrents[0] = phaseCurrents[0] / 2.0;  // Upper arm of phase A
    armCurrents[3] = -phaseCurrents[0] / 2.0; // Lower arm of phase A
    armCurrents[1] = phaseCurrents[1] / 2.0;  // Upper arm of phase B
    armCurrents[4] = -phaseCurrents[1] / 2.0; // Lower arm of phase B
    armCurrents[2] = phaseCurrents[2] / 2.0;  // Upper arm of phase C
    armCurrents[5] = -phaseCurrents[2] / 2.0; // Lower arm of phase C

    // Update submodule capacitor voltages
    for (int arm = 0; arm < 6; arm++) {
        for (int sm = 0; sm < numSubModules; sm++) {
            int idx = arm * numSubModules + sm;
            if (switchingStates[idx] == 1) { // If SM is inserted
                // Update capacitor voltage based on arm current
                double dv = (armCurrents[arm] * timeStep) / capacitancePerSM;
                submoduleVoltages[idx] += dv;
            }
        }
    }

    // Balance submodule capacitor voltages
    balanceCapacitorVoltages();
}

void STATCOM_MMC::calculateLosses() {
    // Calculate losses in each semiconductor device
    for (auto device : semiconductors) {
        // Simplified approach for demonstration
        double voltage = 1500.0; // Approximate voltage across device
        double current = 100.0;  // Approximate current through device
        device->calculateLosses(voltage, current, 1.0e-6);
    }

    // Additional losses in arm inductors and other components
    double armInductorLosses = 0.0;
    for (int arm = 0; arm < 6; arm++) {
        // I²R losses in arm inductor
        armInductorLosses += armCurrents[arm] * armCurrents[arm] * armResistance;
    }
}

void STATCOM_MMC::prepareDeviceData() {
    // Allocate and initialize device memory for CUDA
    if (!d_submoduleVoltages) {
        cudaMalloc(&d_submoduleVoltages, 6 * numSubModules * sizeof(double));
    }

    if (!d_armCurrents) {
        cudaMalloc(&d_armCurrents, 6 * sizeof(double));
    }

    if (!d_switchingStates) {
        cudaMalloc(&d_switchingStates, 6 * numSubModules * sizeof(int));
    }

    // Copy initial data to device
    updateDeviceData();
}

void STATCOM_MMC::updateDeviceData() {
    if (d_submoduleVoltages && d_armCurrents && d_switchingStates) {
        cudaMemcpy(d_submoduleVoltages, submoduleVoltages.data(),
            6 * numSubModules * sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_armCurrents, armCurrents.data(),
            6 * sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_switchingStates, switchingStates.data(),
            6 * numSubModules * sizeof(int), cudaMemcpyHostToDevice);
    }
}

void STATCOM_MMC::retrieveDeviceData() {
    if (d_submoduleVoltages && d_armCurrents && d_switchingStates) {
        cudaMemcpy(submoduleVoltages.data(), d_submoduleVoltages,
            6 * numSubModules * sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(armCurrents.data(), d_armCurrents,
            6 * sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(switchingStates.data(), d_switchingStates,
            6 * numSubModules * sizeof(int), cudaMemcpyDeviceToHost);
    }
}

void STATCOM_MMC::balanceCapacitorVoltages() {
    // Implement the same capacitor voltage balancing method as in VSC_MMC
    // This is a simplified version using sorting algorithm

    for (int arm = 0; arm < 6; arm++) {
        // Extract capacitor voltages for this arm
        std::vector<std::pair<double, int>> sortedVoltages;
        for (int sm = 0; sm < numSubModules; sm++) {
            int idx = arm * numSubModules + sm;
            sortedVoltages.push_back(std::make_pair(submoduleVoltages[idx], sm));
        }

        // Sort based on voltage
        if (armCurrents[arm] > 0) {
            // Sort descending if arm current is positive
            std::sort(sortedVoltages.begin(), sortedVoltages.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
        }
        else {
            // Sort ascending if arm current is negative
            std::sort(sortedVoltages.begin(), sortedVoltages.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
        }

        // Calculate number of SMs to be inserted based on reference
        // In a real implementation, this would be based on reference voltage
        int nInsert = numSubModules / 2; // Simplified example

        // Update switching states
        for (int sm = 0; sm < numSubModules; sm++) {
            int submodule = sortedVoltages[sm].second;
            int idx = arm * numSubModules + submodule;

            // Insert submodules based on sorting
            if (sm < nInsert) {
                switchingStates[idx] = 1; // ON
            }
            else {
                switchingStates[idx] = 0; // OFF
            }
        }
    }
}

// WIND_CONV implementation
WIND_CONV::WIND_CONV(const std::string& name, int type, double rating, int gridSideControl,
    int genSideControl, double dcLinkCapacitance, double dcVoltage,
    double gscSwitchingFreq, double mscSwitchingFreq)
    : PowerElectronicConverter(name, -1), type(type), rating(rating),
    gridSideControl(gridSideControl), genSideControl(genSideControl),
    dcLinkCapacitance(dcLinkCapacitance), dcVoltage(dcVoltage),
    gscSwitchingFreq(gscSwitchingFreq), mscSwitchingFreq(mscSwitchingFreq),
    dcCurrent(0.0), d_stateVars(nullptr) {

    // Set rated parameters
    this->ratedPower = rating;
    this->ratedVoltage = 690.0; // Default for wind turbines
    this->switchingFrequency = gscSwitchingFreq; // Use grid-side switching frequency

    // Initialize currents
    gridSideCurrents.resize(3, 0.0); // Three-phase grid-side currents
    genSideCurrents.resize(3, 0.0);  // Three-phase generator-side currents
}

WIND_CONV::~WIND_CONV() {
    // Free device memory if allocated
    if (d_stateVars) {
        cudaFree(d_stateVars);
        d_stateVars = nullptr;
    }
}

void WIND_CONV::initialize() {
    // Reset state variables
    dcCurrent = 0.0;
    std::fill(gridSideCurrents.begin(), gridSideCurrents.end(), 0.0);
    std::fill(genSideCurrents.begin(), genSideCurrents.end(), 0.0);

    // Initialize all connected controllers
    for (auto controller : controllers) {
        controller->initialize();
    }
}

void WIND_CONV::update(double time, double timeStep) {
    // Update controllers first
    for (auto controller : controllers) {
        controller->update(time, timeStep);
    }

    // Simulate wind speed based on time
    // This is a simplified model with base wind speed plus turbulence
    double baseWindSpeed = 10.0; // m/s
    double turbulenceAmplitude = 2.0; // m/s
    double turbulencePeriod1 = 10.0; // seconds
    double turbulencePeriod2 = 60.0; // seconds

    double windSpeed = baseWindSpeed +
        turbulenceAmplitude * 0.7 * sin(2.0 * M_PI * time / turbulencePeriod1) +
        turbulenceAmplitude * 0.3 * sin(2.0 * M_PI * time / turbulencePeriod2);

    // Apply slow ramp changes to base wind speed
    if (time > 300.0 && time < 600.0) {
        // Ramp up
        double rampFactor = (time - 300.0) / 300.0;
        windSpeed += 5.0 * rampFactor;
    }
    else if (time > 900.0 && time < 1200.0) {
        // Ramp down
        double rampFactor = (time - 900.0) / 300.0;
        windSpeed -= 7.0 * rampFactor;
    }

    // Wind power calculation (simplified cubic relationship)
    // P = 0.5 * rho * A * Cp * v^3
    double airDensity = 1.225; // kg/m³
    double rotorArea = 2000.0; // m² (approx. 50m diameter)
    double powerCoefficient = 0.4; // Typical Cp

    double theoreticalPower = 0.5 * airDensity * rotorArea * powerCoefficient * pow(windSpeed, 3);

    // Limit to rated power
    double windPower = std::min(theoreticalPower, rating * 1.0e6);

    // Apply generator efficiency
    double generatorEfficiency = 0.95;
    double electricalPower = windPower * generatorEfficiency;

    // Handle different converter types
    if (type == 1) { // DFIG
        // For DFIG, only a portion of power goes through converter
        electricalPower *= 0.3; // Typically around 30%
    }

    // Calculate generator-side currents based on power and control mode
    if (genSideControl == 1) { // Speed control
        // In speed control mode, the generator-side converter
        // controls the rotor speed to extract maximum power

        // Calculate optimal speed for this wind speed (simplified)
        double optimalTipSpeedRatio = 7.0; // Typical value
        double rotorRadius = sqrt(rotorArea / M_PI);
        double optimalRotorSpeed = (optimalTipSpeedRatio * windSpeed) / rotorRadius;

        // Calculate torque reference
        double torque = windPower / optimalRotorSpeed;

        // Calculate current reference from torque (simplified)
        double currentMagnitude = torque / 10.0;

        // Generate three-phase currents (simplified)
        double rotorAngle = optimalRotorSpeed * time;
        for (int phase = 0; phase < 3; phase++) {
            double angle = rotorAngle + phase * (2.0 * M_PI / 3.0);
            genSideCurrents[phase] = currentMagnitude * sin(angle);
        }
    }
    else { // Torque control
        // In torque control mode, the generator-side converter
        // controls the torque to follow a reference

        // Simplified model - constant torque based on power
        double rotorSpeed = 10.0; // rad/s (dummy value)
        double torque = electricalPower / rotorSpeed;

        // Calculate current reference from torque (simplified)
        double currentMagnitude = torque / 10.0;

        // Generate three-phase currents (simplified)
        double rotorAngle = rotorSpeed * time;
        for (int phase = 0; phase < 3; phase++) {
            double angle = rotorAngle + phase * (2.0 * M_PI / 3.0);
            genSideCurrents[phase] = currentMagnitude * sin(angle);
        }
    }

    // Grid-side converter control
    if (gridSideControl == 1) { // Voltage control
        // In voltage control mode, the grid-side converter
        // regulates the voltage at its point of connection

        // Simplified model - constant active power flow to grid
        double gridActivePower = electricalPower;

        // Calculate reactive power for voltage control (simplified)
        double measuredVoltage = 1.0; // per unit (dummy value)
        double voltageSetpoint = 1.0; // per unit
        double voltageError = voltageSetpoint - measuredVoltage;
        double reactiveCommand = 100.0 * voltageError * rating; // Simple proportional control

        // Calculate currents
        double gridVoltage = ratedVoltage;
        double currentMagnitude = sqrt(pow(gridActivePower, 2) + pow(reactiveCommand, 2)) / (sqrt(3.0) * gridVoltage);

        // Power factor angle
        double powerFactorAngle = atan2(reactiveCommand, gridActivePower);

        // Generate three-phase currents
        for (int phase = 0; phase < 3; phase++) {
            double angle = 2.0 * M_PI * 60.0 * time + phase * (2.0 * M_PI / 3.0) - powerFactorAngle;
            gridSideCurrents[phase] = currentMagnitude * sin(angle);
        }
    }
    else if (gridSideControl == 2) { // Reactive power control
        // In Q control mode, the grid-side converter
        // follows a reactive power setpoint

        // Simplified model - constant active power flow to grid
        double gridActivePower = electricalPower;

        // Reactive power setpoint (e.g., 0 for unity power factor)
        double reactiveSetpoint = 0.0;

        // Calculate currents
        double gridVoltage = ratedVoltage;
        double currentMagnitude = sqrt(pow(gridActivePower, 2) + pow(reactiveSetpoint, 2)) / (sqrt(3.0) * gridVoltage);

        // Power factor angle
        double powerFactorAngle = atan2(reactiveSetpoint, gridActivePower);

        // Generate three-phase currents
        for (int phase = 0; phase < 3; phase++) {
            double angle = 2.0 * M_PI * 60.0 * time + phase * (2.0 * M_PI / 3.0) - powerFactorAngle;
            gridSideCurrents[phase] = currentMagnitude * sin(angle);
        }
    }
    else { // Power factor control
        // In PF control mode, the grid-side converter
        // maintains a target power factor

        // Simplified model - constant active power flow to grid
        double gridActivePower = electricalPower;

        // Target power factor (e.g., 0.95 leading)
        double targetPF = 0.95;
        double powerFactorAngle = acos(targetPF);

        // Calculate reactive power from PF
        double reactiveSetpoint = gridActivePower * tan(powerFactorAngle);

        // Calculate currents
        double gridVoltage = ratedVoltage;
        double currentMagnitude = sqrt(pow(gridActivePower, 2) + pow(reactiveSetpoint, 2)) / (sqrt(3.0) * gridVoltage);

        // Generate three-phase currents
        for (int phase = 0; phase < 3; phase++) {
            double angle = 2.0 * M_PI * 60.0 * time + phase * (2.0 * M_PI / 3.0) - powerFactorAngle;
            gridSideCurrents[phase] = currentMagnitude * sin(angle);
        }
    }

    // Update DC link voltage
    // In a real implementation, this would involve complex interactions
    // between the generator-side and grid-side converters
    // Simplified model here assumes constant DC voltage
    dcCurrent = electricalPower / dcVoltage;
}

void WIND_CONV::calculateLosses() {
    // Calculate losses in each semiconductor device
    for (auto device : semiconductors) {
        // For simplicity, using the same approach for all devices
        double voltage = dcVoltage / 2.0; // Approximate voltage across device
        double current = std::max(
            *std::max_element(gridSideCurrents.begin(), gridSideCurrents.end()),
            *std::max_element(genSideCurrents.begin(), genSideCurrents.end())
        );

        device->calculateLosses(voltage, current, 1.0e-6);
    }

    // Additional converter losses (simplified)
    double converterLosses = 0.02 * rating * 1.0e6; // Assume 2% losses at rated power
}

void WIND_CONV::prepareDeviceData() {
    // Allocate and initialize device memory for CUDA
    if (!d_stateVars) {
        // 1 DC variable + 6 AC variables (3 grid-side + 3 gen-side currents)
        cudaMalloc(&d_stateVars, 7 * sizeof(double));
    }

    // Copy initial data to device
    updateDeviceData();
}

void WIND_CONV::updateDeviceData() {
    if (d_stateVars) {
        double stateVars[7];

        // DC variable
        stateVars[0] = dcCurrent;

        // AC variables
        for (int i = 0; i < 3; i++) {
            stateVars[1 + i] = gridSideCurrents[i];
            stateVars[4 + i] = genSideCurrents[i];
        }

        cudaMemcpy(d_stateVars, stateVars, 7 * sizeof(double),
            cudaMemcpyHostToDevice);
    }
}

void WIND_CONV::retrieveDeviceData() {
    if (d_stateVars) {
        double stateVars[7];

        cudaMemcpy(stateVars, d_stateVars, 7 * sizeof(double),
            cudaMemcpyDeviceToHost);

        // DC variable
        dcCurrent = stateVars[0];

        // AC variables
        for (int i = 0; i < 3; i++) {
            gridSideCurrents[i] = stateVars[1 + i];
            genSideCurrents[i] = stateVars[4 + i];
        }
    }
}

// Factory function implementation
std::unique_ptr<PowerElectronicConverter> createPowerElectronicConverter(const std::string& converterSpec) {
    std::istringstream iss(converterSpec);
    std::string converterType;
    iss >> converterType;

    if (converterType == "VSC_MMC") {
        std::string name;
        int type, numSubModules, controlType, pwmType;
        double capacitancePerSM, armInductance, armResistance;
        double dcLinkCapacitance, dcVoltage, switchingFreq, carrierPhase;

        iss >> name >> type >> numSubModules >> capacitancePerSM >> armInductance
            >> armResistance >> controlType >> dcLinkCapacitance >> dcVoltage
            >> switchingFreq >> carrierPhase >> pwmType;

        return std::make_unique<VSC_MMC>(name, type, numSubModules,
            capacitancePerSM, armInductance, armResistance,
            controlType, dcLinkCapacitance, dcVoltage,
            switchingFreq, carrierPhase, pwmType);
    }
    else if (converterType == "BESS_VSC") {
        std::string name;
        double capacity, power;
        int batteryType, cellsSeries, cellsParallel, converterType, controlMode;
        double cellVoltage, cellCapacity, responseTime;

        iss >> name >> capacity >> power >> batteryType >> cellsSeries
            >> cellsParallel >> cellVoltage >> cellCapacity >> converterType
            >> responseTime >> controlMode;

        return std::make_unique<BESS_VSC>(name, capacity, power, batteryType,
            cellsSeries, cellsParallel, cellVoltage, cellCapacity,
            converterType, responseTime, controlMode);
    }
    else if (converterType == "SOLAR_INV") {
        std::string name;
        int type, controlType, topology;
        double rating, dcLinkCapacitance, switchingFreq;
        double filterInductance, filterCapacitance, filterDampingR;

        iss >> name >> type >> rating >> controlType >> topology
            >> dcLinkCapacitance >> switchingFreq
            >> filterInductance >> filterCapacitance >> filterDampingR;

        return std::make_unique<SOLAR_INV>(name, type, rating, controlType,
            topology, dcLinkCapacitance, switchingFreq,
            filterInductance, filterCapacitance, filterDampingR);
    }
    else if (converterType == "STATCOM_MMC") {
        std::string name;
        int numSubModules, controlType;
        double capacitancePerSM, armInductance, armResistance;
        double rating, responseTime, switchingFreq;

        iss >> name >> numSubModules >> capacitancePerSM
            >> armInductance >> armResistance >> rating
            >> controlType >> responseTime >> switchingFreq;

        return std::make_unique<STATCOM_MMC>(name, numSubModules, capacitancePerSM,
            armInductance, armResistance, rating,
            controlType, responseTime, switchingFreq);
    }
    else if (converterType == "WIND_CONV") {
        std::string name;
        int type, gridSideControl, genSideControl;
        double rating, dcLinkCapacitance, dcVoltage;
        double gscSwitchingFreq, mscSwitchingFreq;

        iss >> name >> type >> rating >> gridSideControl
            >> genSideControl >> dcLinkCapacitance >> dcVoltage
            >> gscSwitchingFreq >> mscSwitchingFreq;

        return std::make_unique<WIND_CONV>(name, type, rating, gridSideControl,
            genSideControl, dcLinkCapacitance, dcVoltage,
            gscSwitchingFreq, mscSwitchingFreq);
    }

    // Unknown converter type
    std::cerr << "Unknown power electronic converter type: " << converterType << std::endl;
    return nullptr;
}