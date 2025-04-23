#include "thermal_model.h"
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

// IGBTThermalModel implementation
IGBTThermalModel::IGBTThermalModel(const std::string& name, double rthJC, double rthCH,
    double cthJ, double cthC, double ambientTemp)
    : ThermalModel(name, -1), rthJC(rthJC), rthCH(rthCH), cthJ(cthJ), cthC(cthC),
    ambientTemp(ambientTemp), junctionTemp(ambientTemp), caseTemp(ambientTemp),
    d_temperatures(nullptr) {
}

IGBTThermalModel::~IGBTThermalModel() {
    // Free device memory if allocated
    if (d_temperatures) {
        cudaFree(d_temperatures);
        d_temperatures = nullptr;
    }
}

void IGBTThermalModel::initialize(double ambientTemp) {
    this->ambientTemp = ambientTemp;
    junctionTemp = ambientTemp;
    caseTemp = ambientTemp;
}

void IGBTThermalModel::update(double powerLoss, double timeStep) {
    // RC thermal network model:
    // Power loss -> Cj -> Rjc -> Cc -> Rch -> Ambient
    //
    // Where:
    // Cj = junction thermal capacitance
    // Rjc = junction-to-case thermal resistance
    // Cc = case thermal capacitance
    // Rch = case-to-heatsink thermal resistance

    // Calculate heat flow from junction to case
    double heatFlowJC = (junctionTemp - caseTemp) / rthJC;

    // Calculate heat flow from case to ambient
    double heatFlowCH = (caseTemp - ambientTemp) / rthCH;

    // Update junction temperature
    // dT/dt = (Pin - Pout) / C
    double djunctionTemp = (powerLoss - heatFlowJC) / cthJ;
    junctionTemp += djunctionTemp * timeStep;

    // Update case temperature
    // dT/dt = (Pin - Pout) / C
    double dcaseTemp = (heatFlowJC - heatFlowCH) / cthC;
    caseTemp += dcaseTemp * timeStep;

    // Ensure temperatures are physically reasonable
    // (prevent numerical issues)
    junctionTemp = std::max(ambientTemp, junctionTemp);
    caseTemp = std::max(ambientTemp, std::min(junctionTemp, caseTemp));
}

double IGBTThermalModel::getJunctionTemperature() const {
    return junctionTemp;
}

void IGBTThermalModel::prepareDeviceData() {
    // Allocate memory for device variables
    if (!d_temperatures) {
        cudaMalloc(&d_temperatures, 3 * sizeof(double));
    }

    // Copy initial data to device
    updateDeviceData();
}

void IGBTThermalModel::updateDeviceData() {
    if (d_temperatures) {
        double temperatures[3] = {
            junctionTemp,
            caseTemp,
            ambientTemp
        };

        cudaMemcpy(d_temperatures, temperatures, 3 * sizeof(double),
            cudaMemcpyHostToDevice);
    }
}

void IGBTThermalModel::retrieveDeviceData() {
    if (d_temperatures) {
        double temperatures[3];

        cudaMemcpy(temperatures, d_temperatures, 3 * sizeof(double),
            cudaMemcpyDeviceToHost);

        junctionTemp = temperatures[0];
        caseTemp = temperatures[1];
        ambientTemp = temperatures[2];
    }
}

// DiodeThermalModel implementation
DiodeThermalModel::DiodeThermalModel(const std::string& name, double rthJC, double rthCH,
    double cthJ, double cthC, double ambientTemp)
    : ThermalModel(name, -1), rthJC(rthJC), rthCH(rthCH), cthJ(cthJ), cthC(cthC),
    ambientTemp(ambientTemp), junctionTemp(ambientTemp), caseTemp(ambientTemp),
    d_temperatures(nullptr) {
}

DiodeThermalModel::~DiodeThermalModel() {
    // Free device memory if allocated
    if (d_temperatures) {
        cudaFree(d_temperatures);
        d_temperatures = nullptr;
    }
}

void DiodeThermalModel::initialize(double ambientTemp) {
    this->ambientTemp = ambientTemp;
    junctionTemp = ambientTemp;
    caseTemp = ambientTemp;
}

void DiodeThermalModel::update(double powerLoss, double timeStep) {
    // Use the same RC thermal network model as IGBT
    // Power loss -> Cj -> Rjc -> Cc -> Rch -> Ambient

    // Calculate heat flow from junction to case
    double heatFlowJC = (junctionTemp - caseTemp) / rthJC;

    // Calculate heat flow from case to ambient
    double heatFlowCH = (caseTemp - ambientTemp) / rthCH;

    // Update junction temperature
    // dT/dt = (Pin - Pout) / C
    double djunctionTemp = (powerLoss - heatFlowJC) / cthJ;
    junctionTemp += djunctionTemp * timeStep;

    // Update case temperature
    // dT/dt = (Pin - Pout) / C
    double dcaseTemp = (heatFlowJC - heatFlowCH) / cthC;
    caseTemp += dcaseTemp * timeStep;

    // Ensure temperatures are physically reasonable
    junctionTemp = std::max(ambientTemp, junctionTemp);
    caseTemp = std::max(ambientTemp, std::min(junctionTemp, caseTemp));
}

double DiodeThermalModel::getJunctionTemperature() const {
    return junctionTemp;
}

void DiodeThermalModel::prepareDeviceData() {
    // Allocate memory for device variables
    if (!d_temperatures) {
        cudaMalloc(&d_temperatures, 3 * sizeof(double));
    }

    // Copy initial data to device
    updateDeviceData();
}

void DiodeThermalModel::updateDeviceData() {
    if (d_temperatures) {
        double temperatures[3] = {
            junctionTemp,
            caseTemp,
            ambientTemp
        };

        cudaMemcpy(d_temperatures, temperatures, 3 * sizeof(double),
            cudaMemcpyHostToDevice);
    }
}

void DiodeThermalModel::retrieveDeviceData() {
    if (d_temperatures) {
        double temperatures[3];

        cudaMemcpy(temperatures, d_temperatures, 3 * sizeof(double),
            cudaMemcpyDeviceToHost);

        junctionTemp = temperatures[0];
        caseTemp = temperatures[1];
        ambientTemp = temperatures[2];
    }
}

// CoolingSystem implementation
CoolingSystem::CoolingSystem(const std::string& name, int type, double flowRate,
    double ambientTemp, double thermalRes)
    : ThermalModel(name, -1), type(type), flowRate(flowRate), ambientTemp(ambientTemp),
    thermalRes(thermalRes), heatsinkTemp(ambientTemp), d_temperature(nullptr) {
}

CoolingSystem::~CoolingSystem() {
    // Free device memory if allocated
    if (d_temperature) {
        cudaFree(d_temperature);
        d_temperature = nullptr;
    }
}

void CoolingSystem::initialize(double ambientTemp) {
    this->ambientTemp = ambientTemp;
    heatsinkTemp = ambientTemp;
}

void CoolingSystem::update(double powerLoss, double timeStep) {
    // Different cooling system models based on type
    double effectiveThermalResistance = thermalRes;

    // Adjust thermal resistance based on cooling type and flow rate
    if (type == 2) { // Forced air cooling
        // Forced air cooling improves with flow rate (simplified model)
        // Higher flow rate means lower thermal resistance
        effectiveThermalResistance *= 1.0 / (0.5 + 0.5 * flowRate);
    }
    else if (type == 3) { // Liquid cooling
        // Liquid cooling is much more effective than air
        // Flow rate has a stronger effect
        effectiveThermalResistance *= 1.0 / (0.1 + 0.9 * flowRate);
    }

    // Calculate heat flow to ambient
    double heatFlow = (heatsinkTemp - ambientTemp) / effectiveThermalResistance;

    // Update heatsink temperature
    // Simplified thermal model (single capacitance)
    double thermalCapacitance = 5000.0; // J/K (typical for a heatsink)
    double dheatsinkTemp = (powerLoss - heatFlow) / thermalCapacitance;
    heatsinkTemp += dheatsinkTemp * timeStep;

    // Ensure temperature is physically reasonable
    heatsinkTemp = std::max(ambientTemp, heatsinkTemp);
}

double CoolingSystem::getJunctionTemperature() const {
    // This is the heatsink temperature, not junction temperature
    // In a full implementation, this would need to be combined with
    // the device thermal model
    return heatsinkTemp;
}

void CoolingSystem::prepareDeviceData() {
    // Allocate memory for device variables
    if (!d_temperature) {
        cudaMalloc(&d_temperature, 2 * sizeof(double));
    }

    // Copy initial data to device
    updateDeviceData();
}

void CoolingSystem::updateDeviceData() {
    if (d_temperature) {
        double temperatures[2] = {
            heatsinkTemp,
            ambientTemp
        };

        cudaMemcpy(d_temperature, temperatures, 2 * sizeof(double),
            cudaMemcpyHostToDevice);
    }
}

void CoolingSystem::retrieveDeviceData() {
    if (d_temperature) {
        double temperatures[2];

        cudaMemcpy(temperatures, d_temperature, 2 * sizeof(double),
            cudaMemcpyDeviceToHost);

        heatsinkTemp = temperatures[0];
        ambientTemp = temperatures[1];
    }
}

// Factory function implementation
std::unique_ptr<ThermalModel> createThermalModel(const std::string& thermalModelSpec) {
    std::istringstream iss(thermalModelSpec);
    std::string modelType;
    iss >> modelType;

    if (modelType == "THERM_IGBT") {
        std::string name;
        double rthJC, rthCH, cthJ, cthC, ambientTemp;

        iss >> name >> rthJC >> rthCH >> cthJ >> cthC >> ambientTemp;

        return std::make_unique<IGBTThermalModel>(name, rthJC, rthCH, cthJ, cthC, ambientTemp);
    }
    else if (modelType == "THERM_DIODE") {
        std::string name;
        double rthJC, rthCH, cthJ, cthC, ambientTemp;

        iss >> name >> rthJC >> rthCH >> cthJ >> cthC >> ambientTemp;

        return std::make_unique<DiodeThermalModel>(name, rthJC, rthCH, cthJ, cthC, ambientTemp);
    }
    else if (modelType == "COOLING") {
        std::string name;
        int type;
        double flowRate, ambientTemp, thermalRes;

        iss >> name >> type >> flowRate >> ambientTemp >> thermalRes;

        return std::make_unique<CoolingSystem>(name, type, flowRate, ambientTemp, thermalRes);
    }

    // Unknown model type
    std::cerr << "Unknown thermal model type: " << modelType << std::endl;
    return nullptr;
}