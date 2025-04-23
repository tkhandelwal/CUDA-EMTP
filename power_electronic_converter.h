#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "network_element.h"

// Forward declarations
class SemiconductorDevice;
class ControlSystem;
class ThermalModel;

/**
 * @brief Base class for all power electronic converters
 */
class PowerElectronicConverter {
protected:
    std::string name;
    int id;
    std::vector<SemiconductorDevice*> semiconductors;
    std::vector<ControlSystem*> controllers;
    std::vector<ThermalModel*> thermalModels;

    // Common converter parameters
    double ratedPower;
    double ratedVoltage;
    double switchingFrequency;

public:
    PowerElectronicConverter(const std::string& name, int id = -1);
    virtual ~PowerElectronicConverter() = default;

    const std::string& getName() const { return name; }
    int getId() const { return id; }
    void setId(int newId) { id = newId; }

    // Add semiconductor device to converter
    void addSemiconductor(SemiconductorDevice* device);

    // Add controller to converter
    void addController(ControlSystem* controller);

    // Add thermal model to converter
    void addThermalModel(ThermalModel* model);

    // Get rated power
    double getRatedPower() const { return ratedPower; }

    // Get rated voltage
    double getRatedVoltage() const { return ratedVoltage; }

    // Get switching frequency
    double getSwitchingFrequency() const { return switchingFrequency; }

    // Virtual methods to be implemented by derived classes
    virtual void initialize() = 0;
    virtual void update(double time, double timeStep) = 0;
    virtual void calculateLosses() = 0;

    // GPU/CUDA acceleration methods
    virtual void prepareDeviceData() {}
    virtual void updateDeviceData() {}
    virtual void retrieveDeviceData() {}
};

/**
 * @brief Voltage Source Converter with Modular Multilevel Converter topology
 */
class VSC_MMC : public PowerElectronicConverter {
private:
    int type;                   // 1=Half-Bridge, 2=Full-Bridge
    int numSubModules;          // Number of submodules per arm
    double capacitancePerSM;    // Capacitance per submodule (F)
    double armInductance;       // Arm inductance (H)
    double armResistance;       // Arm resistance (ohm)
    int controlType;            // Control type
    double dcLinkCapacitance;   // DC link capacitance (F)
    double dcVoltage;           // DC voltage (kV)
    double carrierPhase;        // Carrier phase shift (degrees)
    int pwmType;                // 1=Carrier Based, 2=SVM

    // Operational state variables
    std::vector<double> submoduleVoltages;    // Capacitor voltages for each submodule
    std::vector<double> armCurrents;          // Current in each arm
    std::vector<int> switchingStates;         // Switching state of each submodule

    // Device data for GPU acceleration
    double* d_submoduleVoltages;
    double* d_armCurrents;
    int* d_switchingStates;

public:
    VSC_MMC(const std::string& name, int type, int numSubModules,
        double capacitancePerSM, double armInductance, double armResistance,
        int controlType, double dcLinkCapacitance, double dcVoltage,
        double switchingFreq, double carrierPhase, int pwmType);
    ~VSC_MMC() override;

    // Implementation of virtual methods
    void initialize() override;
    void update(double time, double timeStep) override;
    void calculateLosses() override;

    // GPU acceleration methods
    void prepareDeviceData() override;
    void updateDeviceData() override;
    void retrieveDeviceData() override;

    // MMC specific methods
    void balanceCapacitorVoltages();
    void updateCirculatingCurrentsControl();
    int getNumSubModules() const { return numSubModules; }
    int getType() const { return type; }
    double getDcVoltage() const { return dcVoltage; }
};

/**
 * @brief Battery Energy Storage System with Voltage Source Converter
 */
class BESS_VSC : public PowerElectronicConverter {
private:
    double capacity;            // Battery capacity (MWh)
    double power;               // Power rating (MW)
    int batteryType;            // 1=Li-Ion, 2=Flow
    int cellsSeries;            // Number of cells in series
    int cellsParallel;          // Number of cells in parallel
    double cellVoltage;         // Cell voltage (V)
    double cellCapacity;        // Cell capacity (Ah)
    int converterType;          // 1=2L, 2=3L, 3=MMC
    double responseTime;        // Response time (ms)
    int controlMode;            // 1=PQ, 2=Vf

    // Battery state variables
    double stateOfCharge;       // Current SOC (%)
    double batteryVoltage;      // Battery terminal voltage (V)
    double batteryCurrent;      // Battery current (A)
    double batteryTemperature;  // Battery temperature (C)

    // Device data for GPU acceleration
    double* d_batteryStateVars;

public:
    BESS_VSC(const std::string& name, double capacity, double power, int batteryType,
        int cellsSeries, int cellsParallel, double cellVoltage, double cellCapacity,
        int converterType, double responseTime, int controlMode);
    ~BESS_VSC() override;

    // Implementation of virtual methods
    void initialize() override;
    void update(double time, double timeStep) override;
    void calculateLosses() override;

    // GPU acceleration methods
    void prepareDeviceData() override;
    void updateDeviceData() override;
    void retrieveDeviceData() override;

    // BESS specific methods
    double getStateOfCharge() const { return stateOfCharge; }
    void setStateOfCharge(double soc) { stateOfCharge = soc; }
    int getControlMode() const { return controlMode; }
    double getCapacity() const { return capacity; }
    double getPower() const { return power; }
    double calculateBatteryVoltage(double soc);
};

/**
 * @brief Solar Inverter for PV applications
 */
class SOLAR_INV : public PowerElectronicConverter {
private:
    int type;                   // 1=String, 2=Central
    double rating;              // Inverter rating (MVA)
    int controlType;            // 1=Grid-following, 2=Grid-forming
    int topology;               // 1=2L, 2=3L, 3=MMC
    double dcLinkCapacitance;   // DC link capacitance (F)
    double filterInductance;    // Filter inductance (H)
    double filterCapacitance;   // Filter capacitance (F)
    double filterDampingR;      // Filter damping resistance (ohm)

    // Operational state variables
    double dcVoltage;           // DC link voltage (V)
    double dcCurrent;           // DC current (A)
    std::vector<double> acCurrents;   // AC currents (A)
    std::vector<double> acVoltages;   // AC voltages (V)

    // Device data for GPU acceleration
    double* d_stateVars;

public:
    SOLAR_INV(const std::string& name, int type, double rating, int controlType,
        int topology, double dcLinkCapacitance, double switchingFreq,
        double filterInductance, double filterCapacitance, double filterDampingR);
    ~SOLAR_INV() override;

    // Implementation of virtual methods
    void initialize() override;
    void update(double time, double timeStep) override;
    void calculateLosses() override;

    // GPU acceleration methods
    void prepareDeviceData() override;
    void updateDeviceData() override;
    void retrieveDeviceData() override;

    // Solar inverter specific methods
    int getControlType() const { return controlType; }
    double getRating() const { return rating; }
    int getTopology() const { return topology; }
};

/**
 * @brief STATCOM with MMC topology
 */
class STATCOM_MMC : public PowerElectronicConverter {
private:
    int numSubModules;          // Number of submodules per arm
    double capacitancePerSM;    // Capacitance per submodule (F)
    double armInductance;       // Arm inductance (H)
    double armResistance;       // Arm resistance (ohm)
    double rating;              // STATCOM rating (MVA)
    int controlType;            // 1=V, 2=Q, 3=PF
    double responseTime;        // Response time (ms)

    // Operational state variables
    std::vector<double> submoduleVoltages;    // Capacitor voltages for each submodule
    std::vector<double> armCurrents;          // Current in each arm
    std::vector<int> switchingStates;         // Switching state of each submodule

    // Device data for GPU acceleration
    double* d_submoduleVoltages;
    double* d_armCurrents;
    int* d_switchingStates;

public:
    STATCOM_MMC(const std::string& name, int numSubModules, double capacitancePerSM,
        double armInductance, double armResistance, double rating,
        int controlType, double responseTime, double switchingFreq);
    ~STATCOM_MMC() override;

    // Implementation of virtual methods
    void initialize() override;
    void update(double time, double timeStep) override;
    void calculateLosses() override;
    void balanceCapacitorVoltages();

    // GPU acceleration methods
    void prepareDeviceData() override;
    void updateDeviceData() override;
    void retrieveDeviceData() override;

    // STATCOM specific methods
    double getRating() const { return rating; }
    int getControlType() const { return controlType; }
    double getResponseTime() const { return responseTime; }
};

/**
 * @brief Wind Turbine Converter (Back-to-Back)
 */
class WIND_CONV : public PowerElectronicConverter {
private:
    int type;                   // 1=DFIG, 2=Full-Converter
    double rating;              // Converter rating (MVA)
    int gridSideControl;        // 1=V, 2=Q, 3=PF
    int genSideControl;         // 1=Speed, 2=Torque
    double dcLinkCapacitance;   // DC link capacitance (F)
    double dcVoltage;           // DC voltage (V)
    double gscSwitchingFreq;    // Grid-side converter switching frequency (kHz)
    double mscSwitchingFreq;    // Machine-side converter switching frequency (kHz)

    // Operational state variables
    double dcCurrent;           // DC current (A)
    std::vector<double> gridSideCurrents;   // Grid-side currents (A)
    std::vector<double> genSideCurrents;    // Generator-side currents (A)

    // Device data for GPU acceleration
    double* d_stateVars;

public:
    WIND_CONV(const std::string& name, int type, double rating, int gridSideControl,
        int genSideControl, double dcLinkCapacitance, double dcVoltage,
        double gscSwitchingFreq, double mscSwitchingFreq);
    ~WIND_CONV() override;

    // Implementation of virtual methods
    void initialize() override;
    void update(double time, double timeStep) override;
    void calculateLosses() override;

    // GPU acceleration methods
    void prepareDeviceData() override;
    void updateDeviceData() override;
    void retrieveDeviceData() override;

    // Wind converter specific methods
    int getType() const { return type; }
    double getRating() const { return rating; }
    int getGridSideControl() const { return gridSideControl; }
    int getGenSideControl() const { return genSideControl; }
};

// Factory function to create power electronic converters
std::unique_ptr<PowerElectronicConverter> createPowerElectronicConverter(const std::string& converterSpec);