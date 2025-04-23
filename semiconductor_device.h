#pragma once

#include <string>
#include <vector>
#include <memory>

// Forward declarations
class ThermalModel;

/**
 * @brief Base class for all semiconductor devices
 */
class SemiconductorDevice {
protected:
    std::string name;
    int id;
    ThermalModel* thermalModel;
    double temperature;  // Junction temperature (캜)

public:
    SemiconductorDevice(const std::string& name, int id = -1);
    virtual ~SemiconductorDevice() = default;

    const std::string& getName() const { return name; }
    int getId() const { return id; }
    void setId(int newId) { id = newId; }

    // Set thermal model
    void setThermalModel(ThermalModel* model) { thermalModel = model; }

    // Get thermal model
    ThermalModel* getThermalModel() const { return thermalModel; }

    // Get junction temperature
    double getTemperature() const { return temperature; }

    // Set junction temperature
    void setTemperature(double temp) { temperature = temp; }

    // Virtual methods to be implemented by derived classes
    virtual void calculateLosses(double voltage, double current, double timeStep) = 0;
    virtual double calculateVoltage(double current) const = 0;
    virtual double calculateCurrent(double voltage) const = 0;

    // GPU/CUDA acceleration methods
    virtual void prepareDeviceData() {}
    virtual void updateDeviceData() {}
    virtual void retrieveDeviceData() {}
};

/**
 * @brief IGBT (Insulated Gate Bipolar Transistor) semiconductor device
 */
class IGBT : public SemiconductorDevice {
private:
    double vce;         // Collector-Emitter voltage (V)
    double ic;          // Collector current (A)
    double ron;         // On-state resistance (mOhm)
    double roff;        // Off-state resistance (MOhm)
    double vf;          // Forward voltage drop (V)
    double ton;         // Turn-on time (탎)
    double toff;        // Turn-off time (탎)
    double eon;         // Turn-on energy loss (mJ)
    double eoff;        // Turn-off energy loss (mJ)
    double tjMax;       // Maximum junction temperature (캜)

    // State variables
    bool isOn;          // Current state (on/off)
    double conductionLoss;      // Conduction loss (W)
    double switchingLoss;       // Switching loss (W)
    double totalLoss;           // Total power loss (W)

    // Device data for GPU acceleration
    double* d_stateVars;

public:
    IGBT(const std::string& name, double vce, double ic, double ron, double roff,
        double vf, double ton, double toff, double eon, double eoff, double tjMax);
    ~IGBT() override;

    // Implementation of virtual methods
    void calculateLosses(double voltage, double current, double timeStep) override;
    double calculateVoltage(double current) const override;
    double calculateCurrent(double voltage) const override;

    // GPU acceleration methods
    void prepareDeviceData() override;
    void updateDeviceData() override;
    void retrieveDeviceData() override;

    // IGBT specific methods
    bool getState() const { return isOn; }
    void setState(bool on) { isOn = on; }
    double getConductionLoss() const { return conductionLoss; }
    double getSwitchingLoss() const { return switchingLoss; }
    double getTotalLoss() const { return totalLoss; }
    double getVce() const { return vce; }
    double getIc() const { return ic; }
};

/**
 * @brief Diode semiconductor device
 */
class Diode : public SemiconductorDevice {
private:
    double vf;                  // Forward voltage drop (V)
    double ifRated;             // Rated forward current (A)
    double ron;                 // On-state resistance (mOhm)
    double roff;                // Off-state resistance (MOhm)
    double trr;                 // Reverse recovery time (탎)
    double err;                 // Reverse recovery energy (mJ)
    double qrr;                 // Reverse recovery charge (킗)
    double tjMax;               // Maximum junction temperature (캜)

    // State variables
    bool isOn;                  // Current state (on/off)
    double conductionLoss;      // Conduction loss (W)
    double recoveryLoss;        // Recovery loss (W)
    double totalLoss;           // Total power loss (W)

    // Device data for GPU acceleration
    double* d_stateVars;

public:
    Diode(const std::string& name, double vf, double ifRated, double ron, double roff,
        double trr, double err, double qrr, double tjMax);
    ~Diode() override;

    // Implementation of virtual methods
    void calculateLosses(double voltage, double current, double timeStep) override;
    double calculateVoltage(double current) const override;
    double calculateCurrent(double voltage) const override;

    // GPU acceleration methods
    void prepareDeviceData() override;
    void updateDeviceData() override;
    void retrieveDeviceData() override;

    // Diode specific methods
    bool getState() const { return isOn; }
    void setState(bool on) { isOn = on; }
    double getConductionLoss() const { return conductionLoss; }
    double getRecoveryLoss() const { return recoveryLoss; }
    double getTotalLoss() const { return totalLoss; }
    double getVf() const { return vf; }
    double getIfRated() const { return ifRated; }
};

/**
 * @brief Thyristor semiconductor device
 */
class Thyristor : public SemiconductorDevice {
private:
    double vdrm;                // Repetitive peak off-state voltage (V)
    double itRated;             // Rated on-state current (A)
    double ron;                 // On-state resistance (mOhm)
    double roff;                // Off-state resistance (MOhm)
    double diDtCritical;        // Critical rate of rise of current (A/탎)
    double dvDtCritical;        // Critical rate of rise of voltage (V/탎)
    double turnOnDelay;         // Turn-on delay (탎)
    double switchingEnergy;     // Switching energy (mJ)
    double holdingCurrent;      // Holding current (A)
    double tjMax;               // Maximum junction temperature (캜)

    // State variables
    bool isOn;                  // Current state (on/off)
    bool isTriggered;           // Gate trigger state
    double conductionLoss;      // Conduction loss (W)
    double switchingLoss;       // Switching loss (W)
    double totalLoss;           // Total power loss (W)

    // Device data for GPU acceleration
    double* d_stateVars;

public:
    Thyristor(const std::string& name, double vdrm, double itRated, double ron, double roff,
        double diDtCritical, double dvDtCritical, double turnOnDelay,
        double switchingEnergy, double holdingCurrent, double tjMax);
    ~Thyristor() override;

    // Implementation of virtual methods
    void calculateLosses(double voltage, double current, double timeStep) override;
    double calculateVoltage(double current) const override;
    double calculateCurrent(double voltage) const override;

    // GPU acceleration methods
    void prepareDeviceData() override;
    void updateDeviceData() override;
    void retrieveDeviceData() override;

    // Thyristor specific methods
    bool getState() const { return isOn; }
    void triggerGate(bool trigger) { isTriggered = trigger; }
    bool isGateTriggered() const { return isTriggered; }
    double getConductionLoss() const { return conductionLoss; }
    double getSwitchingLoss() const { return switchingLoss; }
    double getTotalLoss() const { return totalLoss; }
    double getVdrm() const { return vdrm; }
    double getItRated() const { return itRated; }
};

// Factory function to create semiconductor devices
std::unique_ptr<SemiconductorDevice> createSemiconductorDevice(const std::string& deviceSpec);