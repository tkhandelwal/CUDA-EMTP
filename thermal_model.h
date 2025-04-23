#pragma once

#include <string>
#include <vector>
#include <memory>

/**
 * @brief Base class for all thermal models
 */
class ThermalModel {
protected:
    std::string name;
    int id;

public:
    ThermalModel(const std::string& name, int id = -1) : name(name), id(id) {}
    virtual ~ThermalModel() = default;

    const std::string& getName() const { return name; }
    int getId() const { return id; }
    void setId(int newId) { id = newId; }

    // Virtual methods to be implemented by derived classes
    virtual void initialize(double ambientTemp) = 0;
    virtual void update(double powerLoss, double timeStep) = 0;
    virtual double getJunctionTemperature() const = 0;

    // GPU/CUDA acceleration methods
    virtual void prepareDeviceData() {}
    virtual void updateDeviceData() {}
    virtual void retrieveDeviceData() {}
};

/**
 * @brief IGBT Thermal Model
 */
class IGBTThermalModel : public ThermalModel {
private:
    double rthJC;           // Thermal resistance junction to case (K/W)
    double rthCH;           // Thermal resistance case to heatsink (K/W)
    double cthJ;            // Thermal capacitance of junction (J/K)
    double cthC;            // Thermal capacitance of case (J/K)
    double ambientTemp;     // Ambient temperature (°C)

    // State variables
    double junctionTemp;    // Junction temperature (°C)
    double caseTemp;        // Case temperature (°C)

    // Device data for GPU acceleration
    double* d_temperatures;

public:
    IGBTThermalModel(const std::string& name, double rthJC, double rthCH,
        double cthJ, double cthC, double ambientTemp);
    ~IGBTThermalModel() override;

    // Implementation of virtual methods
    void initialize(double ambientTemp) override;
    void update(double powerLoss, double timeStep) override;
    double getJunctionTemperature() const override;

    // GPU acceleration methods
    void prepareDeviceData() override;
    void updateDeviceData() override;
    void retrieveDeviceData() override;

    // IGBT thermal model specific methods
    double getCaseTemperature() const { return caseTemp; }
    double getRthJC() const { return rthJC; }
    double getRthCH() const { return rthCH; }
};

/**
 * @brief Diode Thermal Model
 */
class DiodeThermalModel : public ThermalModel {
private:
    double rthJC;           // Thermal resistance junction to case (K/W)
    double rthCH;           // Thermal resistance case to heatsink (K/W)
    double cthJ;            // Thermal capacitance of junction (J/K)
    double cthC;            // Thermal capacitance of case (J/K)
    double ambientTemp;     // Ambient temperature (°C)

    // State variables
    double junctionTemp;    // Junction temperature (°C)
    double caseTemp;        // Case temperature (°C)

    // Device data for GPU acceleration
    double* d_temperatures;

public:
    DiodeThermalModel(const std::string& name, double rthJC, double rthCH,
        double cthJ, double cthC, double ambientTemp);
    ~DiodeThermalModel() override;

    // Implementation of virtual methods
    void initialize(double ambientTemp) override;
    void update(double powerLoss, double timeStep) override;
    double getJunctionTemperature() const override;

    // GPU acceleration methods
    void prepareDeviceData() override;
    void updateDeviceData() override;
    void retrieveDeviceData() override;

    // Diode thermal model specific methods
    double getCaseTemperature() const { return caseTemp; }
    double getRthJC() const { return rthJC; }
    double getRthCH() const { return rthCH; }
};

/**
 * @brief Cooling System Model
 */
class CoolingSystem : public ThermalModel {
private:
    int type;               // 1=Natural, 2=Forced Air, 3=Liquid
    double flowRate;        // Flow rate (units depend on cooling type)
    double ambientTemp;     // Ambient temperature (°C)
    double thermalRes;      // Thermal resistance (K/W)

    // State variables
    double heatsinkTemp;    // Heatsink temperature (°C)

    // Device data for GPU acceleration
    double* d_temperature;

public:
    CoolingSystem(const std::string& name, int type, double flowRate,
        double ambientTemp, double thermalRes);
    ~CoolingSystem() override;

    // Implementation of virtual methods
    void initialize(double ambientTemp) override;
    void update(double powerLoss, double timeStep) override;
    double getJunctionTemperature() const override;

    // GPU acceleration methods
    void prepareDeviceData() override;
    void updateDeviceData() override;
    void retrieveDeviceData() override;

    // Cooling system specific methods
    double getHeatsinkTemperature() const { return heatsinkTemp; }
    int getType() const { return type; }
    double getFlowRate() const { return flowRate; }
    double getThermalResistance() const { return thermalRes; }
};

// Factory function to create thermal models
std::unique_ptr<ThermalModel> createThermalModel(const std::string& thermalModelSpec);