#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

/**
 * @brief Base class for all control systems
 */
class ControlSystem {
protected:
    std::string name;
    int id;

public:
    ControlSystem(const std::string& name, int id = -1) : name(name), id(id) {}
    virtual ~ControlSystem() = default;

    const std::string& getName() const { return name; }
    int getId() const { return id; }
    void setId(int newId) { id = newId; }

    // Virtual methods to be implemented by derived classes
    virtual void initialize() = 0;
    virtual void update(double time, double timeStep) = 0;

    // GPU/CUDA acceleration methods
    virtual void prepareDeviceData() {}
    virtual void updateDeviceData() {}
    virtual void retrieveDeviceData() {}
};

/**
 * @brief Phase-Locked Loop (PLL) controller
 */
class PLL : public ControlSystem {
private:
    double bandwidth;           // PLL bandwidth (Hz)
    double dampingRatio;        // Damping ratio
    double kp;                  // Proportional gain
    double ki;                  // Integral gain
    double settlingTime;        // Settling time (ms)
    double overshoot;           // Overshoot (%)
    int type;                   // 1=SRF, 2=DDSRF, 3=DSOGI

    // PLL state variables
    double theta;               // Estimated phase angle
    double omega;               // Estimated frequency
    double integralTerm;        // Integral term
    double vq;                  // q-axis voltage

    // Device data for GPU acceleration
    double* d_stateVars;

public:
    PLL(const std::string& name, double bandwidth, double dampingRatio,
        double kp, double ki, double settlingTime, double overshoot, int type);
    ~PLL() override;

    // Implementation of virtual methods
    void initialize() override;
    void update(double time, double timeStep) override;

    // GPU acceleration methods
    void prepareDeviceData() override;
    void updateDeviceData() override;
    void retrieveDeviceData() override;

    // PLL specific methods
    double getTheta() const { return theta; }
    double getOmega() const { return omega; }
    double getBandwidth() const { return bandwidth; }
    int getType() const { return type; }

    // Process three-phase voltage inputs
    void processVoltages(const std::vector<double>& voltages);
};

/**
 * @brief Current Controller
 */
class CurrentController : public ControlSystem {
private:
    int type;                   // 1=PI, 2=PR, 3=Deadbeat, 4=MPC
    double bandwidth;           // Bandwidth (Hz)
    double dampingRatio;        // Damping ratio
    double kp;                  // Proportional gain
    double ki;                  // Integral gain
    double samplingFrequency;   // Sampling frequency (kHz)
    int predictionHorizon;      // Prediction horizon (MPC only)

    // Controller state variables
    std::vector<double> references;      // Current references (d, q)
    std::vector<double> feedbacks;       // Current feedbacks (d, q)
    std::vector<double> outputs;         // Controller outputs (d, q)
    std::vector<double> integralTerms;   // Integral terms (d, q)

    // Device data for GPU acceleration
    double* d_stateVars;

public:
    CurrentController(const std::string& name, int type, double bandwidth,
        double dampingRatio, double kp, double ki,
        double samplingFrequency, int predictionHorizon);
    ~CurrentController() override;

    // Implementation of virtual methods
    void initialize() override;
    void update(double time, double timeStep) override;

    // GPU acceleration methods
    void prepareDeviceData() override;
    void updateDeviceData() override;
    void retrieveDeviceData() override;

    // Current controller specific methods
    const std::vector<double>& getOutputs() const { return outputs; }
    void setReferences(const std::vector<double>& refs);
    void setFeedbacks(const std::vector<double>& fbs);
    int getType() const { return type; }
};

/**
 * @brief Voltage Controller
 */
class VoltageController : public ControlSystem {
private:
    int type;                   // 1=PI, 2=Droop, 3=Resonant
    double bandwidth;           // Bandwidth (Hz)
    double dampingRatio;        // Damping ratio
    double kp;                  // Proportional gain
    double ki;                  // Integral gain
    double droopCoefficient;    // Droop coefficient (%)
    bool antiWindup;            // Anti-windup (1=On, 0=Off)

    // Controller state variables
    std::vector<double> references;      // Voltage references (d, q)
    std::vector<double> feedbacks;       // Voltage feedbacks (d, q)
    std::vector<double> outputs;         // Controller outputs (d, q)
    std::vector<double> integralTerms;   // Integral terms (d, q)

    // Device data for GPU acceleration
    double* d_stateVars;

public:
    VoltageController(const std::string& name, int type, double bandwidth,
        double dampingRatio, double kp, double ki,
        double droopCoefficient, bool antiWindup);
    ~VoltageController() override;

    // Implementation of virtual methods
    void initialize() override;
    void update(double time, double timeStep) override;

    // GPU acceleration methods
    void prepareDeviceData() override;
    void updateDeviceData() override;
    void retrieveDeviceData() override;

    // Voltage controller specific methods
    const std::vector<double>& getOutputs() const { return outputs; }
    void setReferences(const std::vector<double>& refs);
    void setFeedbacks(const std::vector<double>& fbs);
    int getType() const { return type; }
    double getDroopCoefficient() const { return droopCoefficient; }
};

/**
 * @brief Power Controller
 */
class PowerController : public ControlSystem {
private:
    int type;                   // 1=PI, 2=Droop
    double bandwidth;           // Bandwidth (Hz)
    double dampingRatio;        // Damping ratio
    double kp;                  // Proportional gain
    double ki;                  // Integral gain
    double pDroop;              // Active power droop (%)
    double qDroop;              // Reactive power droop (%)

    // Controller state variables
    double pRef;                // Active power reference
    double qRef;                // Reactive power reference
    double pFeedback;           // Active power feedback
    double qFeedback;           // Reactive power feedback
    std::vector<double> outputs;         // Controller outputs (d, q)
    std::vector<double> integralTerms;   // Integral terms (d, q)

    // Device data for GPU acceleration
    double* d_stateVars;

public:
    PowerController(const std::string& name, int type, double bandwidth,
        double dampingRatio, double kp, double ki,
        double pDroop, double qDroop);
    ~PowerController() override;

    // Implementation of virtual methods
    void initialize() override;
    void update(double time, double timeStep) override;

    // GPU acceleration methods
    void prepareDeviceData() override;
    void updateDeviceData() override;
    void retrieveDeviceData() override;

    // Power controller specific methods
    const std::vector<double>& getOutputs() const { return outputs; }
    void setReferences(double pRef, double qRef);
    void setFeedbacks(double pFeedback, double qFeedback);
    int getType() const { return type; }
    double getPDroop() const { return pDroop; }
    double getQDroop() const { return qDroop; }
};

/**
 * @brief DC Voltage Controller
 */
class DCVoltageController : public ControlSystem {
private:
    int type;                   // 1=PI, 2=Droop
    double bandwidth;           // Bandwidth (Hz)
    double dampingRatio;        // Damping ratio
    double kp;                  // Proportional gain
    double ki;                  // Integral gain
    double droop;               // Droop coefficient (%)

    // Controller state variables
    double reference;           // DC voltage reference
    double feedback;            // DC voltage feedback
    double output;              // Controller output
    double integralTerm;        // Integral term

    // Device data for GPU acceleration
    double* d_stateVars;

public:
    DCVoltageController(const std::string& name, int type, double bandwidth,
        double dampingRatio, double kp, double ki, double droop);
    ~DCVoltageController() override;

    // Implementation of virtual methods
    void initialize() override;
    void update(double time, double timeStep) override;

    // GPU acceleration methods
    void prepareDeviceData() override;
    void updateDeviceData() override;
    void retrieveDeviceData() override;

    // DC voltage controller specific methods
    double getOutput() const { return output; }
    void setReference(double ref);
    void setFeedback(double fb);
    int getType() const { return type; }
    double getDroop() const { return droop; }
};

/**
 * @brief Grid-Forming Controller
 */
class GridFormingController : public ControlSystem {
private:
    int type;                   // 1=VSM, 2=droop, 3=dVOC
    double virtualInertia;      // Virtual inertia (s)
    double droop;               // Droop coefficient (%)
    double damping;             // Damping coefficient (pu)
    double synchronizingCoeff;  // Synchronizing coefficient (pu)
    double lowPassFilterFreq;   // Low-pass filter frequency (Hz)

    // Controller state variables
    double theta;               // Phase angle
    double omega;               // Angular frequency
    double p;                   // Active power
    double q;                   // Reactive power
    std::vector<double> voltageRefs;     // Voltage references (d, q)

    // Device data for GPU acceleration
    double* d_stateVars;

public:
    GridFormingController(const std::string& name, int type, double virtualInertia,
        double droop, double damping, double synchronizingCoeff,
        double lowPassFilterFreq);
    ~GridFormingController() override;

    // Implementation of virtual methods
    void initialize() override;
    void update(double time, double timeStep) override;

    // GPU acceleration methods
    void prepareDeviceData() override;
    void updateDeviceData() override;
    void retrieveDeviceData() override;

    // Grid-forming controller specific methods
    double getTheta() const { return theta; }
    double getOmega() const { return omega; }
    const std::vector<double>& getVoltageRefs() const { return voltageRefs; }
    int getType() const { return type; }
    double getVirtualInertia() const { return virtualInertia; }
    double getDroop() const { return droop; }

    // Process power inputs
    void processPowerInputs(double pInput, double qInput);
};

/**
 * @brief Capacitor Voltage Balancing Algorithm
 */
class CapacitorBalancing : public ControlSystem {
private:
    int type;                   // 1=Sorting, 2=Tolerance Band, 3=Predictive
    double toleranceBand;       // Tolerance band (V)
    double switchingFreqLimit;  // Switching frequency limit (Hz)
    double updateRate;          // Update rate (Hz)

    // Controller state variables
    std::vector<double> capacitorVoltages;   // Capacitor voltages
    std::vector<int> switchingStates;        // Switching states
    std::vector<double> switchingTimes;      // Last switching times

    // Device data for GPU acceleration
    double* d_capVoltages;
    int* d_switchStates;
    double* d_switchTimes;

    // Private methods for different balancing strategies
    void sortingBalancing(double time);
    void toleranceBandBalancing(double time);
    void predictiveBalancing(double time, double timeStep);

public:
    CapacitorBalancing(const std::string& name, int type, double toleranceBand,
        double switchingFreqLimit, double updateRate);
    ~CapacitorBalancing() override;

    // Implementation of virtual methods
    void initialize() override;
    void update(double time, double timeStep) override;

    // GPU acceleration methods
    void prepareDeviceData() override;
    void updateDeviceData() override;
    void retrieveDeviceData() override;

    // Capacitor balancing specific methods
    const std::vector<int>& getSwitchingStates() const { return switchingStates; }
    void setCapacitorVoltages(const std::vector<double>& voltages);
    int getType() const { return type; }
};

// Factory function to create control systems
std::unique_ptr<ControlSystem> createControlSystem(const std::string& controlSystemSpec);