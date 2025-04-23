#pragma once

#include <string>
#include <memory>
#include <vector>

/**
 * @brief Base class for all network elements
 */
class NetworkElement {
protected:
    std::string name;
    int id;

public:
    NetworkElement(const std::string& name, int id = -1) : name(name), id(id) {}
    virtual ~NetworkElement() = default;

    const std::string& getName() const { return name; }
    int getId() const { return id; }
    void setId(int newId) { id = newId; }

    // Virtual methods to be implemented by derived classes
    virtual void updateHistoryTerms(double timeStep, double time) = 0;
    virtual void calculateContributions() = 0;

    // Additional methods for GPU-acceleration
    virtual void prepareDeviceData() {}
    virtual void updateDeviceData() {}
    virtual void retrieveDeviceData() {}
};

/**
 * @brief Transmission line element
 */
class TransmissionLine : public NetworkElement {
private:
    double resistance;    // R (ohms)
    double inductance;    // L (henries)
    double capacitance;   // C (farads)
    double length;        // length (km)

    // Node connections
    int fromNode;
    int toNode;

    // History terms
    double historyTermFrom;
    double historyTermTo;

    // Device data pointers (for GPU acceleration)
    double* d_parameters;
    double* d_historyTerms;

public:
    TransmissionLine(const std::string& name, int fromNode, int toNode,
        double resistance, double inductance, double capacitance,
        double length);
    ~TransmissionLine() override;

    int getFromNode() const { return fromNode; }
    int getToNode() const { return toNode; }
    double getResistance() const { return resistance; }
    double getInductance() const { return inductance; }
    double getCapacitance() const { return capacitance; }
    double getLength() const { return length; }

    void updateHistoryTerms(double timeStep, double time) override;
    void calculateContributions() override;

    // GPU acceleration methods
    void prepareDeviceData() override;
    void updateDeviceData() override;
    void retrieveDeviceData() override;
};

/**
 * @brief Transformer element
 */
class Transformer : public NetworkElement {
private:
    double primaryVoltage;     // kV
    double secondaryVoltage;   // kV
    double rating;             // MVA
    double leakageReactance;   // % on transformer base

    // Node connections
    int primaryNode;
    int secondaryNode;

    // History terms
    double historyTerm;

    // Device data pointers (for GPU acceleration)
    double* d_parameters;
    double* d_historyTerms;

public:
    Transformer(const std::string& name, int primaryNode, int secondaryNode,
        double primaryVoltage, double secondaryVoltage,
        double rating, double leakageReactance);
    ~Transformer() override;

    int getPrimaryNode() const { return primaryNode; }
    int getSecondaryNode() const { return secondaryNode; }
    double getPrimaryVoltage() const { return primaryVoltage; }
    double getSecondaryVoltage() const { return secondaryVoltage; }
    double getRating() const { return rating; }
    double getLeakageReactance() const { return leakageReactance; }

    void updateHistoryTerms(double timeStep, double time) override;
    void calculateContributions() override;

    // GPU acceleration methods
    void prepareDeviceData() override;
    void updateDeviceData() override;
    void retrieveDeviceData() override;
};

/**
 * @brief Load element
 */
class Load : public NetworkElement {
private:
    double activePower;    // MW
    double reactivePower;  // MVAR
    bool isConstantImpedance;  // true if constant impedance model, false if constant power

    // Node connection
    int node;

    // History term
    double historyTerm;

    // Device data pointers (for GPU acceleration)
    double* d_parameters;
    double* d_historyTerm;

public:
    Load(const std::string& name, int node, double activePower, double reactivePower,
        bool isConstantImpedance = true);
    ~Load() override;

    int getNode() const { return node; }
    double getActivePower() const { return activePower; }
    double getReactivePower() const { return reactivePower; }
    bool isConstantZ() const { return isConstantImpedance; }

    void updateHistoryTerms(double timeStep, double time) override;
    void calculateContributions() override;

    // GPU acceleration methods
    void prepareDeviceData() override;
    void updateDeviceData() override;
    void retrieveDeviceData() override;
};

/**
 * @brief Voltage source element
 */
class VoltageSource : public NetworkElement {
private:
    double amplitude;   // kV
    double frequency;   // Hz
    double phase;       // degrees

    // Node connection
    int node;

    // Device data pointers (for GPU acceleration)
    double* d_parameters;

public:
    VoltageSource(const std::string& name, int node,
        double amplitude, double frequency, double phase);
    ~VoltageSource() override;

    int getNode() const { return node; }
    double getAmplitude() const { return amplitude; }
    double getFrequency() const { return frequency; }
    double getPhase() const { return phase; }

    void updateHistoryTerms(double timeStep, double time) override;
    void calculateContributions() override;
    double getValue(double time) const;

    // GPU acceleration methods
    void prepareDeviceData() override;
    void updateDeviceData() override;
    void retrieveDeviceData() override;
};

/**
 * @brief Fault element
 */
class Fault : public NetworkElement {
private:
    double startTime;    // s
    double duration;     // s
    double resistance;   // ohms

    // Node connection
    int node;

    // State
    bool active;

    // Device data pointers (for GPU acceleration)
    double* d_parameters;
    bool* d_active;

public:
    Fault(const std::string& name, int node,
        double startTime, double duration, double resistance);
    ~Fault() override;

    int getNode() const { return node; }
    double getStartTime() const { return startTime; }
    double getDuration() const { return duration; }
    double getResistance() const { return resistance; }

    void updateHistoryTerms(double timeStep, double time) override;
    void calculateContributions() override;
    bool isActive(double time) const;

    // GPU acceleration methods
    void prepareDeviceData() override;
    void updateDeviceData() override;
    void retrieveDeviceData() override;
};

// Factory function to create network elements from string specifications
std::unique_ptr<NetworkElement> createNetworkElement(const std::string& elementSpec);