#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include "simulation_results.h"

// Forward declaration of SQLite structure
struct sqlite3;
struct sqlite3_stmt;

/**
 * @brief Database connector for storing simulation results
 */
class DatabaseConnector {
private:
    sqlite3* db;
    bool inMemory;
    std::string dbFileName;
    int currentBufferSize;
    int maxBufferSize;
    std::mutex dbMutex;

    // Prepared statements
    sqlite3_stmt* insertVoltageStmt;
    sqlite3_stmt* insertCurrentStmt;
    sqlite3_stmt* beginTransactionStmt;
    sqlite3_stmt* endTransactionStmt;

    // Private helper methods
    bool createTables();
    bool prepareStatements();

public:
    /**
     * @brief Constructor for DatabaseConnector
     * @param useInMemory If true, use in-memory database
     * @param fileName Database file name
     * @param bufferSize Number of operations to buffer before commit
     */
    DatabaseConnector(bool useInMemory = true, const std::string& fileName = "simulation.db", int bufferSize = 100);

    /**
     * @brief Destructor
     */
    ~DatabaseConnector();

    /**
     * @brief Connect to database
     * @return True if successfully connected
     */
    bool connect();

    /**
     * @brief Disconnect from database
     */
    void disconnect();

    /**
     * @brief Write data for a single time step
     * @param results Simulation results
     * @param timeStep Time step index
     * @return True if successfully written
     */
    bool writeTimeStepData(const SimulationResults& results, int timeStep);

    /**
     * @brief Write all simulation data in bulk
     * @param results Simulation results
     * @return True if successfully written
     */
    bool writeBulkData(const SimulationResults& results);

    /**
     * @brief Get recent simulation data
     * @param nodeNames List of node names to get data for
     * @param branchNames List of branch names to get data for
     * @param count Number of most recent time steps to get
     * @param nodeVoltages Output map of node voltages
     * @param branchCurrents Output map of branch currents
     * @param timePoints Output vector of time points
     * @return True if successfully retrieved
     */
    bool getRecentData(const std::vector<std::string>& nodeNames,
        const std::vector<std::string>& branchNames,
        int count,
        std::unordered_map<std::string, std::vector<double>>& nodeVoltages,
        std::unordered_map<std::string, std::vector<double>>& branchCurrents,
        std::vector<double>& timePoints);
};