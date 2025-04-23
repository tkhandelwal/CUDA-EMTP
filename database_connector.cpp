#include "database_connector.h"
#include <iostream>
#include <sstream>
#include "/Users/Tanuj.Khandelwal/source/repos/eMTP-CUDA-V2/packages/sqlite-amalgamation-3490100/sqlite3.h"
#include "simulation_results.h"

DatabaseConnector::DatabaseConnector(bool useInMemory, const std::string& fileName, int bufferSize)
    : db(nullptr), inMemory(useInMemory), dbFileName(fileName),
    currentBufferSize(0), maxBufferSize(bufferSize),
    insertVoltageStmt(nullptr), insertCurrentStmt(nullptr),
    beginTransactionStmt(nullptr), endTransactionStmt(nullptr) {
}

DatabaseConnector::~DatabaseConnector() {
    disconnect();
}

bool DatabaseConnector::connect() {
    std::lock_guard<std::mutex> lock(dbMutex);

    if (db != nullptr) {
        // Already connected
        return true;
    }

    // Connect to database
    int rc;
    if (inMemory) {
        // Use shared in-memory database for concurrent access
        rc = sqlite3_open("file:memdb1?mode=memory&cache=shared", &db);
    }
    else {
        rc = sqlite3_open(dbFileName.c_str(), &db);
    }

    if (rc != SQLITE_OK) {
        std::cerr << "Failed to open database: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        db = nullptr;
        return false;
    }

    // Enable WAL mode for better performance
    char* errMsg = nullptr;
    rc = sqlite3_exec(db, "PRAGMA journal_mode = WAL;", nullptr, nullptr, &errMsg);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to enable WAL mode: " << errMsg << std::endl;
        sqlite3_free(errMsg);
    }

    // Set synchronous mode to NORMAL for better performance
    rc = sqlite3_exec(db, "PRAGMA synchronous = NORMAL;", nullptr, nullptr, &errMsg);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to set synchronous mode: " << errMsg << std::endl;
        sqlite3_free(errMsg);
    }

    // Create tables
    if (!createTables()) {
        sqlite3_close(db);
        db = nullptr;
        return false;
    }

    // Prepare statements
    if (!prepareStatements()) {
        sqlite3_close(db);
        db = nullptr;
        return false;
    }

    std::cout << "Connected to SQLite database" << (inMemory ? " (in-memory)" : (" (" + dbFileName + ")")) << std::endl;
    return true;
}

void DatabaseConnector::disconnect() {
    std::lock_guard<std::mutex> lock(dbMutex);

    if (db == nullptr) {
        return;
    }

    // Finalize prepared statements
    if (insertVoltageStmt != nullptr) {
        sqlite3_finalize(insertVoltageStmt);
        insertVoltageStmt = nullptr;
    }

    if (insertCurrentStmt != nullptr) {
        sqlite3_finalize(insertCurrentStmt);
        insertCurrentStmt = nullptr;
    }

    if (beginTransactionStmt != nullptr) {
        sqlite3_finalize(beginTransactionStmt);
        beginTransactionStmt = nullptr;
    }

    if (endTransactionStmt != nullptr) {
        sqlite3_finalize(endTransactionStmt);
        endTransactionStmt = nullptr;
    }

    // Close database
    sqlite3_close(db);
    db = nullptr;

    std::cout << "Disconnected from SQLite database" << std::endl;
}

bool DatabaseConnector::createTables() {
    char* errMsg = nullptr;

    // Create time_points table
    const char* createTimePointsSQL =
        "CREATE TABLE IF NOT EXISTS time_points ("
        "  time_step INTEGER PRIMARY KEY,"
        "  simulation_time REAL NOT NULL"
        ");";

    int rc = sqlite3_exec(db, createTimePointsSQL, nullptr, nullptr, &errMsg);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to create time_points table: " << errMsg << std::endl;
        sqlite3_free(errMsg);
        return false;
    }

    // Create node_voltages table
    const char* createNodeVoltagesSQL =
        "CREATE TABLE IF NOT EXISTS node_voltages ("
        "  time_step INTEGER,"
        "  node_name TEXT,"
        "  voltage REAL NOT NULL,"
        "  PRIMARY KEY (time_step, node_name),"
        "  FOREIGN KEY (time_step) REFERENCES time_points (time_step)"
        ");";

    rc = sqlite3_exec(db, createNodeVoltagesSQL, nullptr, nullptr, &errMsg);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to create node_voltages table: " << errMsg << std::endl;
        sqlite3_free(errMsg);
        return false;
    }

    // Create branch_currents table
    const char* createBranchCurrentsSQL =
        "CREATE TABLE IF NOT EXISTS branch_currents ("
        "  time_step INTEGER,"
        "  branch_name TEXT,"
        "  current REAL NOT NULL,"
        "  PRIMARY KEY (time_step, branch_name),"
        "  FOREIGN KEY (time_step) REFERENCES time_points (time_step)"
        ");";

    rc = sqlite3_exec(db, createBranchCurrentsSQL, nullptr, nullptr, &errMsg);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to create branch_currents table: " << errMsg << std::endl;
        sqlite3_free(errMsg);
        return false;
    }

    // Create indices for faster queries
    const char* createIndicesSQL =
        "CREATE INDEX IF NOT EXISTS idx_node_voltages_time_step ON node_voltages (time_step);"
        "CREATE INDEX IF NOT EXISTS idx_node_voltages_node_name ON node_voltages (node_name);"
        "CREATE INDEX IF NOT EXISTS idx_branch_currents_time_step ON branch_currents (time_step);"
        "CREATE INDEX IF NOT EXISTS idx_branch_currents_branch_name ON branch_currents (branch_name);";

    rc = sqlite3_exec(db, createIndicesSQL, nullptr, nullptr, &errMsg);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to create indices: " << errMsg << std::endl;
        sqlite3_free(errMsg);
        return false;
    }

    return true;
}

bool DatabaseConnector::prepareStatements() {
    int rc;

    // Prepare begin transaction statement
    const char* beginTransactionSQL = "BEGIN TRANSACTION;";
    rc = sqlite3_prepare_v2(db, beginTransactionSQL, -1, &beginTransactionStmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare begin transaction statement: " << sqlite3_errmsg(db) << std::endl;
        return false;
    }

    // Prepare end transaction statement
    const char* endTransactionSQL = "COMMIT TRANSACTION;";
    rc = sqlite3_prepare_v2(db, endTransactionSQL, -1, &endTransactionStmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare end transaction statement: " << sqlite3_errmsg(db) << std::endl;
        return false;
    }

    // Prepare insert voltage statement
    const char* insertVoltageSQL =
        "INSERT OR REPLACE INTO node_voltages (time_step, node_name, voltage) VALUES (?, ?, ?);";
    rc = sqlite3_prepare_v2(db, insertVoltageSQL, -1, &insertVoltageStmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare insert voltage statement: " << sqlite3_errmsg(db) << std::endl;
        return false;
    }

    // Prepare insert current statement
    const char* insertCurrentSQL =
        "INSERT OR REPLACE INTO branch_currents (time_step, branch_name, current) VALUES (?, ?, ?);";
    rc = sqlite3_prepare_v2(db, insertCurrentSQL, -1, &insertCurrentStmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare insert current statement: " << sqlite3_errmsg(db) << std::endl;
        return false;
    }

    return true;
}

bool DatabaseConnector::writeTimeStepData(const SimulationResults& results, int timeStep) {
    std::lock_guard<std::mutex> lock(dbMutex);

    if (db == nullptr) {
        std::cerr << "Not connected to database" << std::endl;
        return false;
    }

    // Set locale to ensure period as decimal separator
    std::locale::global(std::locale("C"));

    // Add bounds checking
    if (timeStep < 0 || timeStep >= static_cast<int>(results.timePoints.size())) {
        std::cerr << "Invalid time step: " << timeStep << " (size: " << results.timePoints.size() << ")" << std::endl;
        return false;
    }

    int rc;

    // Start transaction if this is the first item in the buffer
    if (currentBufferSize == 0) {
        rc = sqlite3_step(beginTransactionStmt);
        if (rc != SQLITE_DONE) {
            std::cerr << "Failed to start transaction: " << sqlite3_errmsg(db) << std::endl;
            return false;
        }
        sqlite3_reset(beginTransactionStmt);
    }

    // Prepare statement for time point
    sqlite3_stmt* timeStmt = nullptr;
    const char* timeInsertSQL = "INSERT OR REPLACE INTO time_points (time_step, simulation_time) VALUES (?, ?);";
    rc = sqlite3_prepare_v2(db, timeInsertSQL, -1, &timeStmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare time insertion: " << sqlite3_errmsg(db) << std::endl;
        return false;
    }

    // Insert time point
    sqlite3_bind_int(timeStmt, 1, timeStep);
    sqlite3_bind_double(timeStmt, 2, results.timePoints[timeStep]);
    rc = sqlite3_step(timeStmt);
    if (rc != SQLITE_DONE) {
        std::cerr << "Failed to insert time point: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_finalize(timeStmt);
        return false;
    }
    sqlite3_finalize(timeStmt);

    // Insert node voltages with bounds checking
    for (const auto& node : results.nodeVoltages) {
        // Make sure the vector is large enough for this time step
        if (timeStep >= static_cast<int>(node.second.size())) {
            std::cerr << "Warning: Node " << node.first << " has insufficient data for time step "
                << timeStep << " (size: " << node.second.size() << ")" << std::endl;
            continue;
        }

        sqlite3_bind_int(insertVoltageStmt, 1, timeStep);
        sqlite3_bind_text(insertVoltageStmt, 2, node.first.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_double(insertVoltageStmt, 3, node.second[timeStep]);

        rc = sqlite3_step(insertVoltageStmt);
        if (rc != SQLITE_DONE) {
            std::cerr << "Failed to insert node voltage: " << sqlite3_errmsg(db) << std::endl;
            return false;
        }

        sqlite3_reset(insertVoltageStmt);
    }

    // Insert branch currents with bounds checking
    for (const auto& branch : results.branchCurrents) {
        // Make sure the vector is large enough for this time step
        if (timeStep >= static_cast<int>(branch.second.size())) {
            std::cerr << "Warning: Branch " << branch.first << " has insufficient data for time step "
                << timeStep << " (size: " << branch.second.size() << ")" << std::endl;
            continue;
        }

        sqlite3_bind_int(insertCurrentStmt, 1, timeStep);
        sqlite3_bind_text(insertCurrentStmt, 2, branch.first.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_double(insertCurrentStmt, 3, branch.second[timeStep]);

        rc = sqlite3_step(insertCurrentStmt);
        if (rc != SQLITE_DONE) {
            std::cerr << "Failed to insert branch current: " << sqlite3_errmsg(db) << std::endl;
            return false;
        }

        sqlite3_reset(insertCurrentStmt);
    }

    // Commit transaction if buffer is full
    currentBufferSize++;
    if (currentBufferSize >= maxBufferSize) {
        rc = sqlite3_step(endTransactionStmt);
        if (rc != SQLITE_DONE) {
            std::cerr << "Failed to commit transaction: " << sqlite3_errmsg(db) << std::endl;
            return false;
        }

        sqlite3_reset(endTransactionStmt);
        currentBufferSize = 0;
    }

    return true;
}

bool DatabaseConnector::writeBulkData(const SimulationResults& results) {
    std::lock_guard<std::mutex> lock(dbMutex);

    if (db == nullptr) {
        std::cerr << "Not connected to database" << std::endl;
        return false;
    }

    int numTimeSteps = results.timePoints.size();

    // Start transaction
    char* errMsg = nullptr;
    int rc = sqlite3_exec(db, "BEGIN TRANSACTION;", nullptr, nullptr, &errMsg);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to start transaction: " << errMsg << std::endl;
        sqlite3_free(errMsg);
        return false;
    }

    // Insert time points one by one instead of in batches
    for (int t = 0; t < numTimeSteps; t++) {
        std::string timePointSQL = "INSERT OR REPLACE INTO time_points (time_step, simulation_time) VALUES (";
        timePointSQL += std::to_string(t) + ", " + std::to_string(results.timePoints[t]) + ");";

        rc = sqlite3_exec(db, timePointSQL.c_str(), nullptr, nullptr, &errMsg);
        if (rc != SQLITE_OK) {
            std::cerr << "Failed to insert time point " << t << ": " << errMsg << std::endl;
            sqlite3_free(errMsg);
            sqlite3_exec(db, "ROLLBACK;", nullptr, nullptr, nullptr);
            return false;
        }
    }

    // Insert node voltages one by one
    for (const auto& node : results.nodeVoltages) {
        for (int t = 0; t < numTimeSteps && t < node.second.size(); t++) {
            std::string voltageSQL = "INSERT OR REPLACE INTO node_voltages (time_step, node_name, voltage) VALUES (";
            voltageSQL += std::to_string(t) + ", '" + node.first + "', " + std::to_string(node.second[t]) + ");";

            rc = sqlite3_exec(db, voltageSQL.c_str(), nullptr, nullptr, &errMsg);
            if (rc != SQLITE_OK) {
                std::cerr << "Failed to insert node voltage for " << node.first << " at time " << t << ": " << errMsg << std::endl;
                sqlite3_free(errMsg);
                sqlite3_exec(db, "ROLLBACK;", nullptr, nullptr, nullptr);
                return false;
            }
        }
    }

    // Insert branch currents one by one
    for (const auto& branch : results.branchCurrents) {
        for (int t = 0; t < numTimeSteps && t < branch.second.size(); t++) {
            std::string currentSQL = "INSERT OR REPLACE INTO branch_currents (time_step, branch_name, current) VALUES (";
            currentSQL += std::to_string(t) + ", '" + branch.first + "', " + std::to_string(branch.second[t]) + ");";

            rc = sqlite3_exec(db, currentSQL.c_str(), nullptr, nullptr, &errMsg);
            if (rc != SQLITE_OK) {
                std::cerr << "Failed to insert branch current for " << branch.first << " at time " << t << ": " << errMsg << std::endl;
                sqlite3_free(errMsg);
                sqlite3_exec(db, "ROLLBACK;", nullptr, nullptr, nullptr);
                return false;
            }
        }
    }

    // Commit transaction
    rc = sqlite3_exec(db, "COMMIT;", nullptr, nullptr, &errMsg);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to commit transaction: " << errMsg << std::endl;
        sqlite3_free(errMsg);
        sqlite3_exec(db, "ROLLBACK;", nullptr, nullptr, nullptr);
        return false;
    }

    std::cout << "Successfully exported " << numTimeSteps << " time steps to database." << std::endl;
    std::cout << "Exported voltage data for " << results.nodeVoltages.size() << " nodes." << std::endl;
    std::cout << "Exported current data for " << results.branchCurrents.size() << " branches." << std::endl;

    return true;
}

bool DatabaseConnector::getRecentData(const std::vector<std::string>& nodeNames,
    const std::vector<std::string>& branchNames,
    int count,
    std::unordered_map<std::string, std::vector<double>>& nodeVoltages,
    std::unordered_map<std::string, std::vector<double>>& branchCurrents,
    std::vector<double>& timePoints) {
    std::lock_guard<std::mutex> lock(dbMutex);

    if (db == nullptr) {
        std::cerr << "Not connected to database" << std::endl;
        return false;
    }

    // Clear output containers
    nodeVoltages.clear();
    branchCurrents.clear();
    timePoints.clear();

    // Get max time step
    sqlite3_stmt* maxTimeStepStmt = nullptr;
    const char* maxTimeStepSQL = "SELECT MAX(time_step) FROM time_points;";

    int rc = sqlite3_prepare_v2(db, maxTimeStepSQL, -1, &maxTimeStepStmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare max time step query: " << sqlite3_errmsg(db) << std::endl;
        return false;
    }

    rc = sqlite3_step(maxTimeStepStmt);
    if (rc != SQLITE_ROW) {
        std::cerr << "Failed to get max time step: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_finalize(maxTimeStepStmt);
        return false;
    }

    int maxTimeStep = sqlite3_column_int(maxTimeStepStmt, 0);
    sqlite3_finalize(maxTimeStepStmt);

    // Calculate start time step
    int startTimeStep = std::max(0, maxTimeStep - count + 1);

    // Get time points
    std::string timePointsSQL =
        "SELECT time_step, simulation_time FROM time_points WHERE time_step >= " +
        std::to_string(startTimeStep) + " ORDER BY time_step;";

    sqlite3_stmt* timePointsStmt = nullptr;
    rc = sqlite3_prepare_v2(db, timePointsSQL.c_str(), -1, &timePointsStmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare time points query: " << sqlite3_errmsg(db) << std::endl;
        return false;
    }

    std::unordered_map<int, int> timeStepToIndex;
    int index = 0;

    while ((rc = sqlite3_step(timePointsStmt)) == SQLITE_ROW) {
        int timeStep = sqlite3_column_int(timePointsStmt, 0);
        double simulationTime = sqlite3_column_double(timePointsStmt, 1);

        timePoints.push_back(simulationTime);
        timeStepToIndex[timeStep] = index++;
    }

    sqlite3_finalize(timePointsStmt);

    // Initialize output containers
    for (const auto& node : nodeNames) {
        nodeVoltages[node] = std::vector<double>(timePoints.size(), 0.0);
    }

    for (const auto& branch : branchNames) {
        branchCurrents[branch] = std::vector<double>(timePoints.size(), 0.0);
    }

    // Get node voltages
    for (const auto& node : nodeNames) {
        std::string nodeVoltageSQL =
            "SELECT time_step, voltage FROM node_voltages "
            "WHERE node_name = ? AND time_step >= " +
            std::to_string(startTimeStep) + " ORDER BY time_step;";

        sqlite3_stmt* nodeVoltageStmt = nullptr;
        rc = sqlite3_prepare_v2(db, nodeVoltageSQL.c_str(), -1, &nodeVoltageStmt, nullptr);
        if (rc != SQLITE_OK) {
            std::cerr << "Failed to prepare node voltage query: " << sqlite3_errmsg(db) << std::endl;
            return false;
        }

        sqlite3_bind_text(nodeVoltageStmt, 1, node.c_str(), -1, SQLITE_STATIC);

        while ((rc = sqlite3_step(nodeVoltageStmt)) == SQLITE_ROW) {
            int timeStep = sqlite3_column_int(nodeVoltageStmt, 0);
            double voltage = sqlite3_column_double(nodeVoltageStmt, 1);

            int vectorIndex = timeStepToIndex[timeStep];
            nodeVoltages[node][vectorIndex] = voltage;
        }

        sqlite3_finalize(nodeVoltageStmt);
    }

    // Get branch currents
    for (const auto& branch : branchNames) {
        std::string branchCurrentSQL =
            "SELECT time_step, current FROM branch_currents "
            "WHERE branch_name = ? AND time_step >= " +
            std::to_string(startTimeStep) + " ORDER BY time_step;";

        sqlite3_stmt* branchCurrentStmt = nullptr;
        rc = sqlite3_prepare_v2(db, branchCurrentSQL.c_str(), -1, &branchCurrentStmt, nullptr);
        if (rc != SQLITE_OK) {
            std::cerr << "Failed to prepare branch current query: " << sqlite3_errmsg(db) << std::endl;
            return false;
        }

        sqlite3_bind_text(branchCurrentStmt, 1, branch.c_str(), -1, SQLITE_STATIC);

        while ((rc = sqlite3_step(branchCurrentStmt)) == SQLITE_ROW) {
            int timeStep = sqlite3_column_int(branchCurrentStmt, 0);
            double current = sqlite3_column_double(branchCurrentStmt, 1);

            int vectorIndex = timeStepToIndex[timeStep];
            branchCurrents[branch][vectorIndex] = current;
        }

        sqlite3_finalize(branchCurrentStmt);
    }

    return true;
}