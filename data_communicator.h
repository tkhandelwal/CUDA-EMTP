#pragma once

#include <string>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <mutex>
#include </Users/Tanuj.Khandelwal/source/repos/eMTP-CUDA-V2/packages/nlohmann.json.3.12.0/build/native/include/nlohmann/json.hpp>
#include "simulation_results.h"

#ifdef _WIN32
#include <WinSock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#pragma comment(lib, "Ws2_32.lib")
#define SOCKET_TYPE SOCKET
#define INVALID_SOCKET_VALUE INVALID_SOCKET
#define SOCKET_ERROR_VALUE SOCKET_ERROR
#define CLOSE_SOCKET(s) closesocket(s)
#define SLEEP_MS(ms) Sleep(ms)
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#define SOCKET_TYPE int
#define INVALID_SOCKET_VALUE -1
#define SOCKET_ERROR_VALUE -1
#define CLOSE_SOCKET(s) close(s)
#define SLEEP_MS(ms) usleep(ms * 1000)
#endif

using json = nlohmann::json;

// Forward declaration
struct SimulationResults;

class DataCommunicator {
private:
    SOCKET_TYPE clientSocket;
    bool connected;
    std::string serverAddress;
    int serverPort;
    std::mutex socketMutex;

    // Connection retry parameters
    int maxRetries;
    int retryDelayMs;
    int currentRetryCount;

public:
    // Constructor declaration (not definition)
    DataCommunicator(const std::string& address = "localhost", int port = 5555,
        int maxRetries = 3, int retryDelayMs = 1000);

    // Destructor declaration
    ~DataCommunicator();

    // Method declarations
    bool connect();
    void disconnect();
    bool isConnected() const { return connected; } // Simple inline method can stay

    bool sendInitialData(const SimulationResults& results,
        const std::vector<std::string>& nodeNames,
        const std::unordered_map<std::string, std::pair<float, float>>& nodePositions,
        const std::vector<std::pair<std::string, std::string>>& branches);

    bool sendTimeStepData(int timeStep, const SimulationResults& results);
    bool sendCommand(const std::string& command, const json& params = json());
};