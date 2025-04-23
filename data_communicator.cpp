#include "data_communicator.h"
#include <iostream>
#include <sstream>
#include <thread>

DataCommunicator::DataCommunicator(const std::string& address, int port,
    int maxRetries, int retryDelayMs)
    : serverAddress(address), serverPort(port), connected(false), clientSocket(INVALID_SOCKET),
    maxRetries(maxRetries), retryDelayMs(retryDelayMs), currentRetryCount(0) {

    // Initialize Winsock
    WSADATA wsaData;
    int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (result != 0) {
        std::cerr << "WSAStartup failed: " << result << std::endl;
    }
}

DataCommunicator::~DataCommunicator() {
    disconnect();
    WSACleanup();
}

// Add these improvements to data_communicator.cpp

bool DataCommunicator::connect() {
    std::lock_guard<std::mutex> lock(socketMutex);

    // If already connected, return success
    if (connected && clientSocket != INVALID_SOCKET) {
        std::cout << "Already connected to visualization server" << std::endl;
        return true;
    }

    currentRetryCount = 0;
    std::cout << "Attempting to connect to visualization server at "
        << serverAddress << ":" << serverPort << std::endl;

    while (currentRetryCount < maxRetries) {
        // Create socket
        clientSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (clientSocket == INVALID_SOCKET) {
            std::cerr << "Error creating socket: " << WSAGetLastError()
                << " (Attempt " << currentRetryCount + 1 << "/" << maxRetries << ")" << std::endl;
            currentRetryCount++;
            Sleep(retryDelayMs);
            continue;
        }

        // Set up server address
        sockaddr_in serverAddr;
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = htons(serverPort);

        // Convert string address to network address
        if (inet_pton(AF_INET, serverAddress.c_str(), &serverAddr.sin_addr) != 1) {
            std::cerr << "Invalid address: " << serverAddress
                << " (Attempt " << currentRetryCount + 1 << "/" << maxRetries << ")" << std::endl;
            closesocket(clientSocket);
            clientSocket = INVALID_SOCKET;
            currentRetryCount++;
            Sleep(retryDelayMs);
            continue;
        }

        // Set socket timeout
        int timeout = 5000; // 5 seconds
        if (setsockopt(clientSocket, SOL_SOCKET, SO_RCVTIMEO, (char*)&timeout, sizeof(timeout)) != 0) {
            std::cerr << "Failed to set receive timeout: " << WSAGetLastError() << std::endl;
            // Continue anyway, not critical
        }

        if (setsockopt(clientSocket, SOL_SOCKET, SO_SNDTIMEO, (char*)&timeout, sizeof(timeout)) != 0) {
            std::cerr << "Failed to set send timeout: " << WSAGetLastError() << std::endl;
            // Continue anyway, not critical
        }

        // Connect to server
        std::cout << "Attempting connection to " << serverAddress << ":" << serverPort
            << " (Attempt " << currentRetryCount + 1 << "/" << maxRetries << ")" << std::endl;

        if (::connect(clientSocket, (SOCKADDR*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
            int error = WSAGetLastError();
            std::cerr << "Connection failed: " << error << " - ";

            // Print human-readable error message
            switch (error) {
            case WSAECONNREFUSED:
                std::cerr << "Connection refused. Is the visualization server running?";
                break;
            case WSAENETUNREACH:
                std::cerr << "Network unreachable";
                break;
            case WSAETIMEDOUT:
                std::cerr << "Connection attempt timed out";
                break;
            default:
                std::cerr << "Socket error";
                break;
            }

            std::cerr << std::endl;

            closesocket(clientSocket);
            clientSocket = INVALID_SOCKET;
            currentRetryCount++;
            Sleep(retryDelayMs);
            continue;
        }

        connected = true;
        std::cout << "Successfully connected to visualization server at "
            << serverAddress << ":" << serverPort << std::endl;
        return true;
    }

    std::cerr << "Failed to connect to visualization server after "
        << maxRetries << " attempts" << std::endl;
    std::cerr << "Make sure the visualization server is running at "
        << serverAddress << ":" << serverPort << std::endl;
    std::cerr << "You can run the visualization server by executing the Python script: "
        << "python visualization_server.py" << std::endl;

    return false;
}

void DataCommunicator::disconnect() {
    std::lock_guard<std::mutex> lock(socketMutex);

    if (connected && clientSocket != INVALID_SOCKET) {
        closesocket(clientSocket);
        clientSocket = INVALID_SOCKET;
        connected = false;
        std::cout << "Disconnected from visualization server" << std::endl;
    }
}

// Enhance the DataCommunicator implementation in data_communicator.cpp

bool DataCommunicator::sendInitialData(const SimulationResults& results,
    const std::vector<std::string>& nodeNames,
    const std::unordered_map<std::string, std::pair<float, float>>& nodePositions,
    const std::vector<std::pair<std::string, std::string>>& branches) {

    std::lock_guard<std::mutex> lock(socketMutex);

    if (!connected) {
        return false;
    }

    // Create JSON with simulation parameters
    json initialData;
    initialData["total_time_steps"] = results.timePoints.size();
    initialData["time_points"] = results.timePoints;
    initialData["node_names"] = nodeNames;

    // Extract branch names
    std::vector<std::string> branchNames;
    for (const auto& branch : results.branchCurrents) {
        branchNames.push_back(branch.first);
    }
    initialData["branch_names"] = branchNames;

    // Add node positions
    json positionsJson = json::object();
    for (const auto& node : nodePositions) {
        positionsJson[node.first] = { node.second.first, node.second.second };
    }
    initialData["node_positions"] = positionsJson;

    // Add branches
    json branchesJson = json::array();
    for (const auto& branch : branches) {
        branchesJson.push_back({ branch.first, branch.second });
    }
    initialData["branches"] = branchesJson;

    // Send the JSON data
    std::string jsonStr = initialData.dump() + "\n";
    if (send(clientSocket, jsonStr.c_str(), static_cast<int>(jsonStr.size()), 0) == SOCKET_ERROR) {
        std::cerr << "Failed to send initial data: " << WSAGetLastError() << std::endl;
        disconnect();
        return false;
    }

    return true;
}

bool DataCommunicator::sendTimeStepData(int timeStep, const SimulationResults& results) {
    std::lock_guard<std::mutex> lock(socketMutex);

    if (!connected || timeStep >= results.timePoints.size()) {
        return false;
    }

    // Create JSON with current time step data
    json data;
    data["time_step"] = timeStep;

    // Add node voltages
    json voltagesJson = json::object();
    for (const auto& node : results.nodeVoltages) {
        if (timeStep < static_cast<int>(node.second.size())) {
            voltagesJson[node.first] = node.second[timeStep];
        }
    }
    data["voltages"] = voltagesJson;

    // Add branch currents
    json currentsJson = json::object();
    for (const auto& branch : results.branchCurrents) {
        if (timeStep < static_cast<int>(branch.second.size())) {
            currentsJson[branch.first] = branch.second[timeStep];
        }
    }
    data["currents"] = currentsJson;

    // Send the JSON data
    std::string jsonStr = data.dump() + "\n";
    int result = send(clientSocket, jsonStr.c_str(), static_cast<int>(jsonStr.size()), 0);
    if (result == SOCKET_ERROR) {
        std::cerr << "Failed to send time step data: " << WSAGetLastError() << std::endl;
        disconnect();
        return false;
    }

    // Wait for acknowledgment
    char buffer[16];
    int bytesRead = recv(clientSocket, buffer, sizeof(buffer) - 1, 0);
    if (bytesRead == SOCKET_ERROR) {
        std::cerr << "Failed to receive acknowledgment: " << WSAGetLastError() << std::endl;
        disconnect();
        return false;
    }

    buffer[bytesRead] = '\0';
    if (std::string(buffer).find("ACK") == std::string::npos) {
        std::cerr << "Invalid acknowledgment received" << std::endl;
        return false;
    }

    return true;
}

bool DataCommunicator::sendCommand(const std::string& command, const json& params) {
    std::lock_guard<std::mutex> lock(socketMutex);

    if (!connected) {
        return false;
    }

    json cmdJson;
    cmdJson["command"] = command;
    cmdJson["params"] = params;

    // Send the JSON data
    std::string jsonStr = cmdJson.dump() + "\n";
    if (send(clientSocket, jsonStr.c_str(), static_cast<int>(jsonStr.size()), 0) == SOCKET_ERROR) {
        std::cerr << "Failed to send command: " << WSAGetLastError() << std::endl;
        disconnect();
        return false;
    }

    // Wait for acknowledgment
    char buffer[16];
    int bytesRead = recv(clientSocket, buffer, sizeof(buffer) - 1, 0);
    if (bytesRead == SOCKET_ERROR) {
        std::cerr << "Failed to receive command acknowledgment: " << WSAGetLastError() << std::endl;
        disconnect();
        return false;
    }

    buffer[bytesRead] = '\0';
    if (std::string(buffer).find("ACK") == std::string::npos) {
        std::cerr << "Invalid command acknowledgment received" << std::endl;
        return false;
    }

    return true;
}