// emtp_master.h
#pragma once

// Standard headers
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cmath>
#include <algorithm>
#include <unordered_map>

// CUDA headers
#include <cuda_runtime.h>
#include <cuda.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cusolverSp.h>

// Project headers
#include "cuda_helpers.h"
#include "simulation_results.h"
#include "network_element.h"
#include "boundary_node.h"
#include "subnetwork.h"
#include "emtp_solver.h"
#include "data_communicator.h"
#include "database_connector.h"
#include "visualization.h"
#include "cuda_kernels.cuh"