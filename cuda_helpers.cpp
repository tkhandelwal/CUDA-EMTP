#include "cuda_helpers.h"
#include <iostream>

// Function is now defined in cuda_kernels.cu
// CUDA runtime error checking
//void check_cuda_error(cudaError_t result, const char* func, const char* file, int line) {
//    if (result != cudaSuccess) {
//        std::cerr << "CUDA error: " << cudaGetErrorString(result)
//            << " at " << file << ":" << line << " '" << func << "'" << std::endl;
//        // In a commercial product, we might want a more graceful error handling mechanism
//        cudaDeviceReset();
//        exit(EXIT_FAILURE);
//    }
//}

// cuSPARSE error checking
void check_cusparse_error(cusparseStatus_t result, const char* func, const char* file, int line) {
    if (result != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "cuSPARSE error: " << result
            << " at " << file << ":" << line << " '" << func << "'" << std::endl;
        throw std::runtime_error("cuSPARSE error");
    }
}

// cuSOLVER error checking
void check_cusolver_error(cusolverStatus_t result, const char* func, const char* file, int line) {
    if (result != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cuSOLVER error: " << result
            << " at " << file << ":" << line << " '" << func << "'" << std::endl;
        throw std::runtime_error("cuSOLVER error");
    }
}

// cuBLAS error checking
void check_cublas_error(cublasStatus_t result, const char* func, const char* file, int line) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error: " << result
            << " at " << file << ":" << line << " '" << func << "'" << std::endl;
        throw std::runtime_error("cuBLAS error");
    }
}