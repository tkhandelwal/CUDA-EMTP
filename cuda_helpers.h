// cuda_helpers.h
#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <cublas_v2.h>

// CUDA runtime error checking - declare as extern
extern void check_cuda_error(cudaError_t result, const char* func, const char* file, int line);

// cuSPARSE error checking
void check_cusparse_error(cusparseStatus_t result, const char* func, const char* file, int line);

// cuSOLVER error checking
void check_cusolver_error(cusolverStatus_t result, const char* func, const char* file, int line);

// cuBLAS error checking
void check_cublas_error(cublasStatus_t result, const char* func, const char* file, int line);

// Define macros
#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)
#define CHECK_CUSPARSE_ERROR(val) check_cusparse_error((val), #val, __FILE__, __LINE__)
#define CHECK_CUSOLVER_ERROR(val) check_cusolver_error((val), #val, __FILE__, __LINE__)
#define CHECK_CUBLAS_ERROR(val) check_cublas_error((val), #val, __FILE__, __LINE__)