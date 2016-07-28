#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "RecordStructs.h"
#include "CSVWriter.h"
#include <time.h>
#include <algorithm>
#include "printRecord.h"

__host__ void CHECK_CUDA(cudaError_t cu);
__device__ void recordRun(params param, double delta, runRecord* recordPointer, unsigned long long int* recordNum);
__device__ params getParams(unsigned long long int offset);
__global__ void calculateGCF(unsigned long long int offset, runRecord* recordPointer, unsigned long long int* recordNum, double convergeTo);