#pragma once

#include <time.h>
#include <iostream>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include "CSVWriter.h"
#include "Params.h"
#include "RunRecord.h"

__host__ void CHECK_CUDA(cudaError_t cu);
__device__ void recordRun(params param, double delta, runRecord* recordPointer, unsigned long long int* recordNum);
__device__ params getParams(unsigned long long int offset);
__global__ void calculateGCF(unsigned long long int offset, runRecord* recordPointer, unsigned long long int* recordNum, double convergeTo);
__device__ double calcFraction(params runPars);

int main();