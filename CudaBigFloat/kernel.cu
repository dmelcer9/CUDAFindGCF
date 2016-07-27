
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>

#define A0MIN -9
#define A0MAX 9
#define B0MIN -9
#define B0MAX 9
#define AMIN -9
#define AMAX 9
#define BMIN -9
#define BMAX 9
#define CMIN -9
#define CMAX 9
#define DMIN -9
#define DMAX 9
#define EMIN -9
#define EMAX 9

#define MAXTERMS 15
#define MAXDELTA 0.00001

#define RANGE(NUM) (NUM##MAX + 1 - NUM##MIN)

#define TPB 1024
#define BLOCKS 1024
#define THREADSATONCE (TPB*BLOCKS)

#define PERMUTATIONS (RANGE(A0)*RANGE(B0)*RANGE(A)*RANGE(B)*RANGE(C)*RANGE(D)*RANGE(E))

#define NUMRUNS (((PERMUTATIONS-1)/THREADSATONCE)+1)  


__host__ void CHECK_CUDA(cudaError_t cu){
	if (cu != cudaSuccess){
		std::cout << "CUDA ERROR: "<< cudaGetErrorString(cu) << std::endl;
	}
}

typedef struct params{
	int a0;
	int b0;
	int a;
	int b;
	int c;
	int d;
	int e;
} params;

typedef struct runRecord{
	params param;
	double delta;
} runRecord;

__device__ void recordRun(params param, double delta, runRecord* recordPointer, unsigned long long int* recordNum){
	unsigned long long int address = atomicAdd(recordNum, 1);

	runRecord curRec;
	curRec.param = param;
	curRec.delta = delta;

	recordPointer[address] = curRec;	
	
}

__device__ params getParams(unsigned long long int offset){
	unsigned long long int blocksz = blockDim.x*blockDim.y*blockDim.z;
	unsigned long long int block1d = threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
	unsigned long long int grid1d = blockIdx.x + gridDim.x*blockIdx.y + gridDim.x*gridDim.y*blockIdx.z;

	unsigned long long int globalIdx1D = grid1d*blocksz + block1d;
	long long int workId = globalIdx1D + offset;

	params par;
	
	par.e = workId % RANGE(E) + EMIN;
	workId /= RANGE(E);

	par.d = workId % RANGE(D) + DMIN;
	workId /= RANGE(D);

	par.c = workId % RANGE(C) + CMIN;
	workId /= RANGE(C);

	par.b = workId % RANGE(B) + BMIN;
	workId /= RANGE(B);

	par.a = workId % RANGE(A) + AMIN;
	workId /= RANGE(A);

	par.b0 = workId % RANGE(B0) + B0MIN;
	workId /= RANGE(B0);

	par.a0 = workId % RANGE(A0) + A0MIN;
	
	return par;

}

__global__ void calcFraction(unsigned long long int offset, runRecord* recordPointer, unsigned long long int* recordNum, double convergeTo){
	params runPars = getParams(offset);
	/*runPars.a0 = 11;
	runPars.b0 = 17;
	runPars.a = -5;
	runPars.b = 3;
	runPars.c = -7;
	runPars.d = 5;
	runPars.e = -1;*/


	int iterNum = 0;

	double hBefore1 = 1;
	double hBefore2 = 0;
	double kBefore1 = 0;
	double kBefore2 = 1;
	double aBefore = 1;

	double delta = 0;
	double prevDelta = convergeTo;

	//recordRun(runPars, threadIdx.x + blockDim.x*blockIdx.x , recordPointer, recordNum);

	while (iterNum < MAXTERMS){
		double newA = (iterNum == 0) ? runPars.a0 : runPars.a*iterNum*iterNum + runPars.b*iterNum + runPars.c;
		double newB = (iterNum == 0) ? runPars.b0 : runPars.d*iterNum + runPars.e;

		double curNum = newB*hBefore1 + aBefore*hBefore2;
		double curDen = newB*kBefore1 + aBefore*kBefore2;

		if (curDen == 0) return;

		double convergent = curNum / curDen;

		delta = abs(convergent - convergeTo);
		

		hBefore2 = hBefore1;
		hBefore1 = curNum;

		kBefore2 = kBefore1;
		kBefore1 = curDen;

		prevDelta = delta;
		iterNum++;
		aBefore = newA;
	}

	if (delta < MAXDELTA){
		recordRun(runPars, delta, recordPointer, recordNum);
	}
}


std::string paramsToString(params par){
	
	return "A0= " + std::to_string(par.a0) + " ,B0= " + std::to_string(par.b0) + " ,A= " + std::to_string(par.a) + 
		" ,B= " + std::to_string(par.b) + " ,C= " + std::to_string(par.c) + " ,D= " + std::to_string(par.d) + " ,E= " + std::to_string(par.e);
	
}

std::string runRecordToString(runRecord rec){
	std::ostringstream os;
	os << paramsToString(rec.param) << std::setprecision(15) << " ,delta= " << rec.delta;
	return os.str();
}



//void compactThread

int main(){
	
	/*

	cudaEvent_t* events = (cudaEvent_t*) malloc(sizeof(cudaEvent_t)*NUMRUNS);
	for (int i = 0; i < NUMRUNS; i++){
		CHECK_CUDA(cudaEventCreate(&events[i]));
	}

	runRecord** recordGrid = (runRecord**) malloc(sizeof(runRecord*)*NUMRUNS);
	*/


	std::cout << std::setprecision(15);

	runRecord* d_recordPointer;
	unsigned long long int* d_recordNum;

	CHECK_CUDA(cudaMalloc(&d_recordPointer, sizeof(runRecord)*THREADSATONCE));
	CHECK_CUDA(cudaMalloc(&d_recordNum, sizeof(unsigned long long int)));


	for (unsigned long long int i = 0; i < NUMRUNS; i++){

		
		CHECK_CUDA(cudaMemset(d_recordNum, 0, sizeof(unsigned long long int)));

		calcFraction << <BLOCKS, TPB >> >(i*THREADSATONCE, d_recordPointer, d_recordNum, 14.13472514173469);
		cudaDeviceSynchronize();
		CHECK_CUDA(cudaGetLastError());

		unsigned long long int h_recordNum;
		CHECK_CUDA(cudaMemcpy(&h_recordNum, d_recordNum, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));


		runRecord* h_recordPointer = (runRecord*)malloc(h_recordNum*sizeof(runRecord));
		CHECK_CUDA(cudaMemcpy(h_recordPointer, d_recordPointer, h_recordNum*sizeof(runRecord), cudaMemcpyDeviceToHost));


		for (int j = 0; j < h_recordNum; j++){
			std::cout << runRecordToString(h_recordPointer[j]) << std::endl;
		}

		
		free(h_recordPointer);
	}

	CHECK_CUDA(cudaFree(d_recordPointer));
	CHECK_CUDA(cudaFree(d_recordNum));
	cudaDeviceReset();
	//system("pause");
	return 0;
}