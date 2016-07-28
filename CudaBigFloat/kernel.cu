
#include "kernel.cuh"

//Minimum and maximum values for search
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

#define PRINTRESULTS //Print results to console
#define PRINTTOFILE //Save results to file
#define PRINTPROGRESS //Print out progress messages periodacally

#define RANGE(NUM) (NUM##MAX + 1 - NUM##MIN)

#define TPB 1024
#define BLOCKS 1024
#define THREADSATONCE (TPB*BLOCKS)

#define PERMUTATIONS (RANGE(A0)*RANGE(B0)*RANGE(A)*RANGE(B)*RANGE(C)*RANGE(D)*RANGE(E))

#define NUMRUNS (((PERMUTATIONS-1)/THREADSATONCE)+1)  



/**
  * Helper function to check for CUDA errors
  * @param cu the error to check
  */
__host__ void CHECK_CUDA(cudaError_t cu){
	if (cu != cudaSuccess){
		std::cout << "CUDA ERROR: "<< cudaGetErrorString(cu) << std::endl;
	}
}


/**
  * Record the result of a run
  * @param param the starting parameters of the function
  * @param result the result to record
  * @param delta the difference between the result and the number
  * @param recordPointer a the array where the records are stored
  * @param recordNum a pointer to the current record index
  */
__device__ void recordRun(params param, float result, float delta, runRecord* recordPointer, unsigned long long int* recordNum){
	unsigned long long int address = atomicAdd(recordNum, 1);

	runRecord curRec;
	curRec.param = param;
	curRec.delta = delta;
	curRec.result = result;

	recordPointer[address] = curRec;	
	
}

/**
  * Get the runtime parameters
  * @param offset the offset tfor the parameters
  */
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

/**
  * Calculate a continued fraction, given the starting parameters.
  * @param par The starting parameters of the calculation
  */
__device__ float calcFraction(params runPars){
	int iterNum = 0;

	float hBefore1 = 1;
	float hBefore2 = 0;
	float kBefore1 = 0;
	float kBefore2 = 1;
	float aBefore = 1;

	float convergent = 0;

	while (iterNum < MAXTERMS){
		float newA = (iterNum == 0) ? runPars.a0 : runPars.a*iterNum*iterNum + runPars.b*iterNum + runPars.c;
		float newB = (iterNum == 0) ? runPars.b0 : runPars.d*iterNum + runPars.e;

		float curNum = newB*hBefore1 + aBefore*hBefore2;
		float curDen = newB*kBefore1 + aBefore*kBefore2;

		if (curDen == 0) return NAN;

		convergent = curNum / curDen;

		hBefore2 = hBefore1;
		hBefore1 = curNum;

		kBefore2 = kBefore1;
		kBefore1 = curDen;

		iterNum++;
		aBefore = newA;
	}

	return convergent;
}

/**
  * The method for calculating the continued fraction and logging the result
  * @param offset the starting thread offset
  * @param recordPointer the array for the result of the calculation
  * @param recordNum the index for the array
  * @param convergeTo the value that the delta is calculated from
  */
__global__ void calculateGCF(unsigned long long int offset, runRecord* recordPointer, unsigned long long int* recordNum, float convergeTo){
	params runPars = getParams(offset);

	float convergent = calcFraction(runPars);

	float delta = abs(convergent - convergeTo);

	if (delta < MAXDELTA){
		recordRun(runPars, convergent, delta, recordPointer, recordNum);
	}
}





int main(){


#ifdef PRINTTOFILE
	time_t currentTime;
	struct tm* timeInfo;
	
	time(&currentTime);
	timeInfo = localtime(&currentTime);

	std::stringstream filename;
	filename << "Result " << timeInfo->tm_year+1900 << " " << timeInfo->tm_mon + 1 << " " <<
		timeInfo->tm_mday << " " << timeInfo->tm_hour << " " << timeInfo->tm_min << " " << timeInfo->tm_sec << ".csv";

	
	CSVWriter fileWriter(filename.str());
#endif

	std::cout << std::setprecision(15);

	runRecord* d_recordPointer;
	unsigned long long int* d_recordNum;

	CHECK_CUDA(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

	CHECK_CUDA(cudaMalloc(&d_recordPointer, sizeof(runRecord)*THREADSATONCE));
	CHECK_CUDA(cudaMalloc(&d_recordNum, sizeof(unsigned long long int)));

	cudaEvent_t startingevent, endevent;
	CHECK_CUDA(cudaEventCreate(&startingevent));
	CHECK_CUDA(cudaEventCreate(&endevent));
	CHECK_CUDA(cudaEventRecord(startingevent));

	for (unsigned long long int i = 0; i < NUMRUNS; i++){

		

		CHECK_CUDA(cudaMemset(d_recordNum, 0, sizeof(unsigned long long int)));

		calculateGCF << <BLOCKS, TPB >> >(i*THREADSATONCE, d_recordPointer, d_recordNum, 14.13472514173469);
		cudaDeviceSynchronize();
		CHECK_CUDA(cudaGetLastError());

		unsigned long long int h_recordNum;
		CHECK_CUDA(cudaMemcpy(&h_recordNum, d_recordNum, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));


		runRecord* h_recordPointer = (runRecord*)malloc(h_recordNum*sizeof(runRecord));
		CHECK_CUDA(cudaMemcpy(h_recordPointer, d_recordPointer, h_recordNum*sizeof(runRecord), cudaMemcpyDeviceToHost));


		for (int j = 0; j < h_recordNum; j++){
#ifdef PRINTRESULTS
			printRecord(h_recordPointer[j]);
#endif

#ifdef PRINTTOFILE
			fileWriter.write(h_recordPointer[j]);
#endif
		}

#ifdef PRINTPROGRESS
		if(i%100==0) std::cout << "Offset: " << i*THREADSATONCE << ", Progress: " << ((double)i * 100) / NUMRUNS << "%" << std::endl;
#endif

		free(h_recordPointer);
	}


#ifdef PRINTTOFILE
	fileWriter.flush();
#endif
	
	CHECK_CUDA(cudaEventRecord(endevent));

	float timeElapsed = 0;
	CHECK_CUDA(cudaDeviceSynchronize());

	CHECK_CUDA(cudaEventElapsedTime(&timeElapsed, startingevent, endevent));
	std::cout << "Time elapsed: " << std::setprecision(0) << timeElapsed/1000 << " seconds" << std::endl;

	CHECK_CUDA(cudaFree(d_recordPointer));
	CHECK_CUDA(cudaFree(d_recordNum));
	cudaDeviceReset();
	system("pause");
	return 0;
}