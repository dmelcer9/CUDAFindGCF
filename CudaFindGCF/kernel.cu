
#include "kernel.cuh"

//Minimum and maximum parameters for search
const long long AMIN = -19L;
const long long AMAX = 19L;
const long long BMIN = -19L;
const long long BMAX = 19L;
const long long CMIN = -19L;
const long long CMAX = 19L;
const long long DMIN = -19L;
const long long DMAX = 19L;
const long long EMIN = -19L;
const long long EMAX = 19L;
const long long FMIN = -19L;
const long long FMAX = 19L;
const long long GMIN = -19L;
const long long GMAX = 19L;

#define FIRSTZERO 1
#define LASTZERO 2

#define NUMZEROS (1 + LASTZERO - FIRSTZERO)

#define MAXTERMS 15
#define MAXDELTA 1e-7

#define PRINTRESULTS //Print results to console
#define PRINTTOFILE //Save results to file
#define PRINTPROGRESS //Print out progress messages periodically
//#define PROCESSFILTEREDFRACTIONS

#define RANGE(NUM) (NUM##MAX + 1 - NUM##MIN)

#define TPB 1024
#define BLOCKS 1024
#define THREADSATONCE (TPB*BLOCKS)

#define PERMUTATIONS (RANGE(A)*RANGE(B)*RANGE(C)*RANGE(D)*RANGE(E)*RANGE(F)*RANGE(G))

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
__device__ void recordRun(params param, double result, double delta, runRecord* recordPointer, unsigned long long int* recordNum){
	unsigned long long int address = atomicAdd(recordNum, 1);

	runRecord curRec;
	curRec.param = param;
	curRec.delta = delta;
	curRec.result = result;

	recordPointer[address] = curRec;	
	
}

/**
  * Get the runtime parameters
  * @param offset the offset for the parameters
  * @return the parameters for the run
  */
__device__ params getParams(unsigned long long int offset, double convergeTo){
	unsigned long long int blocksz = blockDim.x*blockDim.y*blockDim.z;
	unsigned long long int block1d = threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
	unsigned long long int grid1d = blockIdx.x + gridDim.x*blockIdx.y + gridDim.x*gridDim.y*blockIdx.z;

	unsigned long long int globalIdx1D = grid1d*blocksz + block1d;
	long long int workId = globalIdx1D + offset;

	params par;

	
	par.g = workId % RANGE(G) + GMIN;
	workId /= RANGE(G);

	par.f = workId % RANGE(F) + FMIN;
	workId /= RANGE(F);
	
	par.e = workId % RANGE(E) + EMIN;
	workId /= RANGE(E);

	par.d = workId % RANGE(D) + DMIN;
	workId /= RANGE(D);

	par.c = workId % RANGE(C) + CMIN;
	workId /= RANGE(C);

	par.b = workId % RANGE(B) + BMIN;
	workId /= RANGE(B);

	par.a = workId % RANGE(A) + AMIN;
	
	par.b0 = (int)convergeTo;
	
	return par;

}

/**
  * Calculate a continued fraction, given the starting parameters.
  * @param par The starting parameters of the calculation
  * @return the number that the algorithm converged to
  */
__device__ double calcFraction(params runPars){

	double hBefore1 = 1;
	double hBefore2 = 0;
	double kBefore1 = 0;
	double kBefore2 = 1;
	double aBefore = 1;

	double convergent = 0;

	for (int iterNum = 0; iterNum < MAXTERMS; iterNum++){
			
		double newA = runPars.a*iterNum*iterNum*iterNum + runPars.b*iterNum*iterNum + runPars.c*iterNum + runPars.d;
		double newB = (iterNum == 0) ? runPars.b0 : runPars.e*iterNum*iterNum + runPars.f*iterNum + runPars.g;

		double curNum = newB*hBefore1 + aBefore*hBefore2;
		double curDen = newB*kBefore1 + aBefore*kBefore2;

		convergent = curNum / curDen;

		hBefore2 = hBefore1;
		hBefore1 = curNum;

		kBefore2 = kBefore1;
		kBefore1 = curDen;

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
__global__ void calculateGCF(unsigned long long int offset, runRecord* recordPointer, unsigned long long int* recordNum, double convergeTo){
	params runPars = getParams(offset,convergeTo);

	double convergent = calcFraction(runPars);

	double delta = abs(convergent - convergeTo);

	if (delta < MAXDELTA){
		recordRun(runPars, convergent, delta, recordPointer, recordNum);
	}
}



int main(){



	cudaEvent_t startingevent, endevent;
	CHECK_CUDA(cudaEventCreate(&startingevent));
	CHECK_CUDA(cudaEventCreate(&endevent));
	CHECK_CUDA(cudaEventRecord(startingevent));

	
	setupBestToFile();

	for (int z = (FIRSTZERO-1); z < LASTZERO; z++){
		setPrintToFileZeroNum(z + 1);
		ProcessQueue<runRecord> queue;

		queue.addProcess(addResultBestToFile);
		queue.addCleanup(markBestResult);

#ifdef PRINTTOFILE
		queue.addSetup(setupPrintToFile);
		queue.addProcess(processPrintToFile);
		queue.addCleanup(cleanupPrintToFile);
#endif

#ifdef PRINTRESULTS
		queue.addProcess(printRecord);
#endif

#ifdef PROCESSFILTEREDFRACTIONS
		queue.addSetup(setupProcessFilteredFractions);
		queue.addProcess(processFraction);
		queue.addCleanup(cleanupFilteredFractions);
#endif

		queue.setup();

		std::cout << std::setprecision(15);

		runRecord* d_recordPointer;
		unsigned long long int* d_recordNum;

		CHECK_CUDA(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

		CHECK_CUDA(cudaMalloc(&d_recordPointer, sizeof(runRecord)*THREADSATONCE));
		CHECK_CUDA(cudaMalloc(&d_recordNum, sizeof(unsigned long long int)));

		

		for (unsigned long long int i = 0; i < NUMRUNS; i++){



			CHECK_CUDA(cudaMemset(d_recordNum, 0, sizeof(unsigned long long int)));
			//14.13472514173469
			calculateGCF << <BLOCKS, TPB >> >(i*THREADSATONCE, d_recordPointer, d_recordNum, zeroes[z]);
			queue.clearQueue();

			unsigned long long int h_recordNum;
			CHECK_CUDA(cudaMemcpy(&h_recordNum, d_recordNum, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));


			runRecord* h_recordPointer = (runRecord*)malloc(h_recordNum*sizeof(runRecord));
			CHECK_CUDA(cudaMemcpy(h_recordPointer, d_recordPointer, h_recordNum*sizeof(runRecord), cudaMemcpyDeviceToHost));


			for (int j = 0; j < h_recordNum; j++){
				queue.addTask(h_recordPointer[j]);
			}

#ifdef PRINTPROGRESS
			if (i % 100 == 0) std::cout << "Offset: " << i*THREADSATONCE << ", Progress: " << ((double)i * 100) / NUMRUNS << "%" << std::endl;
#endif

			free(h_recordPointer);
		}


		queue.clearQueue();
		queue.finish();

		

		CHECK_CUDA(cudaFree(d_recordPointer));
		CHECK_CUDA(cudaFree(d_recordNum));

	}
	flushBestResultToFile();

	CHECK_CUDA(cudaEventRecord(endevent));

	float timeElapsed = 0;
	CHECK_CUDA(cudaDeviceSynchronize());

	CHECK_CUDA(cudaEventElapsedTime(&timeElapsed, startingevent, endevent));
	std::cout << "Time elapsed: " << std::setprecision(0) << timeElapsed / 1000 << " seconds" << std::endl;

	cudaDeviceReset();

	system("pause");
	return 0;
}