
#include "kernel.cuh"

//Minimum and maximum parameters for search
const long long AMIN = -19L;
const long long AMAX = 19L;
const long long BMIN = -19L;
const long long BMAX = 19L;
const long long CMIN = -19L;
const long long CMAX = 19L;
const long long AIMIN = -19L;
const long long AIMAX = 19L;
const long long BIMIN = -19L;
const long long BIMAX = 19L;
const long long CIMIN = -19L;
const long long CIMAX = 19L;

#define FIRSTZERO 1
#define LASTZERO 100

#define NUMZEROS (1 + LASTZERO - FIRSTZERO)

#define MAXTERMS 15
#define MAXDELTA 1

#define PRINTRESULTS //Print results to console
#define PRINTTOFILE //Save results to file
#define PRINTPROGRESS //Print out progress messages periodically
//#define PROCESSFILTEREDFRACTIONS

#define RANGE(NUM) (NUM##MAX + 1 - NUM##MIN)

#define TPB 1024
#define BLOCKS 1024
#define THREADSATONCE (TPB*BLOCKS)

#define PERMUTATIONS (RANGE(A)*RANGE(B)*RANGE(C)*RANGE(AI)*RANGE(BI)*RANGE(CI))

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

__device__ double2 multiplyImag(double2 arg1, double2 arg2) {
	double2 ret;
	ret.x = ((arg1.x*arg2.x) - (arg1.y*arg2.y));
	ret.y = ((arg1.y*arg2.x) + (arg1.x*arg2.y));
	return ret;
}

__device__ double2 divideImag(double2 numerator, double2 denom){
	double2 ret;
	ret.x = ((numerator.x*denom.x) + (numerator.y*denom.y)) / ((denom.x*denom.x) + (denom.y*denom.y));
	ret.y = ((numerator.y*denom.x) - (numerator.x*denom.y)) / ((denom.x*denom.x) + (denom.y*denom.y));
	return ret;
}

__device__ double2 addImag(double2 arg1, double2 arg2){
	double2 ret;
	ret.x = arg1.x + arg2.x;
	ret.y = arg1.y + arg2.y;
	return ret;
}

__device__ double2 subImag(double2 arg1, double2 arg2){
	double2 ret;
	ret.x = arg1.x - arg2.x;
	ret.y = arg1.y - arg2.y;
	return ret;
}

__device__ double absImag(double2 arg) {
	return sqrt((arg.x*arg.x) + (arg.y*arg.y));
}

/**
  * Record the result of a run
  * @param param the starting parameters of the function
  * @param result the result to record
  * @param delta the difference between the result and the number
  * @param recordPointer a the array where the records are stored
  * @param recordNum a pointer to the current record index
  */
__device__ void recordRun(params param, double2 result, double delta, runRecord* recordPointer, unsigned long long int* recordNum){
	unsigned long long int address = atomicAdd(recordNum, 1);

	runRecord curRec;
	curRec.param = param;
	curRec.delta = delta;
	curRec.resultReal = result.x;
	curRec.resultImag = result.y;

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


	par.ci = workId % RANGE(CI) + CIMIN;
	workId /= RANGE(CI);
	
	par.bi = workId % RANGE(BI) + BIMIN;
	workId /= RANGE(BI);

	par.ai = workId % RANGE(AI) + AIMIN;
	workId /= RANGE(AI);

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
__device__ double2 calcFraction(params runPars){

	double2 hBefore1;
	hBefore1.x= 1;
	double2 hBefore2;
	hBefore2.x = 0;
	double2 kBefore1;
	kBefore1.x = 0;
	double2 kBefore2;
	kBefore2.x = 1;
	double2 aBefore;
	aBefore.x = 1;

	double2 convergent;
	convergent.x = 0;

	double2 paramA, paramB, paramC;
	paramA.x = runPars.a;
	paramA.y = runPars.ai;
	paramB.x = runPars.b;
	paramB.y = runPars.bi;
	paramC.x = runPars.c;
	paramC.y = runPars.ci;

	for (int iterNum = 0; iterNum < MAXTERMS; iterNum++){
		
		double2 iter2;
		iter2.x = iterNum;

		double2 newA = addImag(multiplyImag(paramA, iter2), paramB);
		double2 newB = paramC;

		double2 curNum = addImag(multiplyImag(newB, hBefore1), multiplyImag(aBefore, hBefore2));
		double2 curDen = addImag(multiplyImag(newB, kBefore1), multiplyImag(aBefore, kBefore2));

		convergent = divideImag( curNum , curDen);

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

	double2 convergent = calcFraction(runPars);

	double2 convergeToImag;
	convergeToImag.x = .5;
	convergeToImag.y = convergeTo;

	double delta = absImag(subImag( convergent , convergeToImag));

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