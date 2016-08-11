#include "PrintToFile.h"

namespace PTF{
	CSVWriter* writer = NULL;
}
using PTF::writer;

static int printToFileZeroNum = 1;

void setPrintToFileZeroNum(int z){
	printToFileZeroNum = z;
}

void setupPrintToFile(){
	time_t currentTime;
	struct tm* timeInfo;

	time(&currentTime);
	timeInfo = localtime(&currentTime);

	

	std::stringstream filename;
	filename << "Result GPU Zero#"<<printToFileZeroNum<<" on " << timeInfo->tm_year + 1900 << " " << timeInfo->tm_mon + 1 << " " <<
		timeInfo->tm_mday << " at " << timeInfo->tm_hour << " " << timeInfo->tm_min << " " << timeInfo->tm_sec << ".csv";


	writer = new CSVWriter(filename.str(),CSV_RUNRECORD);

	printToFileZeroNum++;
}

void processPrintToFile(runRecord r){
	writer->write(r);
}

void cleanupPrintToFile(){
	writer->flush();
	delete writer;
}