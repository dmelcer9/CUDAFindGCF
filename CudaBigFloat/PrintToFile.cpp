#include "PrintToFile.h"

namespace PTF{
	CSVWriter* writer = NULL;
}
using PTF::writer;

static int zeroNum = 0;

void setupPrintToFile(){
	time_t currentTime;
	struct tm* timeInfo;

	time(&currentTime);
	timeInfo = localtime(&currentTime);

	zeroNum++;

	std::stringstream filename;
	filename << "Result GPU Zero#"<<zeroNum<<" on " << timeInfo->tm_year + 1900 << " " << timeInfo->tm_mon + 1 << " " <<
		timeInfo->tm_mday << " at " << timeInfo->tm_hour << " " << timeInfo->tm_min << " " << timeInfo->tm_sec << ".csv";


	writer = new CSVWriter(filename.str(),CSV_RUNRECORD);
}

void processPrintToFile(runRecord r){
	writer->write(r);
}

void cleanupPrintToFile(){
	writer->flush();
	delete writer;
}