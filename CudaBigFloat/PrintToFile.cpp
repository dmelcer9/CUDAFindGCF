#include "PrintToFile.h"

CSVWriter* writer = NULL;

void setupPrintToFile(){
	time_t currentTime;
	struct tm* timeInfo;

	time(&currentTime);
	timeInfo = localtime(&currentTime);

	std::stringstream filename;
	filename << "Result " << timeInfo->tm_year + 1900 << " " << timeInfo->tm_mon + 1 << " " <<
		timeInfo->tm_mday << " " << timeInfo->tm_hour << " " << timeInfo->tm_min << " " << timeInfo->tm_sec << ".csv";


	writer = new CSVWriter(filename.str());
}

void processPrintToFile(runRecord r){
	writer->write(r);
}

void cleanupPrintToFile(){
	writer->flush();
	delete writer;
}