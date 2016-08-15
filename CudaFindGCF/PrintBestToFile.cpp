#include "PrintBestToFile.h"


std::vector<runRecord> records;
std::shared_ptr<CSVWriter> recordWriter;

bool lowerDelta(runRecord first, runRecord second){
	return first.delta < second.delta;
}


void setupBestToFile(){
	time_t currentTime;
	struct tm* timeInfo;

	time(&currentTime);
	timeInfo = localtime(&currentTime);

	std::stringstream filename;
	filename << "Best results on " << timeInfo->tm_year + 1900 << " " << timeInfo->tm_mon + 1 << " " <<
		timeInfo->tm_mday << " at " << timeInfo->tm_hour << " " << timeInfo->tm_min << " " << timeInfo->tm_sec << ".csv";


	recordWriter= std::shared_ptr<CSVWriter>(new CSVWriter(filename.str(), CSV_RUNRECORD));
}

void addResultBestToFile(runRecord r){
	records.push_back(r);
}

void markBestResult(){
	std::nth_element(records.begin(), records.begin(), records.end(), lowerDelta);
	recordWriter->write(records[0]);
	records.clear();
	flushBestResultToFile();
}

void flushBestResultToFile(){
	recordWriter->flush();
}


