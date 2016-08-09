#pragma once

#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <list>
#include <iterator>

#include "RunRecord.h"
#include "Params.h"
#include "TTRunRecord.h"

static const int CSV_RUNRECORD = 1;
static const int CSV_TTRECORD = 2;

class CSVWriter{
public:
	CSVWriter(std::string fileName, int recordType);
	void write(runRecord record);
	void write(TTrunRecord record);
	void write(std::string str);
	void flush();
	~CSVWriter();

private:
	std::string fileName;
	std::list<std::string> cache;
	
};