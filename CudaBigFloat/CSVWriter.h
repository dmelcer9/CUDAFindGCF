#pragma once

#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <list>
#include <iterator>

#include "RunRecord.h"
#include "Params.h"

class CSVWriter{
public:
	CSVWriter(std::string fileName);
	void write(runRecord record);
	void flush();
	~CSVWriter();

private:
	std::string fileName;
	std::list<std::string> cache;
};