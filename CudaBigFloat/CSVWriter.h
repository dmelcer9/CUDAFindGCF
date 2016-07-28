#pragma once

#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <iomanip>
#include "kernel.cuh"
#include "RecordStructs.h"
#include <list>
#include <iterator>

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