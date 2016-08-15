#pragma once

#include <string>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "Params.h"

typedef struct runRecord{
	params param;
	double result;
	double delta;
} runRecord;

std::string runRecordToString(runRecord rec);
void printRecord(runRecord rec);