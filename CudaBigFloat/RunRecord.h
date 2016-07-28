#pragma once

#include <string>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "Params.h"

typedef struct runRecord{
	params param;
	float result;
	float delta;
} runRecord;

std::string runRecordToString(runRecord rec);
void printRecord(runRecord rec);