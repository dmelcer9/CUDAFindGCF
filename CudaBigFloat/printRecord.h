#pragma once

#include "RecordStructs.h"
#include <string>
#include <iomanip>
#include <iostream>
#include <sstream>

 std::string paramsToString(params par);
 std::string runRecordToString(runRecord rec);

 void printRecord(runRecord rec);