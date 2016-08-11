#pragma once

#include "CSVWriter.h"
#include "RunRecord.h"

void setupPrintToFile();
void processPrintToFile(runRecord r);
void cleanupPrintToFile();
void setPrintToFileZeroNum(int z);