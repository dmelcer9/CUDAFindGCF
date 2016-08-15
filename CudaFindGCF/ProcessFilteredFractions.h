#pragma once

#include <ttmath\ttmath.h>
#include "Params.h"
#include "RunRecord.h"
#include "CSVWriter.h"
#include "ProcessQueue.h"
#include "TTRunRecord.h"


void setupProcessFilteredFractions();
void processFraction(runRecord rec);
void cleanupFilteredFractions();