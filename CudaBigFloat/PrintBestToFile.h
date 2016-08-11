#pragma once
#include "RunRecord.h"
#include <vector>
#include "CSVWriter.h"
#include <memory>
#include <algorithm>


	void setupBestToFile();
	void addResultBestToFile(runRecord r);
	void markBestResult();
	void flushBestResultToFile();

