#pragma once

#include <string>

typedef struct params{
	int b0;
	int a;
	int b;
	int c;
	int ai;
	int bi;
	int ci;
} params;

std::string paramsToString(params par);