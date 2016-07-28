#pragma once

#include <string>

typedef struct params{
	int a0;
	int b0;
	int a;
	int b;
	int c;
	int d;
	int e;
} params;

std::string paramsToString(params par);