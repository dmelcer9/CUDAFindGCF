#pragma once

#include <string>

typedef struct params{
	int b0;
	int a;
	int b;
	int c;
	int d;
	int e;
	int f;
	int g;
} params;

std::string paramsToString(params par);