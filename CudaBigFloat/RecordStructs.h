#pragma once

typedef struct params{
	int a0;
	int b0;
	int a;
	int b;
	int c;
	int d;
	int e;
} params;

typedef struct runRecord{
	params param;
	double result;
	double delta;
} runRecord;