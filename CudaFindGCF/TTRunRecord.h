#pragma once

#include <ttmath\ttmath.h>
#include "Params.h"

typedef struct TTrunRecord{
	ttmath::Big<1,50> result;
	ttmath::Big<1,50> delta;
	runRecord rec;
}TTrunRecord;