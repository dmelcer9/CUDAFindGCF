#include "Params.h"

std::string paramsToString(params par){

	return  "A= " + std::to_string(par.a) + ", AI= " + std::to_string(par.ai) +
		", B= " + std::to_string(par.b) + ", BI= " + std::to_string(par.bi) +
		", C= " + std::to_string(par.c) + ", CI= " + std::to_string(par.ci);

}