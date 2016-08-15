#include "Params.h"

std::string paramsToString(params par){

	return  "A= " + std::to_string(par.a) +
		", B= " + std::to_string(par.b) + ", C= " + std::to_string(par.c) + ", D= " + std::to_string(par.d) + ", E= " + std::to_string(par.e)+", F= " + std::to_string(par.f) + ", G= " + std::to_string(par.g);

}