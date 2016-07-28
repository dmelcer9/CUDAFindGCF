#include "printRecord.h"

 std::string paramsToString(params par){

	return "A0= " + std::to_string(par.a0) + ", B0= " + std::to_string(par.b0) + ", A= " + std::to_string(par.a) +
		", B= " + std::to_string(par.b) + ", C= " + std::to_string(par.c) + ", D= " + std::to_string(par.d) + ", E= " + std::to_string(par.e);

}

 std::string runRecordToString(runRecord rec){
	std::ostringstream os;
	os << paramsToString(rec.param) << std::setprecision(15) << ", result= " << rec.result << ", delta= " << rec.delta;
	return os.str();
}

 void printRecord(runRecord rec){
	std::cout << runRecordToString(rec) << std::endl;
}