#include "RunRecord.h"

std::string runRecordToString(runRecord rec){
	std::ostringstream os;
	os << paramsToString(rec.param) << std::setprecision(15) << ", result= " << rec.resultReal<<" + "<<rec.resultImag << "i, delta= " << rec.delta;
	return os.str();
}

void printRecord(runRecord rec){
	std::cout << runRecordToString(rec) << std::endl;
}