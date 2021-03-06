#include "CSVWriter.h"

CSVWriter::CSVWriter(std::string outfile, int type):fileName(outfile),cache(){

	if(type == CSV_RUNRECORD) cache.push_back("b0,A,B,C,D,E,F,G,Result,Delta\n");
	else if (type == CSV_TTRECORD) cache.push_back("A,B,C,D,E,F,G,GPU Result, GPU Delta, CPU Result, CPU Delta\n");

}

std::stringstream paramsToCSV(params par){
	std::stringstream strstrm;
	strstrm << par.b0 << ",";

	strstrm << par.a << ",";
	strstrm << par.b << ",";
	strstrm << par.c << ",";
	strstrm << par.d << ",";
	strstrm << par.e << ",";
	strstrm << par.f << ",";
	strstrm << par.g << ",";

	return strstrm;
}

void CSVWriter::write(runRecord r){

	std::stringstream strstrm = paramsToCSV(r.param);

	strstrm << std::setprecision(16);

	strstrm << r.result << ",";
	strstrm << r.delta << std::endl;

	write(strstrm.str());
	
}

void CSVWriter::write(TTrunRecord r){
	std::stringstream strstrm = paramsToCSV(r.rec.param);

	strstrm << r.rec.result << ",";
	strstrm << r.rec.delta << ",";

	
	strstrm << r.result << ",";
	strstrm << r.delta << std::endl;

	write(strstrm.str());
}

void CSVWriter::write(std::string s){
	cache.push_back(s);

	if (cache.size() > 10000) flush();
}

void CSVWriter::flush(){
	std::ofstream file(fileName, std::ofstream::app);
	for (std::list<std::string>::const_iterator l = cache.begin(); l != cache.end(); l++){
		file << *l;
	}

	cache.clear();
}

CSVWriter::~CSVWriter(){
	flush();
}