#include "CSVWriter.h"

CSVWriter::CSVWriter(std::string outfile):fileName(outfile),cache(){

	cache.push_back("A,B,C,D,E,F,G,Result,Delta\n");
	
}

void CSVWriter::write(runRecord r){

	std::stringstream strstrm;

	params par = r.param;
	
	strstrm << par.a << ",";
	strstrm << par.b << ",";
	strstrm << par.c << ",";
	strstrm << par.d << ",";
	strstrm << par.e << ",";
	strstrm << par.f << ",";
	strstrm << par.g << ",";

	strstrm << std::setprecision(16);

	strstrm << r.result << ",";
	strstrm << r.delta << std::endl;

	cache.push_back(strstrm.str());

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