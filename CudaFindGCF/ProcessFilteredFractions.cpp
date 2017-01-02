#include "ProcessFilteredFractions.h"
/*
namespace PFF{
	CSVWriter* writer = NULL;
}
using PFF::writer;

ttmath::Big<1,50> zetaZero = "14.13472514173469379045725198356247027078425711569924317568556746014996342980925676494901039317156101277920297154879743676614269146988225458250536323944713778041338123720597054962195586586020055556672583601073700205410982661507542780517442591306254481978651072304938725629738325774203952157256748093321400349904680343462673144209203773854871413781735639699536542811307968053149168852906782082298049264338666734623320787587617920056048680543568014444246510655975686659032286865105448594432062407272703209427452221304874872092412385141835146054279015244783835425453344004487936806761697300819000731393854983736215013045167266838920039176285123212854220523969133425832275335164060169763527563759695376749203361272092599917304270756830879511844534891800863008264832516911271068291052375961797743181517071354531677549515382893784903644709727019948485532209253574357909226125247736595518016975233461213973160053541259267474557258778014726098308089786007125320875093959979666067537838121489190886497727755442065653205240";

void setupProcessFilteredFractions(){
	time_t currentTime;
	struct tm* timeInfo;

	time(&currentTime);
	timeInfo = localtime(&currentTime);

	std::stringstream filename;
	filename << "Result CPU " << timeInfo->tm_year + 1900 << " " << timeInfo->tm_mon + 1 << " " <<
		timeInfo->tm_mday << " at " << timeInfo->tm_hour << " " << timeInfo->tm_min << " " << timeInfo->tm_sec << ".csv";


	writer = new CSVWriter(filename.str(),CSV_TTRECORD);
}

TTrunRecord calcFracResult(runRecord par, ttmath::Big<1,20> convergeTo){

	params runPars = par.param;

	ttmath::Big<1,50> hBefore1 = 1;
	ttmath::Big<1,50> hBefore2 = 0;
	ttmath::Big<1,50> kBefore1 = 0;
	ttmath::Big<1,50> kBefore2 = 1;
	ttmath::Big<1,50> aBefore = 1;

	ttmath::Big<1,50> convergent = 0;

	for (int iterNum = 0; iterNum < 1000; iterNum++){
		ttmath::Big<1,50> newA = runPars.a*iterNum*iterNum*iterNum + runPars.b*iterNum*iterNum + runPars.c*iterNum + runPars.d;
		ttmath::Big<1,50> newB = (iterNum == 0) ? runPars.b0 : runPars.e*iterNum*iterNum + runPars.f*iterNum + runPars.g;

		ttmath::Big<1,50> curNum = newB*hBefore1 + aBefore*hBefore2;
		ttmath::Big<1,50> curDen = newB*kBefore1 + aBefore*kBefore2;

		convergent = curNum / curDen;

		hBefore2 = hBefore1;
		hBefore1 = curNum;

		kBefore2 = kBefore1;
		kBefore1 = curDen;

		aBefore = newA;
	}

	TTrunRecord result;
	result.rec = par;
	result.result = convergent;
	result.delta = ttmath::Abs(convergent - convergeTo);
	return result;

	
}


void processFraction(runRecord r){
	writer->write(calcFracResult(r,zetaZero));
}

void cleanupFilteredFractions(){
	writer->flush();
	delete writer;
}*/