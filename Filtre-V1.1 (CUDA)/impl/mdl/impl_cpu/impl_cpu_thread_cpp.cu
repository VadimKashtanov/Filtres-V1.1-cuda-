#include "mdl.cuh"

//#include <iostream>
#include <thread>

using namespace std;

//uint p[FIN-DEPART] = {0};

//PAS_OPTIMISER()
void task1(Mdl_t * mdl, uint t0, uint t1, uint de_t, uint a_t, float * res) {
	FOR(de_t, t, a_t) {
		if (t0+t < t1) {
			res[t] = f(mdl, t0+t);
			//printf("t=%i\n", t);
			//p[t] = 1;
		}
	}
};

void cpu_mdt1_f_t0t1(Mdl_t * mdl, float * res, uint t0, uint t1) {

	MSG("CPU thread ne donne plus les memes resultats");

#define BASSIN 16
	
	thread ths[BASSIN];

	uint T = (t1-t0);
	uint combien = (T-(T%BASSIN))/BASSIN + 1;//KER_DIV(T, BASSIN);
	FOR(0, j, BASSIN) {
		ths[j] = thread(
			task1,
			mdl, t0, t1,
			j*combien, (j+1)*combien,
			res
		);
	}
	FOR(0, j, BASSIN) {
		ths[j].join();
	}
	//uint s = 0;
	//FOR(0, i, FIN-DEPART) s += p[i];
	//printf("%i == %i\n", s, (t1-t0));
};
