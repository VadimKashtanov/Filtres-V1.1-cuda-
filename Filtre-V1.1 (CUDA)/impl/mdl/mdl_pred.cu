#include "mdl.cuh"

float pred_mdl(Mdl_t * mdl, int METHODE, uint t0, uint t1) {
	prep_mdl(mdl);
	float p1, p0;
	float res[FIN-DEPART] = {0};
	//
	if (METHODE < 0) {
		if (METHODE == -1)      cpu_mdt0_f_t0t1(mdl, res, t0, t1);
		else if (METHODE == -2) cpu_mdt1_f_t0t1(mdl, res, t0, t1);
		else 			        ERR("Pas de Methode %i.", METHODE);
	} else {
		if (METHODE == 0)       cuda_mdt0_mdl_f_t0t1(mdl, res, t0, t1);
		else if (METHODE == 1)  cuda_mdt1_mdl_f_t0t1(mdl, res, t0, t1);
		else if (METHODE == 2)  cuda_mdt2_mdl_f_t0t1(mdl, res, t0, t1);
		else                    ERR("Pas de Methode %i.", METHODE);
	}
	//
	float pred = 0;
	FOR(t0, t, t1) {
		p1 = prixs[t+L];
		p0 = prixs[ t ];
		pred += (float)(signe(res[t-DEPART]) == signe(p1/p0-1));
		if (t < t0+10) printf("%+f ", res[t-DEPART]);
	}
	printf("\n");
	return pred / (float)(FIN-DEPART);
};

void comportement(Mdl_t * mdl) {
#define T 25
	float arr[T * mdl->VARS];
	//
	FOR(0, t, T) {
		f(mdl, DEPART+t);
		memcpy(arr + t*mdl->VARS, mdl->y, sizeof(float) * mdl->VARS);
	}
	//
	FOR(0, i, mdl->VARS) {
		printf("%3.i| ", i);
		FOR(0, t, T) {
			printf("%+3.3f ", arr[t*mdl->VARS + i]);
		}
		printf("\n");
	}
};