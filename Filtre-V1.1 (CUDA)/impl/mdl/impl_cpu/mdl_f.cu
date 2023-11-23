#include "mdl.cuh"

static float filtre(float * x, float * dif_x, float * f, float * dif_f) {
	float s = 0, d = 0;
	FOR(0, i, N-1) {
		s += sqrtf(1 + fabs(  x[i]   -   f[i]  ));
		d += powf((1 + fabs(dif_x[i] - dif_f[i])), 2);
	};
	s += sqrtf(1 + fabs(x[N-1] - f[N-1]));

	s = s/8-1;
	d = d/7-1;

	return 2*expf(-s*s -d*d)-1;
};

static float perceptron(float * x, float * p, uint _N) {
	float s = p[_N-1+1];
	FOR(0, i, _N) s += x[i]*p[i];
	return ACTIV(s);
};

float f(Mdl_t * mdl, uint t) {
	//	--- Filtres ---
	FOR(0, b, mdl->BLOQUES) {
		FOR(0, i, BLOQUE_ST) {
			uint ligne = mdl->EMA_INT_BLOQUE_ligne[b];
			mdl->y[b*BLOQUE_ST + i] = filtre(
					normalisee + ligne*PRIXS*N_FLTR + t*N_FLTR,
				dif_normalisee + ligne*PRIXS*N_FLTR + t*N_FLTR,
				mdl->f     + b*BLOQUE_ST*N     + i*N,
				mdl->dif_f + b*BLOQUE_ST*(N-1) + i*(N-1)
			);
		};
	};
	
	//	Pensee
	FOR(1, c, mdl->C) {
		FOR(0, y, mdl->ST[c]) {
			mdl->y[mdl->DEPART_VARS[c] + y] = perceptron(
				mdl->y + mdl->DEPART_VARS[c-1],
				mdl->p + mdl->DEPART_POIDS[c] + y*(mdl->ST[c-1]+1),
				mdl->ST[c-1]
			);
		};
	};
	return mdl->y[mdl->VARS-1];
};