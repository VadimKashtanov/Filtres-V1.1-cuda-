#include "mdl.cuh"

void taille_mdl(Mdl_t * mdl) {
	printf("  sizeof(Mdl_t) ~= %3.3f Mo\n",
		(float)sizeof(float) * (mdl->FILTRES + mdl->POIDS + mdl->VARS + mdl->LOCDS) / 1e6
	);
};

void plume_mdl(Mdl_t * mdl) {
	printf("Mdl_t Filtres=%i C=%i (VARS=%i FILTRES=%i POIDS=%i LOCDS=%i)\n",
		mdl->FS, mdl->C,
		mdl->VARS, mdl->FILTRES, mdl->POIDS, mdl->LOCDS);
	printf(" 0| filtre [%3.i] DEPART_VARS=%i\n",
		mdl->FS,
		mdl->DEPART_VARS[0]
	);
	FOR(0, i, EMA_INTS)
		if (mdl->F_par_EMA_INT[i] != 0)
			printf("\t (%2.i) %4.i filtres en ema=%i intervalle=%i\n", ema_ints[i].ligne, mdl->F_par_EMA_INT[i], ema_ints[i].ema, ema_ints[i].interv);
	printf("EMA_INT_BLOQUE_ligne : "); FOR(0, i, mdl->BLOQUES) {printf("%i ", mdl->EMA_INT_BLOQUE_ligne[i]);};printf("\n");
	FOR(1, i, mdl->C) {
		printf("%2.i| pensee:dot1d(tanh) [%4.i]   DEPART_VARS=%i DEPART_POIDS=%i DEPART_LOCDS=%i  (poids=%i)\n",
			i,
			mdl->ST[i],
			mdl->DEPART_VARS[i],
			mdl->DEPART_POIDS[i],
			mdl->DEPART_LOCDS[i],

			(mdl->ST[i-1]+1)*mdl->ST[i]
		);
	}
};