#include "main.cuh"

//	================================================================

PAS_OPTIMISER()
void charger() {
	charger_les_prixs();
	calculer_ema_norm_diff();
	charger_vram_nvidia();
}

PAS_OPTIMISER()
void liberer_tout() {
	liberer_cudamalloc();
}

PAS_OPTIMISER()
void charger_tout() {
	printf("Chargement des prixs : ");
	MESURER(charger());

	printf("           prixs = %3.3f Mo\n", ((float)sizeof(float)*PRIXS)                     / 1e6);
	printf("             ema = %3.3f Mo\n", ((float)sizeof(float)*EMA_INTS*PRIXS)            / 1e6);
	printf("      normalisee = %3.3f Mo\n", ((float)sizeof(float)*EMA_INTS*N_FLTR*PRIXS)     / 1e6);
	printf("  dif_normalisee = %3.3f Mo\n", ((float)sizeof(float)*EMA_INTS*(N_FLTR-1)*PRIXS) / 1e6);
};

//	================================================================

PAS_OPTIMISER()
void verif_f() {
	ASSERT(BLOQUE_ST == 4);	//Pour que ca soit pas trop complexe a lire
	uint F_par_EMA_INT[EMA_INTS] = {
		4,	// 0
		0,	// 1
		8,	// 2
		0,	// 3
		0,	// 4
		0,	// 5
		0,	// 6
		0,	// 7
		0,	// 8
		0,	// 9
		0	//10
	};	//	Somme = 160
//#define C 3
	uint C = 3;
	uint ST[C] = {12, 3, 1};

	Mdl_t * mdl = cree_mdl(
		F_par_EMA_INT,
		C, ST
	);
	plume_mdl(mdl);

	uint depart = DEPART + 56;

	printf("Filtres : ");
	FOR(0, i, mdl->FILTRES) printf("(%i)%+f ", i, mdl->f[i]);
	printf("\n");

	printf("Poids : ");
	FOR(0, i, mdl->POIDS) printf("(%i)%f ", i, mdl->p[i]);
	printf("\n");

	FOR(0, e, EMA_INTS) {
		printf("ema_int=%i\n\tnormalisee     : ", e);
		FOR(0, i, N_FLTR) printf("(%i)%+f ", i, normalisee[e*PRIXS*N_FLTR + depart*N_FLTR + i]);
		printf("\n\tdif_normalisee : ");
		FOR(0, i, N_FLTR-1) printf("(%i)%+f ", i, dif_normalisee[e*PRIXS*N_FLTR + depart*N_FLTR + i]);
		printf("\n");
	}

	f(mdl, depart);
	printf("Vars = \n");
	FOR(0, i, mdl->VARS) printf("%2.i | %+f\n", i, mdl->y[i]);
};

//	================================================================

PAS_OPTIMISER()
static void plumer_perf(int methode, float pred, float secondes, float mdl_par_secondes, float ref) {
	printf(
		"pred=\033[94m%+f\033[0m temps ~= %+3.3f [\033[93m%f\033[0m mdl/s",
		pred, secondes,
		mdl_par_secondes
	);

	if (methode != -1) {
		printf(" mdt=%i %s (x%3.3f plus rapide)]\n",
			methode,
			(methode >= 0 ? "\033[96mCUDA \033[0m" : "\033[95mC/C++\033[0m"),
			mdl_par_secondes / ref
		);
	} else {
		printf(" fonction reference]\n");
	}
};

PAS_OPTIMISER()
void performances() {
	uint F_par_EMA_INT[EMA_INTS] = {
		256,	// 0
		32,		// 1
		64,		// 2
		32,		// 3
		32,		// 4
		64,		// 5
		32,		// 6
		0,		// 7
		0,		// 8
		0,		// 9
		0		//10
	};	//	Somme = 256
#define C 8
	uint ST[C] = {512, 512, 128, 64, 32, 8, 4, 1};

	Mdl_t * mdl = cree_mdl(
		F_par_EMA_INT,
		C, ST
	);
	taille_mdl(mdl);
	plume_mdl(mdl);

	//	-----------------------------------------------
	INIT_CHRONO(a)
	float pred;
	float perf;
	float autre_perf;
	float temps_sec;

	//	---------- CPU methode classique ----------
	DEPART_CHRONO(a)
	pred = pred_mdl(mdl, -1, DEPART, FIN);
	temps_sec = VALEUR_CHRONO(a);
	perf = 1 / temps_sec;
	plumer_perf(-1, pred, temps_sec, perf, 0);

	//	---------- Autres Méthodes CPU   ----------
	/*int METHODES_CPU[0] = {};//{-2};
	FOR(0, i, 0) {
		DEPART_CHRONO(a);
		pred = pred_mdl(mdl, METHODES_CPU[i], DEPART, FIN);
		temps_sec = VALEUR_CHRONO(a);
		autre_perf = 1 / temps_sec;
		plumer_perf(METHODES_CPU[i], pred, temps_sec, autre_perf, perf);
	}*/

	//	----------      Méthodes CUDA   ----------
	int METHODES_GPU[3] = {0, 1, 2};
	FOR(0, i, 3) {
		DEPART_CHRONO(a)
		pred = pred_mdl(mdl, METHODES_GPU[i], DEPART, FIN);
		temps_sec = VALEUR_CHRONO(a);
		autre_perf = 1 / temps_sec;
		plumer_perf(METHODES_GPU[i], pred, temps_sec, autre_perf, perf);
	}

	//	Fin
	//comportement(mdl);
	//plume_mdl(mdl);
	mdl_liberer(mdl);
};

//	==============================================================

PAS_OPTIMISER()
void verif_df() {
	
};