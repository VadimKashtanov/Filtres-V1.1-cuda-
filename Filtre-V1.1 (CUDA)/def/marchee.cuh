#pragma once

#define DEBBUG false

#include "etc.cuh"

#define PRIXS 54273	//53170 * NB_DIFF_EMA * 32-bits = 17014400 bits = 17014.4 Ko = 17 Mo
#define L 2  		//u += u*f*levier*(p[i+L]/p[i]-1)

#define N_FLTR 8
#define BLOQUE_ST 16	//cmb de filtres dans un bloque
//	8*32 = 256
//	256*6 = 1536 = max noyaux par multi-processeur

#define MAX_INTERVALLE 1000

#define DEPART (N_FLTR*MAX_INTERVALLE)
#if (DEBBUG == false)
	#define FIN (PRIXS-L-1)
#else
	#define FIN (DEPART+1)
#endif

typedef struct {
	uint  ligne;
	uint    ema;
	uint interv;
} ema_int;

#define EMA_INTS 11
/*	     ema,   int
	0|     1      1
	1|     2      2
	2|     4      4
	3|     6      6
	4|    10     10
	5|    20     20
	6|    50     50
	7|   100    100
	8|   200    200
	9|   500    500
   10| 1'000  1'000
*/
extern ema_int ema_ints[EMA_INTS];

/*	Note : dans `normalisee` et `dif_normalisee`
les intervalles sont deja calculee. Donc tout
ce qui est avant DEPART n'est pas initialisee.
*/

extern float          prixs[PRIXS];
extern float            ema[EMA_INTS * PRIXS];
extern float     normalisee[EMA_INTS * PRIXS * N_FLTR];
extern float dif_normalisee[EMA_INTS * PRIXS * N_FLTR];

extern float *          prixs__d;	//	nVidia
extern float *            ema__d;	//	nVidia
extern float *     normalisee__d;	//	nVidia
extern float * dif_normalisee__d;	//	nVidia

void      charger_les_prixs();
void calculer_ema_norm_diff();
void    charger_vram_nvidia();

void     liberer_cudamalloc();

//	----------------------------------------------	//
//	--------------- Environnement ----------------	//
//	----------------------------------------------	//

typedef struct  {
	//	Chance de muter
	uint f_QUANTITEE[2];	//	Quantitee de filtres a muter
	uint e_QUANTITEE;		//	Quantitee de ema_int a muter

	//	Taux d'heritage paternel
	float f_COEF[2];

	//	+ sng()*rand(PLUS[0], PLUS[1])
	uint   e_EMA_PLUS[2];
	uint e_INTER_PLUS[2];
} Env_t;
