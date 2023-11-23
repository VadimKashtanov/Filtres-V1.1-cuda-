#pragma once

#include "marchee.cuh"

#define ACTIV(x) tanh(x)

#define MAX_Y 2048

#define N N_FLTR

//	Filtres<FS> -> Pensee0<C1> -> ... -> 1
//	Bloques :
//		- il y a N bloques de `BLOQUE_ST` filtres chaqu'un
//		- comme ca on peut facilement __shared__

#define COUCHES (mdl->C)

typedef struct {
	//	/!\ FS == ST[0]
	uint F_par_EMA_INT[EMA_INTS];
	uint FS;
	uint BLOQUES;			//	FS % 32
	uint * EMA_INT_BLOQUE_ligne;	//	ligne de chaque bloque
	uint * EMA_INT_BLOQUE_ligne__d;

	//	Pensee
	uint C;
	uint * ST, * ST__d;

	//	Totaux
	uint FILTRES, POIDS, VARS, LOCDS;
	uint * DEPART_POIDS, * DEPART_VARS, * DEPART_LOCDS;
	uint * DEPART_POIDS__d;

	//	[CPU] : BLOQUES + PENSEE
	float * f;
	float * p;
	float * y;
	float * locd;

	//	[GPU] : BLOQUES + PENSEE
	float * f_d;
	float * p_d;

	//	[GPU] : grad_BLOQUES + grad_PENSEE
	float * dp_d;
	float * dy_d;

	//	[CPU & GPU] : Espace Optimisation
	float * dif_f;
	float * dif_f_d;
} Mdl_t;

//	--- Allocation & Gestion Memoire ---
/*  F_par_EMA_INT[EMA_INTS] : cmb de filtres pour chaque LIGNE. %BLOQUE_ST==0 (donc minimum 32 filtres par ligne)
    PENSEE[PENSEE_DIM]      : y de chaque couche (y compris la 0'eme des filtres)
*/
Mdl_t * cree_mdl(
	uint * F_par_EMA_INT,
	uint PENSEE_DIM, uint * PENSEE);
void mdl_liberer(Mdl_t * mdl);
void    prep_mdl(Mdl_t * mdl);
void  reinit_mdl(Mdl_t * mdl);

//	--- Transferts ---
void gpu_vers_cpu(Mdl_t * mdl);

//	--- Plume ---
void   taille_mdl(Mdl_t * mdl);
void    plume_mdl(Mdl_t * mdl);
//
void comportement(Mdl_t * mdl);

//	--- CPU ---
float               f(Mdl_t * mdl, uint t);								//	[Model Exacte] Fonction exacte
void  cpu_mdt0_f_t0t1(Mdl_t * mdl, float * res, uint t0, uint t1);		//	[ Méthode -1 ] Simple, pas de fils
void  cpu_mdt1_f_t0t1(Mdl_t * mdl, float * res, uint t0, uint t1);		//	[ Méthode -2 ] mdt-1 mais avec les fils

//	--- GPU ---
void cuda_mdt0_mdl_f_t0t1(Mdl_t * mdl, float * res, uint t0, uint t1);	//	[Méthode 0] Basique copier pareil que CPU
void cuda_mdt1_mdl_f_t0t1(Mdl_t * mdl, float * res, uint t0, uint t1);	//	[Méthode 1] Instructions 
void cuda_mdt2_mdl_f_t0t1(Mdl_t * mdl, float * res, uint t0, uint t1);	//	[Méthode 2] plus de __shared__ et de mini bloques 
																		//		et plus de s = a+b+c+d..n
																		//		et plus de memoire constante
//	--- Pred / Gain / Gain_Moyen ---
float pred_mdl(Mdl_t * mdl, int METHODE, uint t0, uint t1);
/* Methode :
    -2 : 2nd  méthode CPU
	-1 : 1ere méthode CPU
	 0 : 0eme méthode cuda
	 1 : 1ere
	 2 : 2nd
	 3 : 3eme
*/