#include "mdl.cuh"

PAS_OPTIMISER()
Mdl_t * cree_mdl(
	uint * F_par_EMA_INT,
	uint C, uint * ST)
{
	ASSERT(ST[C-1] == 1);

	Mdl_t * mdl = (Mdl_t*)malloc(sizeof(Mdl_t));

	//
	memcpy(mdl->F_par_EMA_INT, F_par_EMA_INT, sizeof(uint) * EMA_INTS);
	mdl->FS = 0;
	FOR(0, i, EMA_INTS) mdl->FS += F_par_EMA_INT[i];
	//
	ASSERT(mdl->FS % BLOQUE_ST == 0);
	//
	mdl->BLOQUES = mdl->FS / BLOQUE_ST;
	mdl->EMA_INT_BLOQUE_ligne = (uint*)malloc(sizeof(uint) * mdl->BLOQUES);
	//
	uint k=0;
	FOR(0, i, EMA_INTS) {
		FOR(0, j, F_par_EMA_INT[i]/BLOQUE_ST) mdl->EMA_INT_BLOQUE_ligne[k++] = i;
	}

	ASSERT(mdl->FS == ST[0]);

	//
	mdl->C  = C;
	mdl->ST = cpyuint(ST, C);

	//
	mdl->FILTRES = mdl->FS * N;
	mdl->POIDS = 0;
	mdl->VARS  = mdl->FS;
	mdl->LOCDS = 0;

	//	---
	mdl->DEPART_POIDS = (uint*)malloc(sizeof(uint) * COUCHES);
	mdl->DEPART_VARS  = (uint*)malloc(sizeof(uint) * COUCHES);
	mdl->DEPART_LOCDS = (uint*)malloc(sizeof(uint) * COUCHES);
	mdl->DEPART_POIDS[0] = 0;
	mdl->DEPART_VARS [0] = 0;
	mdl->DEPART_LOCDS[0] = 0;

	//	Instructions : Pensee (dot1d)
	FOR(1, i, C) {
		ASSERT(ST[i] <= MAX_Y);

		mdl->DEPART_VARS [i] = mdl->VARS ;
		mdl->DEPART_POIDS[i] = mdl->POIDS;
		mdl->DEPART_LOCDS[i] = mdl->LOCDS;
		//
		mdl->VARS  += ST[i];
		mdl->POIDS += (ST[i-1]+1)*ST[i];
		mdl->LOCDS += ST[i];
	};

	//	======= Allocation ========
	mdl->f    = (float*)malloc(sizeof(float) * mdl->FILTRES );
	mdl->p    = (float*)malloc(sizeof(float) * mdl->POIDS   );
	mdl->y    = (float*)malloc(sizeof(float) * mdl->VARS    );
	mdl->locd = (float*)malloc(sizeof(float) * mdl->LOCDS   );

	CONTROLE_CUDA(cudaMalloc((void**)&mdl->f_d,    sizeof(float) * mdl->FILTRES ));
	CONTROLE_CUDA(cudaMalloc((void**)&mdl->p_d,    sizeof(float) * mdl->POIDS   ));

	CONTROLE_CUDA(cudaMalloc((void**)&mdl->dp_d, sizeof(float) * mdl->POIDS ));
	CONTROLE_CUDA(cudaMalloc((void**)&mdl->dy_d, sizeof(float) * mdl->VARS  ));

	mdl->dif_f = (float*)malloc(sizeof(float) * mdl->FS * (N-1));
	CONTROLE_CUDA(cudaMalloc((void**)&mdl->dif_f_d, sizeof(float) * mdl->FS * (N-1)));

	FOR(0, i, mdl->FS) 	  prete(mdl->f + i*N_FLTR, N_FLTR);
	FOR(0, i, mdl->POIDS) mdl->p[i] = (2*rnd()-1) * 0.5;

	//	Qlq uint pour cuda
	CONTROLE_CUDA(cudaMalloc((void**)&mdl->DEPART_POIDS__d,         sizeof(uint) * mdl->C      ));
	CONTROLE_CUDA(cudaMalloc((void**)&mdl->ST__d,                   sizeof(uint) * mdl->C      ));
	CONTROLE_CUDA(cudaMalloc((void**)&mdl->EMA_INT_BLOQUE_ligne__d, sizeof(uint) * mdl->BLOQUES));
	//
	CONTROLE_CUDA(cudaMemcpy(mdl->DEPART_POIDS__d,         mdl->DEPART_POIDS,         sizeof(uint)*mdl->C,       cudaMemcpyHostToDevice));
	CONTROLE_CUDA(cudaMemcpy(mdl->ST__d, 	               mdl->ST,                   sizeof(uint)*mdl->C,       cudaMemcpyHostToDevice));
	CONTROLE_CUDA(cudaMemcpy(mdl->EMA_INT_BLOQUE_ligne__d, mdl->EMA_INT_BLOQUE_ligne, sizeof(uint)*mdl->BLOQUES, cudaMemcpyHostToDevice));

	prep_mdl(mdl);

	return mdl;
};

void mdl_liberer(Mdl_t * mdl) {
	free(mdl->ST);
	free(mdl->EMA_INT_BLOQUE_ligne);
	//
	free(mdl->DEPART_POIDS);
	free(mdl->DEPART_VARS);
	free(mdl->DEPART_LOCDS);
	//
	free(mdl->dif_f);
	//
	free(mdl->f);
	free(mdl->p);
	free(mdl->y);
	free(mdl->locd);
	//
	CONTROLE_CUDA(cudaFree(mdl->f_d));
	CONTROLE_CUDA(cudaFree(mdl->p_d));
	CONTROLE_CUDA(cudaFree(mdl->dp_d));
	CONTROLE_CUDA(cudaFree(mdl->dy_d));
	CONTROLE_CUDA(cudaFree(mdl->dif_f_d));
	//
	CONTROLE_CUDA(cudaFree(mdl->ST__d));
	CONTROLE_CUDA(cudaFree(mdl->DEPART_POIDS__d));
	CONTROLE_CUDA(cudaFree(mdl->EMA_INT_BLOQUE_ligne__d));
};

void prep_mdl(Mdl_t * mdl) {
	memset(mdl->y, 0, sizeof(float) * mdl->VARS);
	memset(mdl->locd, 0, sizeof(float) * mdl->LOCDS);

	CONTROLE_CUDA(cudaMemcpy(mdl->p_d, mdl->p, sizeof(float)*mdl->POIDS, cudaMemcpyHostToDevice));
	CONTROLE_CUDA(cudaMemcpy(mdl->f_d, mdl->f, sizeof(float)*mdl->FILTRES, cudaMemcpyHostToDevice));

	CONTROLE_CUDA(cudaMemset(mdl->dp_d,    0, sizeof(float) * mdl->POIDS));
	CONTROLE_CUDA(cudaMemset(mdl->dy_d,    0, sizeof(float) * mdl->VARS));
	
	FOR(0, i, mdl->FS) {
		FOR(0, j, N-1) {
			mdl->dif_f[i*(N-1) + j] = mdl->f[i*N+j+1]-mdl->f[i*N+j];
		}
	}
	cudaMemcpy(mdl->dif_f_d, mdl->dif_f, sizeof(float)*mdl->FS*(N-1), cudaMemcpyHostToDevice);
};

void gpu_vers_cpu(Mdl_t * mdl) {
	CONTROLE_CUDA(cudaMemcpy(mdl->p, mdl->p_d, sizeof(float)*mdl->POIDS, cudaMemcpyDeviceToHost));
	CONTROLE_CUDA(cudaMemcpy(mdl->f, mdl->f_d, sizeof(float)*mdl->FILTRES, cudaMemcpyDeviceToHost));
};

void reinit_mdl(Mdl_t * mdl) {
	prep_mdl(mdl);
};