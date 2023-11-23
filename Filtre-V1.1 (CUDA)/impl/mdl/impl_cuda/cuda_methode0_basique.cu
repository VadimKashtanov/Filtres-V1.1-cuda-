#include "mdl.cuh"

/*	Filtres : Memoire constante
	Poids   : Memoire constante */

__device__ static float filtre(float * x, float * dif_x, float * f, float * dif_f) {
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

__device__ static float perceptron(float * x, float * p, uint _N) {
	float s = p[_N-1+1];
	FOR(0, i, _N) s += x[i]*p[i];
	return ACTIV(s);
};

__global__ void kerd_mdl(
	uint C,
	uint * EMA_INT_BLOQUE_ligne__d,
	uint  BLOQUES,
	uint * ST__d,
	//
	uint * DEPART_POIDS__d,
	float * f__d, float * p__d,
	float * dif_f__d,
	uint t0, uint t1, float * res__d,
	float * normalisee__d, float * dif_normalisee__d)
{
	uint t = t0 + (threadIdx.x + blockIdx.x * blockDim.x);
	//
	if (t < t1) {
		float r0[MAX_Y];
		float r1[MAX_Y];

		//	------------------------------------

		FOR(0, b, BLOQUES) {
			FOR(0, i, BLOQUE_ST) {
				uint ligne = EMA_INT_BLOQUE_ligne__d[b];
				r0[b*BLOQUE_ST + i] = filtre(
						normalisee__d + ligne*PRIXS*N_FLTR + t*N_FLTR,
					dif_normalisee__d + ligne*PRIXS*N_FLTR + t*N_FLTR,
					f__d     + b*BLOQUE_ST*N     + i*N,
					dif_f__d + b*BLOQUE_ST*(N-1) + i*(N-1)
				);
			//	printf("(%i)%f\n", b*BLOQUE_ST + i, r0[b*BLOQUE_ST + i]);

			};
		};
		
		//	Pensee
		FOR(1, c, C) {
			//printf("==========\n");
			FOR(0, y, ST__d[c]) {
				(c%2==0 ? r0 : r1)[y] = perceptron(
					(c%2==0 ? r1 : r0),
					p__d + DEPART_POIDS__d[c] + y*(ST__d[c-1]+1),
					ST__d[c-1]
				);
				//printf("(%i) %+f (%f)\n", y, (c%2==0 ? r0 : r1)[y], *(p__d + DEPART_POIDS__d[c] + y*(ST__d[c-1]+1) + ST__d[c-1]));
			};
		};

		res__d[t-t0] = ((C-1)%2==0 ? r0 : r1)[0];
	};
};

void cuda_mdt0_mdl_f_t0t1(Mdl_t * mdl, float * res, uint t0, uint t1) {
	float * res_d;
	CONTROLE_CUDA(cudaMalloc((void**)&res_d, sizeof(float)*(t1-t0)));
	CONTROLE_CUDA(cudaMemset(res_d, 0, sizeof(float)*(t1-t0)));

	//	--- Mdl_t ---
	kerd_mdl<<<dim3(KER_DIV((t1-t0), 256)), dim3(256)>>>(
		mdl->C,
		mdl->EMA_INT_BLOQUE_ligne__d,
		mdl->BLOQUES,
		mdl->ST__d,
		//
		mdl->DEPART_POIDS__d,
		mdl->f_d,
		mdl->p_d,
		mdl->dif_f_d,
		t0, t1, res_d,
		normalisee__d, dif_normalisee__d
	);
	ATTENDRE_KER_CUDA();

	CONTROLE_CUDA(cudaMemcpy(
		res,
		res_d,
		sizeof(float)*(t1-t0),
		cudaMemcpyDeviceToHost
	));

	//
	CONTROLE_CUDA(cudaFree(res_d));
};