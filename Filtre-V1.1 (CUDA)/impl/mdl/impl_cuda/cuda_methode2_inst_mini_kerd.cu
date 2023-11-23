#include "mdl.cuh"

/*
	Faire une version avec une somme __partagée__
	et donc (BLOQUE_ST, N_FLT) avec BLOQUE_ST*N_FLT==256

*/

__global__ static
void filtre__kerd(
	uint t0,
	const uint * __restrict__ EMA_INT_BLOQUE_ligne__d,
	const float * __restrict__ normalisee__d,
	const float * __restrict__ f, const float * __restrict__ dif_f,
	float * __restrict__ y, uint Y_MAX)
{
	//	<<<(T, BLOQUES), (BLOQUE_ST)>>> //BLOQUE_ST*N_FLT==256

#define bloque blockIdx.y
#define f_dans_bloque threadIdx.x

	uint t = blockIdx.x;

	//	__partage__
	__shared__ uint ligne;
	if (threadIdx.x == 0)
		ligne = EMA_INT_BLOQUE_ligne__d[bloque];

	//	__partage__
	__shared__ float sh_x[N_FLTR];
	if (f_dans_bloque < N_FLTR)
		sh_x[f_dans_bloque] = normalisee__d[ligne*PRIXS*N_FLTR + (t0+t)*N_FLTR + f_dans_bloque];

	//	fonction de filtre
	float s = 0, d = 0;
	FOR(0, i, N_FLTR-1) {
		s += sqrtf(1 + fabs(     sh_x[i]        -   f[bloque*BLOQUE_ST*N_FLTR + f_dans_bloque*N_FLTR + i]  ));
		d += powf((1 + fabs((sh_x[i+1]-sh_x[i]) - dif_f[bloque*BLOQUE_ST*(N_FLTR-1) + f_dans_bloque*(N_FLTR-1) + i])), 2);
	};
	s += sqrtf(1 + fabs(sh_x[N_FLTR-1] - f[bloque*BLOQUE_ST*N_FLTR + f_dans_bloque*N_FLTR + N_FLTR-1]));

	//	s = a + b + c ...

	s = s/8-1;
	d = d/7-1;

	y[t*Y_MAX + bloque*BLOQUE_ST + f_dans_bloque] = 2*expf(-s*s -d*d)-1;
};

//=================================================

#define K__dot1d__X 16
#define K__dot1d__T  4

__global__ static
void dot1d__kerd(
	uint X,
	uint T, uint Y,
	uint Y_MAX,
	//
	uint c,
	const uint * DEPART_POIDS__d,
	//
	const float * __restrict__ x__d, const float * __restrict__ p__d, float * __restrict__ y__d)
{
	//	<<<(KERD(X,16),KERD(T,4)), (16,4)>>>
	const uint y = threadIdx.x + blockIdx.x * blockDim.x;
	const uint t = threadIdx.y + blockIdx.y * blockDim.y;

	if ( (t<T) && (y<Y) ) {
		float s = p__d[DEPART_POIDS__d[c] + y*(X+1) + X-1+1];
		FOR(0, i, X) {
			s += x__d[t*Y_MAX + i]*p__d[DEPART_POIDS__d[c] + y*(X+1) + i];
		}
		y__d[t*Y_MAX + y] = ACTIV(s);
	};
}

#define BLOCK_SIZE 32

__global__ void matvec_kernel(
	const float * __restrict__ dA,
	const float * __restrict__ dx,
	float * __restrict__ dy,
	const uint nRows,
	const uint nCols)
{
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float x_shared[BLOCK_SIZE];

    T y_val = 0.0;

    #pragma unroll
    for (unsigned int m = 0; m < ((nCols + BLOCK_SIZE - 1)/ BLOCK_SIZE); ++m)
    {
        if ((m * BLOCK_SIZE + threadIdx.x) <  nCols) x_shared[threadIdx.x] = dx[threadIdx.x + m * BLOCK_SIZE];
        else                                         x_shared[threadIdx.x] = 0.f;
        __syncthreads();

        #pragma unroll
        for (unsigned int e = 0; e < BLOCK_SIZE; ++e) {
            // --- Column-major ordering - faster
            y_val += dA[tid + (e + BLOCK_SIZE * m) * nRows] * x_shared[e];
            // --- Row-major ordering - slower
            //y_val += dA[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
        }

        __syncthreads();
    }

    if (tid < nRows) dy[tid] = y_val;

}

/*
__global__ static
void dot1d__mini_somme__kerd(

	)
{
	//	<<<(), ()>>>
	//	<<<(KERD(Y,16),KERD(T,8),KERD(X,16)), (16,8,8)>>>

	//	<<<(16Y, 8X, )>>>
	uint y = threadIdx.x + blockIdx.x * blockDim.x;
	uint t = threadIdx.y + blockIdx.y * blockDim.y;

	__shared__ float X16[16];
	if ()

	if ( (t<T) && (y<Y) ) {
		float s = p__d[DEPART_POIDS__d[c] + y*(X+1) + X-1+1];
		FOR(0, i, X) {
			s += x__d[t*Y_MAX + i]*p__d[DEPART_POIDS__d[c] + y*(X+1) + i];
		}
		y__d[t*Y_MAX + y] = ACTIV(s);
	};
};

__global__ static
void dot1d__ACTIV__kerd(
	)
{
	//	<<<(KERD(T, 32), KERD(X, 8)),  (32,8)>>>
	//	<<<(KERD(X,16),KERD(T,4)), (16,4)>>>
	uint y = threadIdx.x + blockIdx.x * blockDim.x;
	uint t = threadIdx.y + blockIdx.y * blockDim.y;

	if ( (t<T) && (y<Y) ) {
		y__d[t*Y_MAX + y] = ACTIV(y__d[t*Y_MAX + y]);
	};
};*/

//	==============================================================

static __global__ void enregistrer_les_resultats__kerd(
	float * res_d, float * y__d,
	uint Y_MAX, uint T)
{
	uint t = threadIdx.x + blockIdx.x * blockDim.x;

	if (t < T) {
		res_d[t] = y__d[t*Y_MAX + 0];
	};
};

void cuda_mdt2_mdl_f_t0t1(Mdl_t * mdl, float * res, uint t0, uint t1) {
	float * res_d;
	CONTROLE_CUDA(cudaMalloc((void**)&res_d, sizeof(float)*(t1-t0)));
	CONTROLE_CUDA(cudaMemset(res_d, 0, sizeof(float)*(t1-t0)));

	//	--- Allocation des r0, r1 ---
	uint T = t1 - t0;
	float * r0__d, * r1__d;
	CONTROLE_CUDA(cudaMalloc((void**)&r0__d, sizeof(float) * MAX_Y * T));
	CONTROLE_CUDA(cudaMalloc((void**)&r1__d, sizeof(float) * MAX_Y * T));

	//	--- Mdl_t ---

	//	--- Filtres ---
	filtre__kerd<<<dim3(T, mdl->BLOQUES), dim3(BLOQUE_ST)>>>(
		t0,
		mdl->EMA_INT_BLOQUE_ligne__d,
		normalisee__d,
		mdl->f_d, mdl->dif_f_d,
		r0__d, MAX_Y
	);
	/*filtre__mini_kerd__kerd<<<dim3(T, mdl->BLOQUES), dim3(BLOQUE_ST, N_FLTR)>>>(
		t0,
		mdl->EMA_INT_BLOQUE_ligne__d,
		normalisee__d,
		mdl->f_d, mdl->dif_f_d,
		r0__d, MAX_Y
	);*/
	/*filtre__mini_kerd__kerd____v_s_puis_d<<<dim3(T, mdl->BLOQUES), dim3(BLOQUE_ST, N_FLTR)>>>(
		t0,
		mdl->EMA_INT_BLOQUE_ligne__d,
		normalisee__d,
		mdl->f_d, mdl->dif_f_d,
		r0__d, MAX_Y
	);*/
	ATTENDRE_KER_CUDA();

	//	--- Pensee Perceptronnale ---
	FOR(1, c, mdl->C) {
		dim3 grille(KER_DIV(mdl->ST[c], K__dot1d__X), KER_DIV(T, K__dot1d__T));
		dim3 noyaux(K__dot1d__X, K__dot1d__T);
		dot1d__kerd<<<grille, noyaux>>>(
			mdl->ST[c-1],
			T, mdl->ST[c],
			MAX_Y,
			//
			c,
			mdl->DEPART_POIDS__d,
			//
			(c%2==0 ? r1__d : r0__d),	//x__d
			mdl->p_d,
			(c%2==0 ? r0__d : r1__d)	//y__d
		);
		ATTENDRE_KER_CUDA();
	};

	//	--- Ecrire dans res ---
	enregistrer_les_resultats__kerd<<<dim3(KER_DIV(T,256)), dim3(256)>>>(
		res_d, ((mdl->C-1)%2==0 ? r0__d : r1__d),
		MAX_Y, T);
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