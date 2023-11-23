#pragma once

#define pi 3.14159265358979f

//============================= NvProfil ==========================

#define __NVPROFIL__ true

#if __NVPROFIL__ == true
	#define START_NVPROFIL cudaProfilerStart();
	#define END_NVPROFIL cudaProfilerStop();
#else
	#define START_NVPROFIL MSG("__NVPROFIL__ est définie a false");
	#define END_NVPROFIL 0;
#endif

//	======== Memoire constante =======
#define MAX_CONST_FLOTANTS 1 << 14
__constant__ float const_mem[MAX_CONST_FLOTANTS];

//#define MAX_PARTAGE_FLOTANTS (uint)(49152 / 4)

//	Memoire partagee dynamique inter-ker
//<<<grid,block,shared_amount>>> amount will be alloc there.
extern __shared__ float partage_dynamique[];	//no needs for size

//	Aide au deboguage
#define CONTROLE_CUDA(call) do { cudaError_t err = call; if (err != cudaSuccess)ERR("Erreure Cuda : %s", cudaGetErrorString(err));} while(0);

//	70 divise en thread de 32 == 3 block (dont le derniers qui n'utilisera pas tous les threads). 70 == 2*32 + 6 == 3*32 - 26
#define KER_DIV(TOTAL, div) (((TOTAL - TOTAL%div)/div)+1)
//#define KER_DIV(X,div) (X<=div ? 1 : ( X%div==0 ? X/div : ((X-X%div)/div+1)))

//	Attendre la fin du kernel
#define ATTENDRE_KER_CUDA() do{ CONTROLE_CUDA(cudaDeviceSynchronize()); CONTROLE_CUDA(cudaPeekAtLastError()); } while(0);
