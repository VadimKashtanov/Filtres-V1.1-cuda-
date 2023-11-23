#include "mdl.cuh"

void cpu_mdt0_f_t0t1(Mdl_t * mdl, float * res, uint t0, uint t1) {
	uint t;
//#pragma omp parallel for private(t)  //(Incoherances et inexactitudes !)
	for (t=t0; t < t1; t++) {
		res[t-t0] = f(mdl, t);
	}
};
