#include "main.cuh"

PAS_OPTIMISER()
static void titre_partie(char * str) {
	printf("========= %s =========\n", str);
};

PAS_OPTIMISER()
int main(int argc, char ** argv) {
	//	-- Init --
	srand(0);
	cudaSetDevice(0);

	titre_partie("Charger tout"); charger_tout();
    //titre_partie("   Verif f  ");      verif_f();
	titre_partie("Performances"); performances();
	titre_partie("  Verif df  ");     verif_df();

	//===============
	titre_partie("  Programme Generale  ");

	//	-- Fin --
	liberer_tout();
};
