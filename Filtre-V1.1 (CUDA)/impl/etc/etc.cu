#include "etc.cuh"

double secondes() {
	struct timespec now;
	timespec_get(&now, TIME_UTC);
	return 1000.0*(((int64_t) now.tv_sec) * 1000 + ((int64_t) now.tv_nsec) / 1000000);
};

const uint factoriel[13] = { 1, 1, 2, 6, 24, 120, 720, 5040, 40320, 
                                    362880, 3628800, 39916800, 479001600 };

float rnd() {
	return (float)(rand()%100000)/100000.0;	//rand()%100 pour avoire 1.0
};

float ___exp(float x)  // cubic spline approximation
{
    union { float f; int i; } reinterpreter;

    reinterpreter.i = (int)(12102203.0f*x) + 127*(1 << 23);
    int m = (reinterpreter.i >> 7) & 0xFFFF;  // copy mantissa
    // empirical values for small maximum relative error (8.34e-5):
    reinterpreter.i +=
         ((((((((1277*m) >> 14) + 14825)*m) >> 14) - 79749)*m) >> 11) - 626;
    return reinterpreter.f;
}

float ___gauss(float x) {return ___exp(-x*x);};
float ___d_gauss(float x) {return -2*x*___gauss(x);};

float ___logistique(float x) {return 1.0/(1.0+___exp(-x));};
float ___d_logistique(float x) {return ___logistique(x)*(1 - ___logistique(x));};

float ___tanh(float x) {return tanhf(x);};
float ___d_tanh(float x) {return 1 - powf(___tanh(x), 2);};


float signe(float x) {return (x>=0 ? 1:-1);};

void gnuplot(float * arr, uint len, char * titre) {
	char buff[200];
	//
	FILE * fp = fopen("gnuplot_dat.dat", "w");
	//
	for (uint i=0; i < len; i++) {
		snprintf(buff, 100, "%i ", i);
		fputs(buff, fp);
		//
		snprintf(buff, 100, "%f\n", arr[i]);
		fputs(buff, fp);
	}
	fclose(fp);
	//
	snprintf(
		buff,
		200,
		"gnuplot -p -e \"set title \'%s\'; plot 'gnuplot_dat.dat' w lp\"",
		titre);
	//
	assert(!system(buff));
	//
	assert(!system("rm gnuplot_dat.dat"));
};

uint* cpyuint(uint * arr, uint len) {
	uint * ret = (uint*)malloc(sizeof(uint) * len);
	memcpy(ret, arr, sizeof(uint) * len);
	return ret;
}

float* allouer_flotants(uint nb) {
	return (float*)malloc(sizeof(float) * nb);
}

uint u_max(uint * x, uint len) {
	uint _max=x[0];
	FOR(1,i,len) {
		if (x[i] > _max)
			_max = x[i];
	};
	return _max;
}

PAS_OPTIMISER()
void normer(float * arr, uint n) {
	float max=arr[0], min=arr[0];
	FOR(1, i, n) {
		if (arr[i] > max) max = arr[i];
		if (arr[i] < min) min = arr[i];
		//printf("%f ", arr[i]);
	}
	//printf("\n");
	FOR(0, i , n) {
	//	printf("%f %f %f\n", arr[i], min, max);
		arr[i] = (arr[i]-min)/(max-min);
		assert(arr[i]>=0);
	}
}

PAS_OPTIMISER()
void normer_moins_un_un(float * arr, uint n) {
	normer(arr, n);
	FOR(0, i, n) arr[i] = 2*arr[i] - 1;
};

PAS_OPTIMISER()
void prete(float * arr, uint n) {
	float s = rnd()-.5;
	float d = rnd()-.5;
	FOR(0, i, n) {
		if (i%1==0) s += rnd()-.5;
		if (i%2==0) d += rnd()-.5;
		arr[i] = s/2 + d/2;
	}
	normer(arr, n);
};

PAS_OPTIMISER()
void lisser(float * arr, uint n, float A) {
	FOR(0, i, n) {
		arr[i] = A*roundf(arr[i]/A);
	}
};

/*void __5050(float * arr, uint n) {
	assert(n == 49);
};*/

PAS_OPTIMISER()
float lire_flotant(char * fichier) {
	FILE * fp = fopen(fichier, "rb");
	SI_EXISTE(fp, fichier);
	//
	int fd = fileno(fp);
	flock(fd, LOCK_EX);
	//
	float res;
	(void)!fread(&res, sizeof(float), 1, fp);
	//
	flock(fd, LOCK_UN);
	fclose(fp);
	return res;
}

PAS_OPTIMISER()
void ecrire_flotant(char * fichier, float a) {
	FILE * fp = fopen(fichier, "wb");
	//
	int fd = fileno(fp);
	flock(fd, LOCK_EX);
	//
	(void)!fwrite(&a, sizeof(float), 1, fp);
	flock(fd, LOCK_UN);
	fclose(fp);
};

PAS_OPTIMISER()
void ecrire_uint(char * fichier, uint a) {
	FILE * fp = fopen(fichier, "wb");
	//
	int fd = fileno(fp);
	flock(fd, LOCK_EX);
	//
	(void)!fwrite(&a, sizeof(uint), 1, fp);
	flock(fd, LOCK_UN);
	fclose(fp);
};

PAS_OPTIMISER()
uint lire_uint(char * fichier) {
	FILE * fp = fopen(fichier, "rb");
	SI_EXISTE(fp, fichier);
	//
	int fd = fileno(fp);
	flock(fd, LOCK_EX);
	//
	uint res;
	(void)!fread(&res, sizeof(uint), 1, fp);
	flock(fd, LOCK_UN);
	fclose(fp);
	return res;
};

PAS_OPTIMISER()
void lire_N_uint(char * fichier, uint * _uint, uint _N) {
	FILE * fp = fopen(fichier, "rb");
	SI_EXISTE(fp, fichier);
	//
	int fd = fileno(fp);
	flock(fd, LOCK_EX);
	//
	(void)!fread(_uint, sizeof(uint), _N, fp);
	flock(fd, LOCK_UN);
	fclose(fp);
};

PAS_OPTIMISER()
void ecrire_N_uint(char * fichier, uint * _uint, uint _N) {
	FILE * fp = fopen(fichier, "wb");
	//
	int fd = fileno(fp);
	flock(fd, LOCK_EX);
	//
	(void)!fwrite(_uint, sizeof(uint), _N, fp);
	flock(fd, LOCK_UN);
	fclose(fp);
};

//	-- char
PAS_OPTIMISER()
void ecrire_char(char * fichier, char a) {
	FILE * fp = fopen(fichier, "wb");
	//
	int fd = fileno(fp);
	flock(fd, LOCK_EX);
	//
	(void)!fwrite(&a, sizeof(char), 1, fp);
	flock(fd, LOCK_UN);
	fclose(fp);
};

PAS_OPTIMISER()
char lire_char(char * fichier) {
	FILE * fp = fopen(fichier, "rb");
	SI_EXISTE(fp, fichier);
	//
	int fd = fileno(fp);
	flock(fd, LOCK_EX);
	//
	char res;
	(void)!fread(&res, sizeof(char), 1, fp);
	flock(fd, LOCK_UN);
	fclose(fp);
	return res;
};

//	=== Fonctions Variadiques ===

char **   va_allouer_chars(uint N, ...) {
	va_list ptr;
    va_start(ptr, N);

	char ** ret = (char**)malloc(sizeof(char*) * N);
	FOR(0, i, N) {
		ret[i] = va_arg(ptr, char*);
	}

	va_end(ptr);
	return ret;
};

float ** va_allouer_floats(uint N, ...) {
	va_list ptr;
    va_start(ptr, N);

	float ** ret = (float**)malloc(sizeof(float*) * N);
	FOR(0, i, N) {
		ret[i] = va_arg(ptr, float*);
	}

	va_end(ptr);
	return ret;
};

void va_chars_free(uint N, char  **    chars) {
	FOR(0, i, N) free(chars[i]);
	free(chars);
};

void va_uints_free(uint N, float ** flotants) {
	FOR(0, i, N) free(flotants[i]);
	free(flotants);
};

//	=== Matplotlib ===

#define  MATPLOTLIB_COURBES 0
#define MATPLOTLIB_MATRICES 1

static char* demande_tmpt_matplotlib() {
	char * nom_fichier = (char*)malloc(sizeof(char) * 50);
	uint nb = 0;
	FOR(0, x, 100) {
		snprintf(nom_fichier, 50, "tmpt/tempt%i", x);
		FILE * fp = fopen(nom_fichier, "rb");
		if (fp == 0) {
			nb = x + 1;
			break;
		} else {
			fclose(fp);
		}
	}
	if (nb == 0) ERR("Plus de 100 fichier ont ete cree");
	snprintf(nom_fichier, 50, "tmpt/tempt%i", nb-1);
	return nom_fichier;
};

static void lancer_matplotlib(char * fichier, uint mode) {
/*
	Modes :
		0| Courbes avec noms
		1| Matrices avec noms
*/
	char commande[100];
	snprintf(commande, 100, "python3 python_matplotlib.py %s %i", fichier, mode);
	printf("python_matplotlib.py sortie avec %i\n",
		system(commande)
	);
};

void matplotlib_matrices(float ** mat, char ** noms, uint N, uint X, uint Y) {
	char * fichier = demande_tmpt_matplotlib();
	//
	FILE * fp = fopen(fichier, "wb");
	FOPEN_LOCK(fp, fichier);
	//
	FWRITE(&N, sizeof(uint), 1, fp);
	FWRITE(&X, sizeof(uint), 1, fp);
	FWRITE(&Y, sizeof(uint), 1, fp);
	FOR(0, i, N) FWRITE(mat[i], sizeof(float), X * Y, fp);
	FOR(0, i, N) {
		uint _strlen = strlen(noms[i]);
		FWRITE(&_strlen, sizeof(uint), 1,       fp);
		FWRITE(noms[i],   sizeof(char), _strlen, fp);
	}
	//
	FCLOSE_UNCLOCK(fp);
	//
	lancer_matplotlib(fichier, MATPLOTLIB_MATRICES);
	free(fichier);
};

void  matplotlib_courbes(float ** crb, char ** noms, uint L, uint N) {
	char * fichier = demande_tmpt_matplotlib();
	//
	FILE * fp = fopen(fichier, "wb");
	FOPEN_LOCK(fp, fichier);
	//
	FWRITE(&N, sizeof(uint), 1, fp);
	FWRITE(&L, sizeof(uint), 1, fp);
	FOR(0, i, N) FWRITE(crb[i], sizeof(float), L, fp);
	FOR(0, i, N) {
		uint _strlen = strlen(noms[i]);
		FWRITE(&_strlen, sizeof(uint), 1,       fp);
		FWRITE(noms[i],   sizeof(char), _strlen, fp);
	}
	//
	FCLOSE_UNCLOCK(fp);
	//
	lancer_matplotlib(fichier, MATPLOTLIB_COURBES);
	free(fichier);
};