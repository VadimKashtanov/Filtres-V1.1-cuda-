#pragma once

#include "macros.cuh"
#include "cuda.cuh"

//	=== Petites fonctions ===

float   rnd(       );
float signe(float x);

//	=== Fonctions Variadiques ===

char  **  va_allouer_chars(uint N, ...);
float ** va_allouer_floats(uint N, ...);

void va_chars_free(uint N, char  **    chars);
void va_uints_free(uint N, float ** flotants);

//  === Factorielles ===

extern const uint factoriel[13];
//Pourquoi [13] ?
//	Il a 12 factorielles possibles en i32 et 20 factorielles possibles en i64
//	Mmeme i128 en a que 34.
//Il est donc plus rentable de les ecrires toutes en liste.
//	FACT(n) est dans macros.h

//  === Math & Optimisation de Fonctions ===

float          ___exp(float x);

float        ___gauss(float x);
float      ___d_gauss(float x);

float   ___logistique(float x);    // Alternative : 2*(tanh(x)) + 0.5
float ___d_logistique(float x);

float         ___tanh(float x);    // Alternative : x / (0.5 + |x|)
float       ___d_tanh(float x);

//  === Utilitaire GRAPHIQUE ===

void             gnuplot(float *  arr, uint len, char * titre);
void matplotlib_matrices(float ** mat, char ** noms, uint N, uint X, uint Y);
void  matplotlib_courbes(float ** crb, char ** noms, uint L, uint N);

//	=== Allocation Memoire ===

uint* cpyuint(uint * arr, uint len);
float* allouer_flotants(uint nb);

//	=== Normalisation ===

void normer(float * arr, uint n);
void normer_moins_un_un(float * arr, uint n);
void prete(float * arr, uint n);
void lisser(float * arr, uint n, float A);

//	====== Lire fichier ======

float lire_flotant(char * fichier);
void ecrire_flotant(char * fichier, float a);
//
void ecrire_uint(char * fichier, uint a);
uint lire_uint(char * fichier);
void lire_N_uint(char * fichier, uint * _uint, uint _N);
void ecrire_N_uint(char * fichier, uint * _uint, uint _N);
//
void ecrire_char(char * fichier, char a);
char lire_char(char * fichier);

//	=========== Bruit de perlin ============

float bruit_de_perlin(float x, float y);
void perlin_carteXY(float * carte, uint X, uint Y);
