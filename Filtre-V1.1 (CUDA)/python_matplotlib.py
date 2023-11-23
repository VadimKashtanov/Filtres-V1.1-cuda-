#! /usr/bin/python3

import struct as st

def lire_uint(text, N):
	_bin = list(st.unpack('I'*N, text[:N*st.calcsize('I')]))
	return _bin, text[N*st.calcsize('I'):]

def lire_flotants(text, N):
	_bin = list(st.unpack('f'*N, text[:N*st.calcsize('f')]))
	return _bin, text[N*st.calcsize('f'):]

def lire_char(text, N):
	_bin = text[:N*st.calcsize('f')].decode()
	return _bin, text[N:]

from sys import argv
from os import system
import matplotlib.pyplot as plt

_, fichier, mode = argv

if int(mode) == 0:
	#	--- Courbes ---
	with open(fichier, "rb") as co:
		text = co.read()
		(N, L), text = lire_uint(text, 2)

		courbes = []
		for i in range(N):
			crb, text = lire_flotants(text, L)
			courbes += [crb]

		noms = []
		for i in range(N):
			(l,), text = lire_uint(text, 1)
			nom, text = lire_char(text, l)
			noms += [nom]

		C = int(N**.5)+1
	
		fig, axs = plt.subplots(C, C)
	
		for i in range(N):
			x = i % C
			y = int((i - i%C)/C)
			axs[y][x].plot(courbes[i], label=noms[i])
			
		plt.legend()
		plt.show()

if int(mode) == 1:
	#	--- Matrices ---
	with open(fichier, "rb") as co:
		text = co.read()
		(N, X,Y), text = lire_uint(text, 3)
		mats = []
		for i in range(N):
			mat, text = lire_flotants(text, X*Y)
			mats += [mat]	
		noms = []
		for i in range(N):
			(l,), text = lire_uint(text, 1)
			nom, text = lire_char(text, l)
			noms += [nom]
		
		C = int(N**.5)+1

		fig, axs = plt.subplots(C, C)
		
		for i in range(N):
			x = i % C
			y = int((i - i%C)/C)
			axs[y][x].imshow(
				[[mat[i][y][x] for x in range(X)] for y in range(Y)],
				label=noms[i]
			)
	
		plt.legend()
		plt.show()

system(f"rm {fichier}")