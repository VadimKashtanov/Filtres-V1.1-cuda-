#! /usr/bin/python3

def normaliser(lst):
	_max = max(lst)
	_min = min(lst)
	return [(i-_min)/(_max-_min) for i in lst], _max, _min

def lire_eurousd(fichier, col=2):
	with open(fichier, 'r') as co:
		lignes = co.read().replace('\n\n','').split('\n')[1:]
		return [float(ligne.split('\t')[col]) for ligne in lignes]

def lire_btcusdt(fichier, col=3):
	with open(fichier, 'r') as co:
		lignes = co.read().replace('\n\n','').split('\n')[2:-2]
		return [float(ligne.split(',')[col]) for ligne in lignes][::-1]
		
from sys import argv

#'EURUSD_H1.csv' : [1:] col=2  '\t'
#Binance_BTCUSDT_1h.csv : [2:] col=3  ','

#prixs = lire_eurousd(argv[1])
prixs = lire_btcusdt(argv[1])

import struct as st

#"prixs.bin"
with open(argv[2], "wb") as co:
	#co.write(st.pack( 'ffI', _max, _min, len(prixs) ))
	print(len(prixs))
	co.write(st.pack('I', len(prixs)))
	co.write(st.pack('f'*len(prixs), *prixs))
