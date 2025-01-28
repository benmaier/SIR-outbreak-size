import numpy as np
import matplotlib.pyplot as pl
import pathlib
from pathlib import Path
import csv
import contact_matrices.countries as countries

#import matplotlib as mpl
#mpl.rcParams['lines.markersize'] = 5
#mpl.rcParams['lines.markeredgewidth'] = 1

matrix_root = './contact_matrices/exported_contact_data/{0}.csv'
demog_root = './contact_matrices/exported_contact_data/{0}_demography.csv'

methods = ['assortative', 'disassort.', 'disassort. symmetric']

def load_matrix(code,offset=0):
    i = offset
    fn = matrix_root.format(code.upper())
    matrix = np.loadtxt(fn, skiprows=1,delimiter=',')
    return matrix

def load_demography(code,offset=0):
    i = offset
    fn = demog_root.format(code.upper())
    with open(fn,'r') as f:
        reader = csv.DictReader(f)
        N = [ row['population'] for row in reader ]
    return np.array(N, dtype=float)

def load_contact_data(code,offset=0):
    gamma = load_matrix(code,offset)
    N = load_demography(code,offset)
    return gamma, N

