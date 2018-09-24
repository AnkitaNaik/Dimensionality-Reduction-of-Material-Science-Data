__author__ = 'Ankita'
'''
Code for constructing checkerboard microstructure and calculating the 2-point
auto and cross correlations using PyMKS package for this 2-D microstructure.

Input files : None
'''

import numpy as np
import matplotlib.pyplot as plt

from pymks import PrimitiveBasis
from pymks.datasets import make_checkerboard_microstructure
from pymks.stats import autocorrelate
from pymks.stats import crosscorrelate
from pymks.tools import draw_microstructures
from pymks.tools import draw_autocorrelations
from pymks.tools import draw_crosscorrelations


# Make checkerboard microstructure and observe it.
X = make_checkerboard_microstructure(square_size = 10, n_squares =6)
draw_microstructures(X)

# Define the basis for 2-point statistics
prim_basis = PrimitiveBasis(n_states=2)
X_ = prim_basis.discretize(X)

# Computing auto-correlatiions of the microstructure function and drawing the same
X_auto = autocorrelate(X,basis = PrimitiveBasis(n_states = 2), periodic_axes= (0,1))
correlations = [('white','white'),('black','black')]
draw_autocorrelations(X_auto[0],autocorrelations=correlations)

# Checking the volume fraction of both the phases i.e. (0,0) value of auto-correlations
centre =(X_auto.shape[1] +1)/2
print('Volume fraction of black phase', X_auto[0, centre, centre, 0])
print('Volume fraction of black phase', X_auto[0, centre, centre, 1])


# Computing the cross correlation of the microstructure function and drawing the same
X_cross = crosscorrelate(X,basis =PrimitiveBasis(n_states = 2), periodic_axes =(0,1))
correlations =[('black','white')]
draw_crosscorrelations(X_cross[0],crosscorrelations=correlations)

# Crosscorrelation at (0,0) should be zero
print('Center value : ', X_cross[0,centre,centre,0])