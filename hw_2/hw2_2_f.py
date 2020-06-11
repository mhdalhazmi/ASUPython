######################################################
# The program will calculate total energy (W) given 
# off by a black body per unit area per second

import numpy as np
import math
from scipy.integrate import quad

BOLTZ =  1.380649e-23
C = 2.99792e8                                                       #Speed of light
HBAR = 1.054571e-34

def f(z):
    integral = np.power( (1/(1-z)) ,2) * np.power(z/(1-z),3) / ( np.exp(z/(1-z)) -1)
    return integral 

constant= ( np.power(BOLTZ,4) / (4* np.pi**2 *np.power(C,2)* np.power(HBAR,3)) )
result,error = quad(f,0,1)
print("The Stefan-Boltzmann constant is", format(result*constant,'.3e'))

