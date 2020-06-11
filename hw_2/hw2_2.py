
import numpy as np
import math
from scipy.integrate import quad

KB =  1.380649e-23
C = 2.99792e8                                                       #Speed of light
H = 1.054571e-34

#equation = x**3/(math.exp(x)-1)

def f(z):
    #equation = (1/(1-z)**2)*(np.power(x,3)/math.exp(x)-1)
    #stefan_boltzmann_constant= ((KB**4)/(4*sp.pi**2*np.power(C,2)*np.power(H,3)))
    #new = np.power(1/(1-z),2)* np.power(z/(1-z),3) / ( math.exp(z/(1-z)) -1)
    new = np.power(1/(1-z),2)* np.power(z/(1-z),3) / ( math.exp(z/(1-z)) -1)
    return new 


stefan_boltzmann_constant= ( np.power(KB,4) / (4* np.pi**2 *np.power(C,2)* np.power(H,3)) )
print(stefan_boltzmann_constant)
x,y = quad(f,0,0.998)
#print(quad(f,0,0.5))
print(format(x*stefan_boltzmann_constant,'.3e'))
# np.power(z,3)/((-1 + math.exp(z/(1 - z)))*np.poewr((1 - z)zyyy5) dz = 6.49394

