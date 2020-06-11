#############################################################
# The program will calculate the Stefan–Boltzmann constant #
# using scipy.integrate.quad to demonstrate its function   #
#############################################################

# import all requried libraries to compute Stefan–Boltzmann constant
import numpy as np

import math

from scipy.integrate import quad

# Define the constanta
BOLTZ =  1.380649e-23                                                           # Boltzmann constant

C = 2.99792e8                                                                   # speed of light

HBAR = 1.054571e-34                                                             # Planck constant

##############################################################
# The function to be integrated since quad expect a function #
# Limit of the function has been changed from infinity to 1  #
# using the conversion x = z/(1-z) and dx = dz/(1-z)^2       #
##############################################################
def f(z):

    numerator = np.power( 1/(1-z) ,2) * np.power(z/(1-z) ,3)                   # calculate the numerator alone for simplicity
    denominator = np.exp( z/(1-z) ) - 1                                            # calculate the denominator alone for simplicity
    integral = numerator / denominator
                                              # function after changing the infinity limit to 1
    return integral

constant= ( np.power(BOLTZ,4) / (4* np.pi**2 *np.power(C,2)* np.power(HBAR,3)) ) # to simplify, all constants have been taken out and calcuated outside of integral

result,error = quad(f,0,1)                                                      # to compute the integration

print("The Stefan-Boltzmann constant is", format(result*constant,'.3e'))        # the result upto 4 significiant digits
