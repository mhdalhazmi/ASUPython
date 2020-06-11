# nonlinear least squares example
# perhaps similar to david gay dn2fb
# author: allee@asu.edu updated by sdm

import numpy as np                    # math and arrays
from scipy.optimize import leastsq    # optimization function
import matplotlib.pyplot as plt       # so we can plot

N=1000                                # number of samples to generate

################################################################################
# Function to fit                                                              #
# Inputs:                                                                      #
#    kd - first value                                                          #
#    p0 - second value                                                         #
# Output:                                                                      #
#    value of the function                                                     #
################################################################################

def func(kd,p0):
    return 0.5*(-1-((2*p0)/kd) + np.sqrt(4*(p0/kd)+(((2*p0)/kd)-1)**2))

################################################################################
# Function to compute the difference between the actual and predited values    #
# Inputs:                                                                      #
#    kd_guess - the guess for the value of the first parameter                 #
#    p0 - the second parameter, which is known                                 #
#    actual - the sampled value                                                #
# Output:                                                                      #
#    difference between the actual value and the value calculated with guess   #
################################################################################

def residuals(kd_guess,p0,actual):
    return actual - func(kd_guess,p0)

# Create a noisy signal based on a known value of kd of 3
# since random returns uniformly distributed [0,1), subtract .5 gives
# [-.5,.5) so we get +/- noise
kd=3.0                                                # the "known" value
p0 = np.linspace(0,10,N)                              # an array for p0
actual = func(kd,p0)+(np.random.random(N)-0.5)*0.1    # create the noisy signal

# now try to extract the the known value of kd by minimizing the residuals
kd_match,cov,infodict,mesg,ier = leastsq(residuals,5,args=(p0,actual),full_output=True)

print(' kd guess', kd_match)                   # this is the guess for kd
# print(cov)
print("actual kd was",kd)
# print(infodict)
# print(mesg)
#print(ier)

plt.plot(p0,actual)                            # plot the noisy signal
plt.plot(p0,func(kd_match,p0))                 # along with the estimate
plt.show()

resid = residuals(kd_match,p0,actual)          # calculate the residuals
plt.plot(p0,resid)                             # and plot them
plt.show()

