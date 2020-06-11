# Example to find maximum efficiency temp of light bulb
# author: olhartin@asu.edu updated by sdm

import numpy as np                     # for math and arrays
from scipy.integrate import quad       # to integrate
import matplotlib.pyplot as plt        # to plot the curve

# Constants used in the calculations
LAMBDA1 = 430e-9                       # lower wavelength
LAMBDA2 = 750e-9                       # upper wavelength
BOLTZ = 1.38064852e-23                 # Boltzmann Constant J/K
C = 2.99792458e8                       # Speed of light in vacuum m/s
PLANCK = 6.626070040e-34               # Planck's Constant J s

# these coefficients are used in the effincandescent_bulb function
COEFF1 = PLANCK*C/(LAMBDA1*BOLTZ)      # so only compute once!
COEFF2 = PLANCK*C/(LAMBDA2*BOLTZ)      # so only compute once!
COEFF3 = 15.0 / ( np.pi**4.0 )         # so only compute once!

################################################################################
# A test function to find a max for... Clearly, the max is at x = 2.           #
# Input:                                                                       #
#    x - value at which to evaluate the function                               #
# Output:                                                                      #
#    the value of the function evaluated at x                                  #
################################################################################

def func(x):
    return(float(10.0-(x-2.0)**2))
    
################################################################################
# Function to be integrated for lightbulb efficiency                           #
# Input:                                                                       #
#    x - variable inside integrand                                             #
# Output:                                                                      #
#    function evaluated at x                                                   #
################################################################################

def expfun(x):
    return(x**3/(np.exp(x)-1.0))

################################################################################
# Function to perform the lightbulb efficiency integration                     #
# Input:                                                                       #
#    temp - temperature at which to do the calculation                         #
# Output:                                                                      #
#    the efficiency                                                            #
################################################################################

def effincandescent_bulb(temp):
    upperlimit = COEFF1/temp                     # calculate integration limits
    lowerlimit = COEFF2/temp
    res,err = quad(expfun,lowerlimit,upperlimit) # do the integration
    effic = COEFF3 * res                         # mult by constant out front
    return(effic)

################################################################################
# Function to implement the golden search                                      #
# Inputs:                                                                      #
#    func - the function for which a max is desired                            #
#    x1   - the current lower value                                            #
#    x2   - the next to lowest value                                           #
#    x3   - the next to highest value                                          #
#    x4   - the current upper value                                            #
#    tol  - the tolerance: quit if x1 and x4 are closer than tol               #
# Outputs:                                                                     #
#    x1   - the new lower value                                                #
#    x2   - the new next to lowest value                                       #
#    x3   - the new next to highest value                                      #
#    x4   - the new upper value                                                #
################################################################################

def goldsearch(func,x1,x2,x3,x4,tol):
    if (x4-x1>tol):                                         # not close yet?
        if (max(func(x2),func(x3))>max(func(x1),func(x4))): # a middle > outside
            if (func(x2)>func(x3)):   # if x2 is bigger than x3...
                x4 = x3                   # slide x4 down to x3
                x3 = x2                   # slide x3 down to x2
                x2 = (x1+x3)/2.0          # x2 is average of x1 and old x2
            else:                     # if x3 was bigger than x2...
                x1 = x2                   # slide x1 up to x2
                x2 = x3                   # slide x2 up to x3
                x3 = (x2+x4)/2.0          # x3 is average of x4 and old x3

            # either way, we search again with the new values
            x1,x2,x3,x4 = goldsearch(func,x1,x2,x3,x4,tol)

        #else:  # the max is outside the range of x2:x3 same as at x1 or x4
            #print(x1,x2,x3,x4,func(x1),func(x2),func(x3),func(x4))

    #else:  # we're close enough...
        #print(x1,x2,x3,x4,func(x1),func(x2),func(x3),func(x4))

    return(x1,x2,x3,x4)
    
# First, find the maximum in the practic function
practice_x = np.arange(0,3,0.1)         # select a range to plot and search
#print(practice_x)
practice_y = np.zeros(len(practice_x),float)  # create empty array to hold y
for i in range(0,len(practice_x)):            # fill the y array
    practice_y[i] = func(practice_x[i])
    #print(i,practice_x[i],func(practice_x[i]))

plt.plot(practice_x,practice_y)               # plot the function
plt.show()

x1,x2,x3,x4 = goldsearch(func,0,1,2,2.5,0.001)     # now find the max
print("max is at : ",max(x2,x3), "where it is f(x): ", func(max(x2,x3)),"\n\n")

# find the temperature of bulb with the peak efficiency
# first look at 300 K, where it is not very efficient
temp = 300
eff = effincandescent_bulb(temp)
print('efficiency ',eff, ' T ',temp)

# now do the search
T1,T2,T3,T4 = goldsearch(effincandescent_bulb,300,2500,7500,10000,0.001)
print("max is at : ",max(T2,T3), "where it is f(x): ", effincandescent_bulb(max(T2,T3)))

# plot the efficiency over a range of temperatures from 300 - 10000K
seeds = np.linspace(0,10,50,float)        # get 50 values from 0 to 10
temps = seeds * 1000 + 300                # then spread them out from 300 K to 10,300 K
Eff = []                                  # create an empty list to hold efficiencies
for a_temp in temps:                      # for each temperature...
    #print (T)
    Eff.append(effincandescent_bulb(a_temp))   # append the efficiency at that temperature

peak_temp = max(T2,T3)                    # this is the peak temperature and print results
print(' peak at temperature ', peak_temp)
print(' actual filament temperature 2000 to 3300 K')
print(' suggesting an efficiency from ',
        effincandescent_bulb(2000), ' to ',
        effincandescent_bulb(3300) )

plt.title(' Efficiency vs Temperature ')  # now create a plot
plt.plot(temps,Eff)
plt.grid()
plt.ylabel(' Efficiency ')
plt.xlabel(' Temperature K ')

# place arrow at peak:  plt.arrow(x,y,dx,dy) draws line
Effpeak = effincandescent_bulb(peak_temp)
plt.arrow(peak_temp,0,.1,Effpeak)

#  place arrow over range of operating temp of 2000 to 3300
Effoprange = effincandescent_bulb(3300)
plt.arrow(2000,0,1300,0)
plt.arrow(3300,0,0,Effoprange)
plt.show()
