#################################################################
# The program will calculate period of an anharmonic oscillator #
# given its potential function of X^6+2X^2 and the various      #
# amplitudes from 1-2 in 0.01 steps                             #
#################################################################


# import all requried libraries to compute period of oscillation
import numpy as np

from matplotlib import pyplot as plt

from scipy.integrate import quad

# Initialize constants
mass = 1                                    # mass of the particle
amplitude_list= np.arange(0.1,2.01,0.01)    # list of amplitudes
period =[]                                  # Initialize the period list

####################################################################
# The function to be integrated since quad expect a function       #
# The function will calculate the period for each given amplitude  #
# The function takes in three variables the variable of integation #
# the mass of the particle and the amplitude                       #
####################################################################

def f(x,mass,amplitude):

  constant = (8*mass)
  potential = np.power(x,6) + 2*np.power(x,2)
  potential_amplitude = np.power(amplitude,6) + 2*np.power(amplitude,2)

  return np.power(constant / (potential_amplitude - potential),0.5)

# Integrate the function for every amplitude in the list from 0 to that amplitude
for i in amplitude_list:

    result,err = quad(f,0,i, args=(mass,i))
    period.append(result)                   # save the calcualted period of oscillation


##print(period[-1], amplitude_list[-1])
plt.plot(period,amplitude_list)             # print the period verces amplitude
plt.xlabel("Period of oscillation\n")       # identify the X axis as the period of oscillation
plt.ylabel("Amplitude of oscillation\n")    # identify the Y axis as the Amplitude
plt.title("Period of anharmonic oscillator \nof potentionl of X^6+2X^2", color="red")    # give the figure a title and change its color to red
plt.grid()                                  # put a grid to facilitate  reading
plt.show()                                  # show the graph
