import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad



def f(x,mass,amplitude):
  constant = (8*m)
  potential = np.power(x,6) + 2*np.power(x,2)
  potential_amplitude = np.power(amplitude,6) + 2*np.power(amplitude,2)
  
  return np.power(constant / (potential_amplitude - potential),0.5)
    
mass = 1
amplitude_list= np.arange(0.1,2.01,0.01)
period =[]
for i in amplitude_list:
    result,err = quad(f,0,i, args=(mass,i))
    period.append(result)  




print(period[-1], amplitude_list[-1])
plt.plot(period,amplitude_list)
plt.xlabel("Period of oscillation\n") 
plt.ylabel("Amplitude of oscillation\n")
plt.title("Anharmonic oscillation of potentionl of X^6+2X^2\n", color="red") 
plt.grid()
plt.show()

