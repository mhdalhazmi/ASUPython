import numpy as np
from scipy.optimize import fsolve, leastsq
import matplotlib.pyplot as plt
import pandas as pd


def diode_current(v_diode):
    i_diode = i_s * ( np.exp( (Q*v_diode) / (ide*KB*T) ) - 1)
    return i_diode

def diode_voltage(v_diode, v_source):

    i_diode = diode_current(v_diode)
    err = (v_diode - v_source)/r + i_diode
    return err


i_s = 1e-9
ide = 1.7
r = 11000
T = 350
src_v_p1 = np.arange(.1,2.5,.1)
KB = 1.380648e-23
Q = 1.6021766208e-19
v_diode = np.zeros(len(src_v_p1))
i_diode = np.zeros(len(src_v_p1))
v_diode_guess = .1

for i in range((len(src_v_p1))):
        v_diode[i]= fsolve(diode_voltage,v_diode_guess,(src_v_p1[i]))
        v_diode_guess = v_diode[i]
        i_diode[i] = np.log10(diode_current(v_diode[i]))
        #i_diode[i] = diode_current(v_diode[i])

print(src_v_p1 )
print("\n")
print(v_diode)
print("\n")
print(i_diode)

plt.plot(src_v_p1,i_diode, "-b", label='Diode Voltage')
plt.plot(v_diode,i_diode, "-r", label='Diode Voltage')
#plt.plot(src_v_p1,i_diode, "--r" ,label='Source Voltage')
#plt.legend()
plt.grid(True)
plt.show()
