# Example root finder with diode and resistor circuit
# Here, there two equations to solve simultaneously
# author: allee updated by sdm

from scipy.optimize import fsolve       # root finder
import numpy as np                      # for math

Vdd = 5        # supply voltage
R1 = 1e3       # various resister values
R2 = 4e3
R3 = 3e3
R4 = 2e3
Io = 3e-9      # diode saturation current
Vt = 0.05      # diode threshold voltage

################################################################################
# Function which has two equations to be optimized to be 0 for the two nodes   #
# in the circuit.                                                              #
# Input:                                                                       #
#    v - a list of two entries representing the voltages of the two nodes      #
# Output:                                                                      #
#    a, b - the results of apply the voltages to the equations - want 0!       #
################################################################################

def f(v):
    diode = Io*(np.exp((v[0]-v[1])/Vt)-1.0)      # current through the diode
    a = (v[0]-Vdd)/R1 + v[0]/R2 + diode          # solve node 1
    b = (v[1]-Vdd)/R3 + v[1]/R4 - diode          # solve node 2
    return [a,b]

root = fsolve(f,[4,2])   # returns the values for V1 and V2
print(root)
