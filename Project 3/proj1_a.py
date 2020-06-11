import numpy as np
from scipy.optimize import fsolve, leastsq
import matplotlib.pyplot as plt
import pandas as pd


def Diode_Current(v_d):

    i_d = nodal(v_s,v_d,r)
    a = i_s*( np.exp( (v_s-i_d*r)*Q / (n*k*T) ) - 1 )-i_d

    return a

def nodal(v_s,v_d,r):
    i_d = (v_s-v_d)/r
    return i_d


i_s = 1e-9
n = 1.7
r = 11000
T = 350
src_v_p1 = np.arange(.1,2.6,.1)
k = 1.380648e-23
Q = 1.6021766208e-19
v_d = []
i_diode = []

guess = 4

for i in src_v_p1:
    v_s = i
    v_d.append(fsolve(Diode_Current,guess))
    #guess = v_d[i-1]

for i in range(len(v_d)):
    i_diode.append(nodal(src_v_p1[i],v_d[i],r))



for i in range(len(src_v_p1)):
    err = v_d[i]/r - src_v_p1[i]/r + i_diode[i]


    #print(err)



plt.plot(v_d,i_diode, "-b", label='Diode Voltage')
plt.plot(src_v_p1,i_diode, "--r" ,label='Source Voltage')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()



###############################
def Diode_Current_2(v_d,v_s,r,n,T,Is):

    i_d = nodal(v_s,v_d,r)
    Vt = n*k*T/Q
    a = Is*(np.exp(v_d/Vt)-1)-i_d

    return a


def compute_diode_current(Vd,n,T,Is):
    Vt = n*k*T/Q
    return Is*(np.exp(Vd/Vt)-1)

#############
def opt_r(r_value,ide_value,phi_value,area,temp,src_v,meas_i):
    est_v   = np.zeros_like(src_v)       # an array to hold the diode voltages
    diode_i = np.zeros_like(src_v)       # an array to hold the diode currents
    prev_v = P1_VDD_STEP                 # an initial guess for the voltage
    # need to compute the reverse bias saturation current for this phi!
    is_value = area * temp * temp * np.exp(-phi_value * Q / ( k * temp ) )

    for index in range(len(src_v)):
        prev_v = fsolve(Diode_Current_2,prev_v,
				(src_v[index],r_value,ide_value,temp,is_value),
                                xtol=1e-12)[0]
        est_v[index] = prev_v            # store for error analysis


    # compute the diode current
    diode_i = compute_diode_current(est_v,ide_value,temp,is_value)



    return meas_i - diode_i

#########
def opt_ide(ide_value,r_value,phi_value,area,temp,src_v,meas_i):
    est_v   = np.zeros_like(src_v)       # an array to hold the diode voltages
    diode_i = np.zeros_like(src_v)       # an array to hold the diode currents
    prev_v = P1_VDD_STEP                 # an initial guess for the voltage

    # need to compute the reverse bias saturation current for this phi!
    is_value = area * temp * temp * np.exp(-phi_value * Q / ( k * temp ) )

    for index in range(len(src_v)):
        prev_v = fsolve(Diode_Current_2,prev_v,
				(src_v[index],r_value,ide_value,temp,is_value),
                                xtol=1e-12)[0]
        est_v[index] = prev_v            # store for error analysis

    # compute the diode current
    diode_i = compute_diode_current(est_v,ide_value,temp,is_value)
    return (meas_i - diode_i)/(meas_i + diode_i+1e-15)

############
def opt_phi(phi_value,r_value,ide_value,area,temp,src_v,meas_i):
    est_v   = np.zeros_like(src_v)       # an array to hold the diode voltages
    diode_i = np.zeros_like(src_v)       # an array to hold the diode currents
    prev_v = P1_VDD_STEP                 # an initial guess for the voltage

    # need to compute the reverse bias saturation current for this phi!
    is_value = area * temp * temp * np.exp(-phi_value * Q / ( k * temp ) )

    for index in range(len(src_v)):
        prev_v = fsolve(Diode_Current_2,prev_v,
				(src_v[index],r_value,ide_value,temp,is_value),
                                xtol=1e-12)[0]
        est_v[index] = prev_v            # store for error analysis

    # compute the diode current
    diode_i = compute_diode_current(est_v,ide_value,temp,is_value)

    return (meas_i - diode_i)/(meas_i + diode_i+1e-15)


A = 1e-8
T = 375
ide_val = 1.8
phi_val = 0.8
r_val= 10000
P1_VDD_STEP = 0


df = pd.read_csv("DiodeIV.txt", sep=" ", names=["src_v","diode_i"])

meas_diode_i =np.zeros(25)
src_v = np.zeros(25)

for i in range(len(src_v)):
    meas_diode_i[i] = df["diode_i"][i]
    src_v[i] = df["src_v"][i]

###################


r_val_opt = leastsq(opt_r,r_val,
                             args=(ide_val,phi_val,A,T,
                                   src_v,meas_diode_i))
r_val = r_val_opt[0][0]


print("r: ",r_val)

#################
ide_val_opt = leastsq(opt_ide,ide_val,
                             args=(r_val,phi_val,A,T,
                                   src_v,meas_diode_i))
ide_val = ide_val_opt[0][0]
print("n: ",ide_val)
#################
phi_val_opt = leastsq(opt_phi,phi_val,
                             args=(r_val,ide_val,A,T,
                                   src_v,meas_diode_i))
phi_val = phi_val_opt[0][0]
print("phi: ",phi_val)
