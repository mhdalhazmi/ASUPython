import numpy as np
from scipy.optimize import fsolve, leastsq
import matplotlib.pyplot as plt
import pandas as pd

def compute_diode_current(Vd,n,T,Is):
    Vt = n*k*T/Q
    return  Is*(np.exp(Vd/Vt)-1)


def Diode_Current(v_diode, v_src,r,n,T,i_s):
    i_diode = DiodeI(v_diode,A,phi_val,n,T)
    err = (v_diode-v_src)/r + i_diode
    return err


def opt_r(r_value,ide_value,phi_value,area,temp,src_v,meas_i):
    est_v   = np.zeros_like(src_v)       # an array to hold the diode voltages
    diode_i = np.zeros_like(src_v)       # an array to hold the diode currents
    prev_v = 2                 # an initial guess for the voltage ############################################
    # need to compute the reverse bias saturation current for this phi!
    is_value = area * temp * temp * np.exp(-phi_value * Q / ( k * temp ) )

    for index in range(len(src_v)):
        prev_v = fsolve(Diode_Current,prev_v,
				(src_v[index],r_value,ide_value,temp,is_value),
                                xtol=1e-12)[0]
        est_v[index] = prev_v            # store for error analysis


    # compute the diode current
    diode_i = compute_diode_current(est_v,ide_value,temp,is_value)



    return meas_i - diode_i


def opt_phi(phi_value,ide_value,r_value,area,temp,src_v,meas_i):
    est_v   = np.zeros_like(src_v)       # an array to hold the diode voltages
    diode_i = np.zeros_like(src_v)       # an array to hold the diode currents
    prev_v = 2                 # an initial guess for the voltage ############################################
    # need to compute the reverse bias saturation current for this phi!
    is_value = area * temp * temp * np.exp(-phi_value * Q / ( k * temp ) )

    for index in range(len(src_v)):
        prev_v = fsolve(Diode_Current,prev_v,
				(src_v[index],r_value,ide_value,temp,is_value),
                                xtol=1e-12)[0]
        est_v[index] = prev_v            # store for error analysis


    # compute the diode current
    diode_i = compute_diode_current(est_v,ide_value,temp,is_value)


    return (meas_i - diode_i)/(meas_i + diode_i + 1e-15)



A = 1e08
T= 375
k = 1.380648e-23
Q = 1.6021766208e-19
phi_val = 0.8
ide_val = 1.5
r_val = 10000


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

phi_val_opt = leastsq(opt_phi,phi_val,
                             args=(ide_val,r_val,A,T,
                                   src_v,meas_diode_i))
phi_val = phi_val_opt[0][0]


print("r: ",phi_val)
