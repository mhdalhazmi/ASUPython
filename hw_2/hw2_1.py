from gaussxw import gaussxw
import numpy as np
from matplotlib import pyplot as plt

def f(x):
  v = np.power(x,6)+2*np.power(x,4)
  v_a = np.power(2,6)+2*np.power(2,4)
  return ((8*m)**0.5)/((v_a-v)**0.5)

N = 20
a = 0
b = 2
m = 1

x,w = gaussxw(N)

xp = 0.5*(b-a)*x + 0.5*(b+a)
wp = 0.5*(b-a)*w

wp_plot = []
xp_plot = []

s=0.0
for k in range(N):
  s+= wp[k]*f(xp[k])
  wp_plot.append(xp[k])
  xp_plot.append(f(xp[k]))

  
  
plt.scatter(wp_plot,xp_plot)
plt.xlim(0,2)
plt.ylim(0,2)
plt.show()

print(s)
