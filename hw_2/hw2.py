from gaussxw import gaussxw
import numpy as np
from matplotlib import pyplot as plt

grap_1 =[]
grap_2 =[]


for a in np.arange(0.1,2.1,0.01):
    def f(x,a):
      v = np.power(x,6)+2*np.power(x,4)
      v_a = np.power(0.11,6)+2*np.power(0.11,4)
      return -1*((8*m)**0.5)/((np.power(a,6)+2*np.power(a,4)-np.power(x,6)+2*np.power(x,4))**0.5)
    
    N = 20
    b = 0
    m = 1
    
    x,w = gaussxw(N)
    
    xp = 0.5*(b-a)*x + 0.5*(b+a)
    wp = 0.5*(b-a)*w
    
    wp_plot = []
    xp_plot = []
    
    s=0.0
    for k in range(N):
      s+= wp[k]*f(xp[k],a)
      #wp_plot.append(xp[k])
      #xp_plot.append(f(xp[k]))
    
    grap_1.append(s)
    grap_2.append(a)
#print(grap_1,grap_2)  
plt.plot(grap_1,grap_2)  
#plt.scatter(wp_plot,xp_plot)
#plt.xlim(0,2)
#plt.ylim(0,2)
plt.show()

print(s)
