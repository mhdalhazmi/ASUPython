
import numpy as np
a = np.array([ [1, -49.653j],
                [0, 1] ])
c = np.array([ [0.8182+0.0170j, 15+165.61j],
                [2e-3j, 0.8182+0.0170j] ])

b = np.array([ [1, -49.653j],
                [0, 1] ])


x = np.dot(a,b)
y = np.dot(x, c)
print(y)
#y = np.vdot(x,b)
#print(y)
