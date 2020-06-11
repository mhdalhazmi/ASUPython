
###########################
# The program will list the prime numbers from 1 to 10,000
# Using that you only need to check the divisors upto sqrt(num) effeciently
###########################


import tkinter
from tkinter import *
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import math
import random

r = float(input("mean return: "))
stand_dev = int(input("stand dev: "))
yearly_cont = float(input("yearly cont: "))
year_count = int(input("no of years of cont: "))
year_ret = int(input("no of years to retairment: "))
fun = float(input("actual retiremnt spend: "))



i,x = 0,0
w = np.zeros((10,70))
while x < 10:
    w[x][0] = yearly_cont
    x += 1

while i <10:
    year_counter = 1
    while year_counter < year_count:
        x = random.uniform(-1,1)
        w[i][year_counter] = w[i][year_counter-1]*(1+r/100+ x*stand_dev/100)+yearly_cont
        year_counter +=1
    while year_counter >= year_count and year_counter < year_ret:
        x = random.uniform(-1,1)
        w[i][year_counter] = w[i][year_counter-1]*(1+r/100+ x*stand_dev/100)
        year_counter +=1
    while year_counter >= year_ret and year_counter < 70:
        x = random.uniform(-1,1)
        w[i][year_counter] = w[i][year_counter-1]*(1+r/100+ x*stand_dev/100) - fun
        if w[i][year_counter] <= 0:
            w[i][year_counter] = 0
            break
        year_counter +=1
    i +=1

plt.plot(range(1,71),w)
plt.show()
