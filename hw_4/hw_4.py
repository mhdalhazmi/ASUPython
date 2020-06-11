
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



def calculate ():

####### Read User Input ##############
    r = float(mean_return_entry.get())
    stand_dev = float(stand_dev_entry.get())
    yearly_cont = float(yearly_cont_entry.get())
    year_count = int(yearly_count_entry.get())
    year_ret = int(year_ret_entry.get())
    fun = float(ann_spend_entry.get())

##### Initialize the first year in the array ##########
    i,x = 0,0
    w = np.zeros((10,70))
    while x < 10:
        w[x][0] = yearly_cont
        x += 1

############# The 10 Scenarios start here ########

    while i <10:
        year_counter = 1
######## Are they still contributing ##############
        while year_counter < year_count:
            x = np.random.randn()
            w[i][year_counter] = w[i][year_counter-1]*(1+r/100+ x*stand_dev/100)+yearly_cont
            year_counter +=1
######## Have they finished contributing ##############
        while year_counter >= year_count and year_counter < year_ret:
            x = np.random.randn()
            w[i][year_counter] = w[i][year_counter-1]*(1+r/100+ x*stand_dev/100)
            year_counter +=1
######## Have they retired ##############
        while year_counter >= year_ret and year_counter < 70:
            x = np.random.randn()
            w[i][year_counter] = w[i][year_counter-1]*(1+r/100+ x*stand_dev/100) - fun
            if w[i][year_counter] <= 0:
                w[i][year_counter] = 0
                break
            year_counter +=1
        i +=1
####### Find the average of all scenarios for each year ######
    average= []
    j ,z= 0,0
    while j <70 :
        average.append(np.average(w[:,j]))
        j += 1
####### Plot all the scenarios with respect to year ######
    while z < 10:
        plt.plot(range(1,71),w[z,:])
        z += 1
    wealth_label_2.configure(text = " $ {:,.2f}" .format(average[year_ret-1]))
    plt.xlabel("Years")
    plt.ylabel("Wealth")
    plt.title("Wealth in 70 Years")
    plt.grid(True)
    plt.show()

####### GUI ######

root = Tk()

root.title("See How Rich You Are Going To Be")

calculate_button = Button(root, text="Calculate", pady=10, padx=10, command = calculate)
calculate_button.grid(row = 7, column =6)
quit_button = Button(root, text="Quit", pady=10, padx=10, command = root.destroy)
quit_button.grid(row = 7, column =5)

mean_return_label = Label(root, text = "Mean Return (%)")
mean_return_entry = Entry(root,width = 50)
mean_return_entry.grid(row=0, column =5, columnspan = 5)
mean_return_label.grid(row=0, column =0, columnspan = 5)

stand_dev_label = Label(root, text = "Std Dev Return (%)")
stand_dev_entry = Entry(root,width = 50)
stand_dev_entry.grid(row=1, column =5, columnspan = 5)
stand_dev_label.grid(row=1, column =0, columnspan = 5)

yearly_cont_label = Label(root, text = "Yearly Contribution ($)")
yearly_cont_entry = Entry(root,width = 50)
yearly_cont_entry.grid(row=2, column =5, columnspan = 5)
yearly_cont_label.grid(row=2, column =0, columnspan = 5)

yearly_count_label = Label(root, text = "No. of Years of Contribution ")
yearly_count_entry = Entry(root,width = 50)
yearly_count_entry.grid(row=3, column =5, columnspan = 5)
yearly_count_label.grid(row=3, column =0, columnspan = 5)

year_ret_label = Label(root, text = "No. of Years to Retirement")
year_ret_entry = Entry(root,width = 50)
year_ret_entry.grid(row=4, column =5, columnspan = 5)
year_ret_label.grid(row=4, column =0, columnspan = 5)

ann_spend_label = Label(root, text = "Annual Spend in Retirement")
ann_spend_entry = Entry(root,width = 50)
ann_spend_entry.grid(row=5, column =5, columnspan = 5)
ann_spend_label.grid(row=5, column =0, columnspan = 5)

wealth_label_1 = Label(root, text = "Your Expected wealth is ")
wealth_label_2 = Label(root, text = "$ 0")
wealth_label_2.grid(row=6, column =5, columnspan = 5)
wealth_label_1.grid(row=6, column =0, columnspan = 5)

mean_return_entry.insert(0,6)
stand_dev_entry.insert(0,20)
yearly_count_entry.insert(0,40)
yearly_cont_entry.insert(0,20000)
year_ret_entry.insert(0,50)
ann_spend_entry.insert(0,120000)



mainloop()
