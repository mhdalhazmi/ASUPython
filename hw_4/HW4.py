
###########################
# The program will list the prime numbers from 1 to 10,000
# Using that you only need to check the divisors upto sqrt(num) effeciently
###########################

# Import math library to use square roo and ceil
import math
from random import randint
stand_dev = int(input("stand dev: "))
yearly_cont = float(input("yearly cont"))
r = float(input("mean return"))
year_cont = int(input("no of years of cont"))
year_ret = int(input("no of years to retairment"))
fun = float(input("actual retiremnt spend"))

year_counter = 0
i = 0
w = yearly_cont
while year_counter <= year_cont:
    while i < 10:
        x = randint(-stand_dev/100,stand_dev/100 )
        w += w*(1+r/100+ x)+yearly_cont
        i +=1
        print("bye")
    i = 0
    print(year_counter)
    year_counter +=1

while year_counter > year_cont and year_counter <=year_ret:
    while i < 10:
        x = randint(-stand_dev/100,stand_dev/100 )
        w += w*(1+r/100+ x)
        i +=1
        print("hi")
    i = 0
    print(year_counter)
    year_counter +=1

while year_counter > year_ret:
    while i < 10:
        x = randint(-stand_dev/100,stand_dev/100 )
        w += w*(1+r/100+ x)-fun
        i +=1
        print("hi")
    i = 0
    print(year_counter)
    year_counter +=1
print (w)









root = Tk()

root.title("Your death")
calculate_button = Button(root, text="Calculate", pady=10, padx=10)
calculate_button.grid(row = 6, column =0)
quit_button = Button(root, text="Quit", pady=10, padx=10)
quit_button.grid(row = 6, column =1)
mean_return_label = Label(root, text = "Mean Return (%)")
mean_return = Entry(root,width = 50)
mean_return.grid(row=0, column =1)
mean_return_label.grid(row=0, column =0)

stand_dev_label = Label(root, text = "Std Dev Return (%)")
stand_dev = Entry(root,width = 50)
stand_dev.grid(row=1, column =1)
stand_dev_label.grid(row=1, column =0)

yearly_cont_label = Label(root, text = "Yearly Contribution ($)")
yearly_cont = Entry(root,width = 50)
yearly_cont.grid(row=2, column =1)
yearly_cont_label.grid(row=2, column =0)

yearly_count_label = Label(root, text = "No. of Years of Contribution ")
yearly_count = Entry(root,width = 50)
yearly_count.grid(row=3, column =1)
yearly_count_label.grid(row=3, column =0)

year_ret_label = Label(root, text = "No. of Years to Retirement")
year_ret = Entry(root,width = 50)
year_ret.grid(row=4, column =1)
year_ret_label.grid(row=4, column =0)

ann_spend_label = Label(root, text = "Annual Spend in Retirement")
ann_spend = Entry(root,width = 50)
ann_spend.grid(row=5, column =1)
ann_spend_label.grid(row=5, column =0)






mainloop()
