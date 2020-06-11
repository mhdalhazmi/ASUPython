
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