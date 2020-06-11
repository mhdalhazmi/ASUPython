###########################
# The program will list prime numbers from 1 to 10,000
# Efficiently by checking the divisors upto sqrt(num). If no divisors have been found then number is prime 
########################### 


import math                                                     Import math library to use square root and floor functions      
my_prime_list=[2]                                              # Initialize the list of prime numbers with 2 to avoid duplication while executing the program again 

for num in range(3, 10001):                                    # Initiate the lower and upper limit of the numbers to be checked
   
   for divisor in range(2, int(math.sqrt(num))+1):             # Initiate the lower and upper limit for the divisor 
       if (num % divisor) == 0:                                # If the number has any divisors (i.e. remainder = 0) in the range then the number is not a prime and no need to check any further 
           break
  
   else:                                                       # This else statement will only be executed if the loop terminates properly with no breaks
       my_prime_list.append(num)                               # If the number has no divisors in the range then it is a prime and add it to the list

print(my_prime_list)                                           # print the list of prime numbers found in the range



