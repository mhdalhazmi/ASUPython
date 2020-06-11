###########################
# The program will list the prime numbers from 1 to 10,000
# Using that you only need to check the divisors upto sqrt(num) effeciently
########################### 

# Import math library to use square roo and ceil
import math

# Initialize the list to avoid duplication while appending 
my_prime_list=[2]
# Initiate the lower and upper limit of the numbers to checked
for num in range(3, 11):
   # Initiate the lower and upper limit for the divisor 
   for divisor in range(2, math.sqrt(num))):
       # If the number you are checking divided by the divisors has a remainder then it is not a prime so ignore and exit the loop
       if (num % divisor) == 0:
           break
   # If the number you are checking divided by all the divisors upto sqrt(num) do not have a remainder then it is a prime and add it to our list
   else:
       my_prime_list.append(num)
# print the list of prime numbers found
print(my_prime_list)



