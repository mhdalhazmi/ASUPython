###########################
# The program call three functions that do the following:
# 1- find a factorial of an integer given by the user
# 2- calculate the catlan value of an integer given by the user
# 3- find the common divisors of two integers given by the user
########################### 

# Ask the user to input values to find the factorial, catlan value and the GDC of two numbers
factorial_number = int(input("Enter an integer for a factorial computation: "))
catlan_number = int(input("Enter an integer for a Catalan number computation: "))
gcd_1 = int(input("Enter the first of two integers for a GCD calculation: "))
gcd_2 = int(input("Enter the second integer for the GCD calculation: "))

#  Function that uses recursion to compute the factorial of a number
def factorial(integer):
    if integer == 1:
        return 1
    
    else:
        return integer*factorial(integer-1)  # Call the function again and multiply current number with number-1

#  Function that uses recursion to compute the Catlan of a number    
def catalan(integer):
    if integer == 0:
        return 1
    
    else:
        catalan_equation = ( ( 4*integer - 2)*catalan(integer-1) )/(integer + 1)
        return catalan_equation
 
#  Function that uses recursion to compute the GCD of two numbers using Euclidean Algorithm
def greatest_common_divisor(m,n):
    if n == 0:
        return m
    
    else:
        return greatest_common_divisor(n,m % n)

# Output the computed values to the users along with the input
print ("")
print ("factorial of {} is {}".format(factorial_number, factorial(factorial_number)))
print ("catalan value of {} is {}".format(catlan_number,float(catalan(catlan_number))))
print ("greatest common divisor of {} and {} is {} ".format(gcd_1,gcd_2,greatest_common_divisor(gcd_1,gcd_2)))