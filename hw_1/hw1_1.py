###################################
# This  program will let you calculate the time it takes for an object to fall and hit the ground
# when thrown from a certain height. User will control the initial height and upward or downward velocity 
###################################
import math

GRAVITY= -9.81                                                                  #Earth Gravity and it is negative since it is downward 

ball_initial_height= float(input("enter height: "))                             # Initial height of the ball inputted by the user 

if ball_initial_height>=10 and ball_initial_height<=1000:                       # Checks if the initial height is within the limit 
    
    initial_vertical_velocity = float(input("enter initial upward velocity: ")) # Initial velocity of the ball inputted by the user. Positive value means upward and negative means downward
    if -20 <= initial_vertical_velocity <= 20:        # Checks if the initial velocity is within the limit 
                                                                                # Following lines will solve the quadratic equation to find the time. 0.5gt^2+v0t+h0=0
        root= math.sqrt( (initial_vertical_velocity**2) \
                        -(4 * 0.5 * GRAVITY * ball_initial_height) )
        fall_time = (-initial_vertical_velocity - root) / GRAVITY
        
        print("time to hit ground %.2f seconds" %fall_time)                     # Print the time it takes for the ball to hit the ground upto two decimal places
    
    else:
        
        print("Initial velocity is too large! Slow down!")                      # User did not meet the initial velocity limit so exit

else:                                                                           # User did not meet the initial hiegt limit so exit            
    
    print("Bad height specified. Please try again")
