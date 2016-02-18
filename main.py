from math import cos, sin
import random

# compute the next states given the force and the current states
def simulate(force, state):
    x = state[0];
    x_dot = state[1];
    theta = state[2];
    theta_dot = state[3];
    
    GRAVITY=9.8;
    MASSCART=1.0;
    MASSPOLE=0.1;
    TOTAL_MASS=MASSPOLE + MASSCART;
    LENGTH=0.5;		  
    POLEMASS_LENGTH=MASSPOLE * LENGTH;
    STEP=0.02;
    FOURTHIRDS=4.0/3;

    costheta = cos(theta);
    sintheta = sin(theta);

    temp = (force + POLEMASS_LENGTH * theta_dot  *theta_dot * sintheta)/ TOTAL_MASS;

    thetaacc = (GRAVITY * sintheta - costheta* temp)/(LENGTH * (FOURTHIRDS - MASSPOLE * costheta * costheta/ TOTAL_MASS));

    xacc  = temp - POLEMASS_LENGTH * thetaacc* costheta / TOTAL_MASS;

    # Update the four state variables, using Euler's method.

    return (x+STEP*x_dot,
            x_dot+STEP*xacc,
            theta+STEP*theta_dot,
            theta_dot+STEP*thetaacc
            );
    
def system_safe(state):
    return abs(state[0]) < 2.4 and abs(state[2]) < 12


state = (0.0,0,0,0)
for i in range(0, 1000, 2):
    print("%.2fsec" % (i/100.0), state);
    print(system_safe(state))
    state = simulate(10 if bool(random.getrandbits(1)) else -10, state);