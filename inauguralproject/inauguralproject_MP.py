# Importerer packacges

from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Define parameters
alpha = 1/3 #alpha is the preference parameter for good 1
beta = 2/3 #beta is the preference parameter for good 2
N=75 
omega_A1 = 0.8 #omega_A1 is the initial endowment of good 1 for agent A
omega_A2 = 0.3 #omega_A2 is the initial endowment of good 2 for agent A
omega_B1 = 1-omega_A1 #omega_B1 is the initial endowment of good 1 for agent B
omega_B2 = 1-omega_A2 #omega_B2 is the initial endowment of good 2 for agent B
p2 = 1  # Numeraire price of good 2

# Utility functions
def uA(x1, x2):
    return x1**alpha * x2**(1-alpha) #utility function of agent A

def uB(x1, x2):
    return x1**beta * x2**(1-beta) #utility function of agent B

#Demand functions 
def demand_A_x1(omega_A1, omega_A2, p1, alpha):
    return alpha*(p1*omega_A1 + p2*omega_A2)/p1 #demand function for good 1 for agent A

def demand_A_x2(omega_A1, omega_A2, p1, alpha):
    return (1-alpha)*(p1*omega_A1 + p2*omega_A2)/p2 #demand function for good 2 for agent A

def demand_B_x1(omega_A1, omega_A2, p1, beta):
    return beta*(p1*omega_A1 + p2*omega_A2)/p1 #demand function for good 1 for agent B

def demand_B_x2(omega_A1, omega_A2, p1, beta):
    return (1-beta)*(p1*omega_A1 + p2*omega_A2)/p2 #demand function for good 2 for agent B

# Walras market equilibrium clearing conditions
 # The sum of the demands for good 1 from agent A and agent B must equal the total supply of good 1
def market_clearing_x1(p1)
    total_demand_x_1= demand_A_x1(omega_A1, omega_A2, p1, alpha) + demand_B_x1(omega_A1, omega_A2, p1, beta)
    total_supply_x_1 = omega_A1 + omega_B1
    return total_demand_x_1 - total_supply_x_1

def market_clearing_x2(p1):
    total_demand_x_2= demand_A_x2(omega_A1, omega_A2, p1, alpha) + demand_B_x2(omega_A1, omega_A2, p1, beta)
    total_supply_x_2 = omega_A2 + omega_B2
    return total_demand_x_2 - total_supply_x_2

