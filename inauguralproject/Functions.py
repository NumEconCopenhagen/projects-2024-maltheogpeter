from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt  
import scipy.optimize as optimize
import seaborn as sns
from scipy.optimize import minimize_scalar
class ExchangeEconomyClass:

#Define initial parameters
    def __init__(self):
        par = self.par = SimpleNamespace()
        par.alpha = 1/3
        par.beta = 2/3
        par.w1A = 0.8
        par.w2A = 0.3
        par.w1B = 1 - par.w1A
        par.w2B = 1 - par.w2A
        par.p2 = 1

#Define utility and demand functions

    def demand_A(self, p1):
        x1A = self.par.alpha * ((p1 * self.par.w1A + self.par.p2 * self.par.w2A) / p1)
        x2A = (1 - self.par.alpha) * ((p1 * self.par.w1A + self.par.p2 * self.par.w2A) / self.par.p2)
        return x1A, x2A

    def demand_B(self, p1):
        x1B = (self.par.beta * (p1 * self.par.w1B + self.par.p2 * self.par.w2B)) / p1
        x2B = ((1 - self.par.beta) * (self.par.p2 * self.par.w2B + p1 * self.par.w1B)) / self.par.p2
        return x1B, x2B
    
    def utility_A(self, x1A, x2A):
        return x1A**self.par.alpha * x2A**(1 - self.par.alpha)

    def utility_B(self, x1B, x2B):
        return x1B**self.par.beta * x2B**(1 - self.par.beta)
    
    def demand_A_x1(self, p1):
        return self.par.alpha * (p1 * self.par.w1A + self.par.p2 * self.par.w2A) / p1

    def demand_A_x2(self, p1):
        return (1 - self.par.alpha) * (p1 * self.par.w1A + self.par.p2 * self.par.w2A) / self.par.p2

    def demand_B_x1(self, p1):
        return self.par.beta * (p1 * self.par.w1B + self.par.p2 * self.par.w2B) / p1

    def demand_B_x2(self, p1):
        return (1 - self.par.beta) * (p1 * self.par.w1B + self.par.p2 * self.par.w2B) / self.par.p2

#Market clearing conditions
    def market_clearing_x1(self, p1):
        total_demand_x1 = self.demand_A_x1(p1) + self.demand_B_x1(p1)
        total_supply_x1 = self.par.w1A + self.par.w1B
        return total_demand_x1 - total_supply_x1

    def market_clearing_x2(self, p1):
        total_demand_x2 = self.demand_A_x2(p1) + self.demand_B_x2(p1)
        total_supply_x2 = self.par.w2A + self.par.w2B
        return total_demand_x2 - total_supply_x2
    
    def check_market_clearing(self, p1):
        x1A, x2A = self.demand_A(p1)
        x1B, x2B = self.demand_B(p1)
        eps1 = x1A + x1B - (self.par.w1A + self.par.w1B)
        eps2 = x2A + x2B - (self.par.w2A + self.par.w2B)
        return eps1, eps2

#Pareto optimal combinations
    def pareto(self, N=75):
        x1A_vek = np.linspace(0, 1, N)
        x2A_vek = np.linspace(0, 1, N)
        uA_bar = self.utility_A(self.par.w1A, self.par.w2A)
        uB_bar = self.utility_B(self.par.w1B, self.par.w2B)
        combinations = [(x1a, x2a) for x1a in x1A_vek for x2a in x2A_vek
                        if self.utility_A(x1a, x2a) >= uA_bar and
                        self.utility_B(1 - x1a, 1 - x2a) >= uB_bar]
        return combinations
#Plotting
    def edgeworth_box_1(self, N=75):
        par = self.par
        pareto_combinations = self.pareto(N)
        x1A, x2A = zip(*pareto_combinations)
        fig = plt.figure(figsize=(6, 6), dpi=100)
        ax_A = fig.add_subplot(1, 1, 1)
        ax_A.set_xlabel("$x_1^A$", fontsize=12)
        ax_A.set_ylabel("$x_2^A$", fontsize=12)
        ax_B = ax_A.twinx().twiny()
        ax_B.set_xlabel("$x_1^B$", fontsize=12)
        ax_B.set_ylabel("$x_2^B$", fontsize=12)
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()
        ax_A.scatter(x1A, x2A, marker='o', color='skyblue', label='Pareto Optimal (A)', s=50)
        ax_A.scatter(par.w1A, par.w2A, marker='x', color='navy', label='Initial Endowment (A)', s=100)
        ax_A.set_xlim([-0.1, 1.1])
        ax_A.set_ylim([-0.1, 1.1])
        ax_B.set_xlim([1.1, -0.1])
        ax_B.set_ylim([1.1, -0.1])
        ax_A.legend( frameon=True)
        plt.show()

#Market clearing conditions
    def market_clearing_conditions_2(self, N=75):
        p1_values = np.linspace(0.5, 2.5, N) 
        epsilon_1 = []
        epsilon_2 = []
        p1_valuess= []
        for p1 in p1_values:
            error_1 = self.market_clearing_x1(p1)  # Fix: Added 'self.' to call the method within the class
            error_2 = self.market_clearing_x2(p1)
            epsilon_1.append(error_1)
            epsilon_2.append(error_2)
            p1_valuess.append(p1)
        plt.figure(figsize=(14, 7))
        plt.plot(p1_valuess, epsilon_1, label='Error in Market Clearing for Good 1')
        plt.plot(p1_valuess, epsilon_2, label='Error in Market Clearing for Good 2')
        plt.xlabel('$p_1$', fontsize=14)
        plt.ylabel('Market Clearing Error', fontsize=14)
        plt.title('Market Clearing Errors for Good 1 and Good 2', fontsize=16)
        plt.axhline(0, color='black', lw=0.5, ls='--')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

#Optimal price
    def optimize_price_3(self):
        result = optimize.minimize_scalar(self.abs_error, bounds=(0.5, 2.5), method='bounded')
        return result.x
#Optimal p1 with p1 as market setter
    def maximize_utility_A_via_grid_search_4A(self, p1_values=np.linspace(0.5, 2.5, 75)):
        utility_1 = -np.inf
        price_1 = None
        
        for p1 in p1_values:
            demand_B_x1, demand_B_x2 = self.demand_B(p1)
            if 1 - demand_B_x1 > 0 and 1 - demand_B_x2 > 0:
                utilitymax_A = self.utility_A(1 - demand_B_x1, 1 - demand_B_x2)
                if utilitymax_A > utility_1:
                    utility_1 = utilitymax_A
                    price_1 = p1
        
        x1A_allocation = 1 - self.demand_B(price_1)[0]
        x2A_allocation = 1 - self.demand_B(price_1)[1]
        
        return price_1, utility_1, x1A_allocation, x2A_allocation
 
 #Optimal p1 with p1 as market setter, undbounded   
    def maximize_utility_A_unbounded(self):
        def negative_utility_A(p1):
            demand_B_x1, demand_B_x2 = self.demand_B(p1)
            if 1 - demand_B_x1 > 0 and 1 - demand_B_x2 > 0:
                return -self.utility_A(1 - demand_B_x1, 1 - demand_B_x2)
            else:
                return np.inf  # Return a very high value if the allocation is not feasible

        result = minimize_scalar(negative_utility_A, bounds=(1e-3, 1e3), method='bounded')
        
        optimal_p1 = result.x
        optimal_utility_A = -result.fun
        x1A_allocation = 1 - self.demand_B(optimal_p1)[0]
        x2A_allocation = 1 - self.demand_B(optimal_p1)[1]

        return optimal_p1, optimal_utility_A, x1A_allocation, x2A_allocation


    def find_optimal_price(self, price_range):
        smallest_excess_demand_1 = 10
        smallest_excess_demand_2 = 10
        optimal_excess_demand_1, optimal_excess_demand_2, optimal_price = None, None, None

        for price in price_range:
            current_excess_demand_1, current_excess_demand_2 = self.check_market_clearing(price)

            if np.abs(current_excess_demand_1) < np.abs(smallest_excess_demand_1) and np.abs(current_excess_demand_2) < np.abs(smallest_excess_demand_2):
                smallest_excess_demand_1, smallest_excess_demand_2 = current_excess_demand_1, current_excess_demand_2
                optimal_excess_demand_1, optimal_excess_demand_2, optimal_price = current_excess_demand_1, current_excess_demand_2, price

        return optimal_excess_demand_1, optimal_excess_demand_2, optimal_price

    def abs_error(self, p1):
        error_1 = self.market_clearing_x1(p1)
        error_2 = self.market_clearing_x2(p1)
        return abs(error_1) + abs(error_2) 
    
    def aggregate_utility(self, x):
        x1A, x2A = x  
        x1B, x2B = 1 - x1A, 1 - x2A  
        utility_A = self.utility_A(x1A, x2A)
        utility_B = self.utility_B(x1B, x2B)
        return -(utility_A + utility_B)  
    
    def generate_random_endowments(self, num_samples=50):
        endowments = np.random.uniform(0, 1, (num_samples, 2))
        return endowments

    def find_market_equilibrium_allocation(self, w1A, w2A):
        self.par.w1A = w1A
        self.par.w2A = w2A
        self.par.w1B = 1 - w1A
        self.par.w2B = 1 - w2A
        
        def market_clearing_error(p1):
            error1, error2 = self.check_market_clearing(p1)
            return abs(error1) + abs(error2)
        
        result = minimize_scalar(market_clearing_error, bounds=(1e-3, 1e3), method='bounded')
        optimal_p1 = result.x
        x1A, x2A = self.demand_A(optimal_p1)
        x1B, x2B = self.demand_B(optimal_p1)
        return x1A, x2A, x1B, x2B

    def plot_edgeworth_box_with_allocations(self, allocations):
        fig = plt.figure(figsize=(6, 6), dpi=100)
        ax_A = fig.add_subplot(1, 1, 1)
        ax_A.set_xlabel("$x_1^A$", fontsize=12)
        ax_A.set_ylabel("$x_2^A$", fontsize=12)
        ax_B = ax_A.twinx().twiny()
        ax_B.set_xlabel("$x_1^B$", fontsize=12)
        ax_B.set_ylabel("$x_2^B$", fontsize=12)
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()

        x1A_vals, x2A_vals = zip(*allocations)
        x1B_vals = [1 - x1A for x1A in x1A_vals]
        x2B_vals = [1 - x2A for x2A in x2A_vals]

        ax_A.scatter(x1A_vals, x2A_vals, marker='o', color='skyblue', label='Allocations (A)', s=50)
        ax_A.scatter(self.par.w1A, self.par.w2A, marker='x', color='navy', label='Initial Endowment (A)', s=100)

        ax_A.set_xlim([-0.1, 1.1])
        ax_A.set_ylim([-0.1, 1.1])
        ax_B.set_xlim([1.1, -0.1])
        ax_B.set_ylim([1.1, -0.1])
        ax_A.legend(frameon=True)
        plt.show()
