from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt  
import scipy.optimize as optimize
class ExchangeEconomyClass:

    def __init__(self):
        par = self.par = SimpleNamespace()
        par.alpha = 1/3
        par.beta = 2/3
        par.w1A = 0.8
        par.w2A = 0.3
        par.w1B = 1 - par.w1A
        par.w2B = 1 - par.w2A
        par.p2 = 1

    def utility_A(self, x1A, x2A):
        return x1A**self.par.alpha * x2A**(1 - self.par.alpha)

    def utility_B(self, x1B, x2B):
        return x1B**self.par.beta * x2B**(1 - self.par.beta)

    def demand_A(self, p1):
        x1A = (self.par.alpha * p1 * self.par.w1A + self.par.p2 * self.par.w2A) / p1
        x2A = ((1 - self.par.alpha) * p1 * self.par.w1A + self.par.p2 * self.par.w2A) / self.par.p2
        return x1A, x2A

    def demand_B(self, p1):
        x1B = (self.par.beta * p1 * self.par.w1B + self.par.p2 * self.par.w2B) / p1
        x2B = ((1 - self.par.beta) * self.par.p2 * self.par.w2B + p1 * self.par.w1B) / self.par.p2
        return x1B, x2B

    def demand_A_x1(self, p1):
        return self.par.alpha * (p1 * self.par.w1A + self.par.p2 * self.par.w2A) / p1

    def demand_A_x2(self, p1):
        return (1 - self.par.alpha) * (p1 * self.par.w1A + self.par.p2 * self.par.w2A) / self.par.p2

    def demand_B_x1(self, p1):
        return self.par.beta * (p1 * self.par.w1B + self.par.p2 * self.par.w2B) / p1

    def demand_B_x2(self, p1):
        return (1 - self.par.beta) * (p1 * self.par.w1B + self.par.p2 * self.par.w2B) / self.par.p2

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

    def pareto(self, N=75):
        x1A_vek = np.linspace(0, 1, N)
        x2A_vek = np.linspace(0, 1, N)
        uA_bar = self.utility_A(self.par.w1A, self.par.w2A)
        uB_bar = self.utility_B(self.par.w1B, self.par.w2B)
        combinations = [(x1a, x2a) for x1a in x1A_vek for x2a in x2A_vek
                        if self.utility_A(x1a, x2a) >= uA_bar and
                        self.utility_B(1 - x1a, 1 - x2a) >= uB_bar]
        return combinations

    def edgeworth_box(self, N=75):
        par = self.par
        pareto_combinations = self.pareto(N)
        x1A, x2A = zip(*pareto_combinations)
        plt.style.use('seaborn-ticks')
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

    def price_vector(self, N=75):
        P_1 = np.linspace(0.5, 2.5, N)
        return P_1

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