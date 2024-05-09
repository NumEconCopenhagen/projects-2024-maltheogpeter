import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve

class ADASModel:
    def __init__(self, beta1, beta2, beta3, gamma, a, expected_inflation, foreign_inflation, foreign_interest_rate, foreign_real_interest_rate, previous_exchange_rate, potential_output, supply_shock):
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.gamma = gamma
        self.a = a
        self.expected_inflation = expected_inflation
        self.foreign_inflation = foreign_inflation
        self.foreign_interest_rate = foreign_interest_rate
        self.foreign_real_interest_rate = foreign_real_interest_rate
        self.previous_exchange_rate = previous_exchange_rate
        self.potential_output = potential_output
        self.supply_shock = supply_shock

    def calculate_z_t(self, output):
        fiscal_gap = self.a * (self.potential_output - output)
        return self.beta3 * fiscal_gap

    def aggregate_demand(self, inflation):
        inflation_gap = self.foreign_inflation - inflation
        interest_diff = self.foreign_interest_rate - self.foreign_real_interest_rate
        z_t = self.calculate_z_t(self.potential_output)
        output_gap = self.beta1 * (self.previous_exchange_rate + inflation_gap) - self.beta2 * interest_diff + z_t
        return output_gap + self.potential_output

    def aggregate_supply(self, output):
        return self.expected_inflation + self.gamma * (output - self.potential_output) + self.supply_shock

    def find_intersection(self):
        output, inflation = symbols('output inflation')
        ad_equation = Eq(self.aggregate_demand(inflation), output)
        sras_equation = Eq(self.aggregate_supply(output), inflation)
        intersection = solve((ad_equation, sras_equation), (output, inflation))
        return intersection

    def plot_graph(self):
        output_values = np.linspace(60, 140, 100)
        inflation_values = np.linspace(0, 4, 100)
        ad_values = [self.aggregate_demand(infl) for infl in inflation_values]
        sras_values = [self.aggregate_supply(out) for out in output_values]
        plt.figure(figsize=(10, 8))
        plt.plot(ad_values, inflation_values, label="AD")
        plt.plot(output_values, sras_values, label="SRAS")
        plt.axvline(x=self.potential_output, label="LRAS", color="black", linestyle=":")
        plt.xlabel("Output (Y)")
        plt.ylabel("Inflation (π)")
        plt.legend()
        plt.grid(True)
        plt.title("AD-AS Model (Analytical solution)")
        plt.show()

    def objective(self, x):
        output, inflation = x
        ad_output = self.aggregate_demand(inflation)
        sras_inflation = self.aggregate_supply(output)
        return abs(ad_output - output) + abs(sras_inflation - inflation)

    def find_numerical_solution(self, initial_guess):
        solution = minimize(self.objective, initial_guess, method='Nelder-Mead')
        return solution

    def optimize_parameters(self, initial_guess=None, method='numerical'):
        if method == 'numerical':
            if not initial_guess:
                raise ValueError("Initial guess required for numerical optimization.")
            solution = self.find_numerical_solution(initial_guess)
            if solution.success:
                print("Intersection Point (Numerical):", solution.x)
                return solution.x
            else:
                print("Optimization failed:", solution.message)
                return None
        elif method == 'analytical':
            intersection = self.find_intersection()
            return intersection
