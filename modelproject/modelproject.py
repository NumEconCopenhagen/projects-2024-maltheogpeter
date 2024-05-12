import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve

class ADASModel:
    def __init__(self, beta1, beta2, beta3, gamma, a, expected_inflation, foreign_inflation, foreign_interest_rate, foreign_real_interest_rate, previous_exchange_rate, potential_output, delta, demand_shock=0):
        # Initialize the existing parameters
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
        self.delta = delta
        self.s_t = 0  # Initialize the stochastic term
        self.demand_shock = demand_shock  # New parameter for demand shock
        self.update_normal_distribution = False  # Flag to control the updating behavior

    def update_s_t(self):
        """Conditionally update the stochastic term using s_t = delta * s_{t-1} + N(0, 1)."""
        if self.update_normal_distribution:
            # Update only if the flag is set to update with a normal distribution
            self.s_t = self.delta * self.s_t + np.random.normal(0, 1)
        else:
            # Keep s_t fixed at zero
            self.s_t = 0

    def calculate_z_t(self, output):
        fiscal_gap = self.a * (self.potential_output - output)
        return self.beta3 * fiscal_gap

    def aggregate_demand(self, inflation):
        inflation_gap = self.foreign_inflation - inflation
        interest_diff = self.foreign_interest_rate -0 - self.foreign_real_interest_rate
        z_t = self.calculate_z_t(self.potential_output)

        # Incorporate the demand shock directly into the output gap
        output_gap = self.beta1 * (self.previous_exchange_rate + inflation_gap) - self.beta2 * interest_diff + z_t + self.demand_shock

        return output_gap + self.potential_output


    def aggregate_supply(self, output):
        """Incorporate the stochastic supply shock s_t."""
        return self.expected_inflation + self.gamma * (output - self.potential_output) + self.s_t

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
        plt.grid(True)
        plt.title("AD-AS Model (Analytical Solution)")
        plt.show()

    def objective(self, x):
        output, inflation = x
        ad_output = self.aggregate_demand(inflation)
        sras_inflation = self.aggregate_supply(output)
        return abs(ad_output - output) + abs(sras_inflation - inflation)

    def find_numerical_solution(self, initial_guess):
        solution = minimize(self.objective, initial_guess, method='Nelder-Mead')
        return solution

    def plot_graphs_all_periods_numerical(self, num_periods, initial_guess):
        output_values = np.linspace(60, 140, 100)
        inflation_values = np.linspace(0, 4, 100)
        final_numerical_solution = None

        plt.figure(figsize=(12, 8))
        
        for period in range(num_periods):
            # Update the stochastic supply term
            self.update_s_t()

            # Compute the AD and SRAS curves for this period
            ad_values = [self.aggregate_demand(infl) for infl in inflation_values]
            sras_values = [self.aggregate_supply(out) for out in output_values]

            # Plot each period's AD and SRAS curves
            plt.plot(ad_values, inflation_values, label=f"AD (Period {period + 1})", linestyle='--')
            plt.plot(output_values, sras_values, label=f"SRAS (Period {period + 1})", linestyle='-')

            # Annotate the last SRAS curve with the period number
            plt.text(output_values[-1], sras_values[-1], f"Period {period + 1}", fontsize=10)

        # Calculate the numerical solution after the final period
        final_solution = self.find_numerical_solution(initial_guess)
        if final_solution.success:
            final_numerical_solution = final_solution.x

        plt.axvline(x=self.potential_output, label="LRAS", color="black", linestyle=":")
        plt.xlabel("Output (Y)")
        plt.ylabel("Inflation (π)")
        plt.legend()
        plt.grid(True)
        plt.title(f"AD-AS Model Over {num_periods} Periods (Numerical)")
        plt.show()

        return final_numerical_solution

    def plot_graphs_all_periods_analytical(self, num_periods):
        output_values = np.linspace(60, 140, 100)
        inflation_values = np.linspace(0, 4, 100)
        final_analytical_solution = None

        plt.figure(figsize=(12, 8))
        
        for period in range(num_periods):
            # Update the stochastic supply term
            self.update_s_t()

            # Compute the AD and SRAS curves for this period
            ad_values = [self.aggregate_demand(infl) for infl in inflation_values]
            sras_values = [self.aggregate_supply(out) for out in output_values]

            # Plot each period's AD and SRAS curves
            plt.plot(ad_values, inflation_values, label=f"AD (Period {period + 1})", linestyle='--')
            plt.plot(output_values, sras_values, label=f"SRAS (Period {period + 1})", linestyle='-')

            # Annotate the last SRAS curve with the period number
            plt.text(output_values[-1], sras_values[-1], f"Period {period + 1}", fontsize=10)

        # Find the analytical intersection at the final period
        final_analytical_solution = self.find_intersection()

        plt.axvline(x=self.potential_output, label="LRAS", color="black", linestyle=":")
        plt.xlabel("Output (Y)")
        plt.ylabel("Inflation (π)")
        plt.legend()
        plt.grid(True)
        plt.title(f"AD-AS Model Over {num_periods} Periods (Analytical)")
        plt.show()

        return final_analytical_solution