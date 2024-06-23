import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_inflation_vs_rate_over_time(merged_df):
    # Line plot for inflation and federal funds rate over time
    plt.figure(figsize=(12, 6))
    plt.plot(merged_df['DATE'], merged_df['inflation'], label='Inflation', color='blue')
    plt.plot(merged_df['DATE'], merged_df['rate'], label='Federal Funds Rate', color='red')
    
    # Set date format on x-axis
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator(5))  # Major ticks every 5 years
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))  # Minor ticks every year
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Enable the grid for minor ticks
    ax.grid(which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(which='both', width=1)
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4, color='gray')

    plt.xlabel('Date')
    plt.ylabel('Percentage')
    plt.title('Inflation vs Federal Funds Rate Over Time')
    plt.legend()
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to make room for x-axis labels
    plt.savefig('line_plot.png')
    plt.show()



def plot_inflation_vs_rate_scatter(merged_df):
    # Scatter plot for inflation vs federal funds rate
    plt.figure(figsize=(8, 6))
    plt.scatter(merged_df['inflation'], merged_df['rate'], color='purple')
    plt.xlabel('Inflation')
    plt.ylabel('Federal Funds Rate')
    plt.title('Inflation vs Federal Funds Rate')
    plt.grid(True)
    plt.savefig('scatter_plot.png')
    plt.show()

# Histogram for inflation and federal funds rate
def histogram_inflation_rate(merged_df):
    plt.figure(figsize=(6, 3))
    plt.hist(merged_df['inflation'], bins=30, color='blue', alpha=0.7)
    plt.xlabel('Inflation')
    plt.ylabel('Frequency')
    plt.title('Histogram of Inflation')
    plt.grid(True)
    plt.savefig('inflation_histogram.png')
    plt.show()

    # Histogram for federal funds rate
def histogram_rate(merged_df):
    plt.figure(figsize=(6, 3))
    plt.hist(merged_df['rate'], bins=30, color='red', alpha=0.7)
    plt.xlabel('Federal Funds Rate')
    plt.ylabel('Frequency')
    plt.title('Histogram of Federal Funds Rate')
    plt.grid(True)
    plt.savefig('rate_histogram.png')
    plt.show()

# Create a figure and a set of subplots
def plot_inflation_rate_money_supply(dataset_2010_2023):
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot inflation and interest rate on the left y-axis
    ax1.plot(dataset_2010_2023['DATE'], dataset_2010_2023['inflation'], color='b', label='Inflation')
    ax1.plot(dataset_2010_2023['DATE'], dataset_2010_2023['rate'], color='r', label='Interest Rate')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Inflation and Interest Rate', color='k')
    ax1.tick_params(axis='y', labelcolor='k')

    # Create a second y-axis for the money supply
    ax2 = ax1.twinx()
    ax2.plot(dataset_2010_2023['DATE'], dataset_2010_2023['money_supply_pct_change'], color='g', label='Money Supply Change')
    ax2.set_ylabel('Money Supply Change (%)', color='k')
    ax2.tick_params(axis='y', labelcolor='k')

    # Add a legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    # Show the plot
    plt.title('Inflation, Interest Rate, and Money Supply Change (2010-2023)')
    plt.show()


