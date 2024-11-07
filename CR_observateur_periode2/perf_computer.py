import numpy as np
import pandas as pd
import tkinter as tk
from tqdm import tqdm
from tkinter import ttk
import matplotlib.pyplot as plt
from modules import ASTWC, NN_based_STWC, basic_system, pendule

################################################## SIMULATION ########################################################
time = 10
Te = 0.001
n = int(time / Te) 
times = np.linspace(0, time, n)
y_ref = 10 * np.sin(times)

# dynamics
system = basic_system(times)
# system = pendule(times)

# Parameters Overview ####################################################
#       gamma : adaptation protocole parameter (learning rate)           #
#               eta : kernel width (to check later)                      #
#                       n : number of neurons                            #
#                           Te : sampling time                           #
##########################################################################


# Performance Indices ####################################################
#     standard deviation of the perturbation approximation               #
#     mean quadratic error of perturbation apporximation                 #
#     correlation coefficient of the perturbation approximation          #
##########################################################################

# Simulation
gamma_tests     = np.arange(0.025, 1, 0.025)
n_neurons_tests = np.arange(10, 110, 10)

# Performance indices
performance = pd.DataFrame(columns=['gamma', 
                                    'n_neurons', 
                                    'mean_quad_perturbation_approximation_error', 
                                    'perturbation_approximation_std_deviation',
                                    'perturbation_approximation_correlation_coefficient'
                            ])

for n_neurons in tqdm(n_neurons_tests):
    for gamma in tqdm(gamma_tests):
        # seconde controler with neural network
        NN_inner_controler = ASTWC(time, Te, reference=None)
        NN_controler = NN_based_STWC(NN_inner_controler, time, Te, neurones=n_neurons, gamma=gamma)
        
        # simulation
        for i in range(n):
            NN_controler.compute_input(i)
            
            # Update system response using the control input for NN_based_ASTWC
            u = NN_controler.u[i]
            x1_dot, x2_dot = system.compute_dynamics(
                                NN_controler.controler.x1[i], 
                                NN_controler.controler.x2[i], 
                                u, 
                                NN_controler.controler.times[i]
                            )
            
            # Update states for NN_based_ASTWC
            NN_controler.controler.x1[i + 1] = NN_controler.controler.x1[i] + x1_dot * NN_controler.controler.Te
            NN_controler.controler.x2[i + 1] = NN_controler.controler.x2[i] + x2_dot * NN_controler.controler.Te

        # perturbations
        real_perturbation_nn = system.get_perturbation(times, NN_controler.controler.x1, NN_controler.controler.x2)
        NN_perturbation_approximation = NN_controler.perturbation[:-1]
        
        # index performance
        mean_quadratic_error = np.mean((NN_perturbation_approximation - real_perturbation_nn) ** 2)
        mean_error = np.mean(NN_perturbation_approximation - real_perturbation_nn)
        std_deviation = np.sqrt(np.mean((NN_perturbation_approximation - real_perturbation_nn - mean_error) ** 2))

        # correlation coefficient
        correlation_coefficient = np.corrcoef(NN_perturbation_approximation, real_perturbation_nn)[0, 1]
        
        # add the performance values into the dataframe
        performance = pd.concat([performance, pd.DataFrame([{
            'gamma': gamma, 
            'n_neurons': n_neurons, 
            'mean_quad_perturbation_approximation_error': mean_quadratic_error, 
            'perturbation_approximation_std_deviation': std_deviation,
            'perturbation_approximation_correlation_coefficient': correlation_coefficient
        }], dtype=object)], ignore_index=True)

# Save the performance DataFrame to a CSV file
performance.to_csv(f'results_{system.name}.csv', index=False)

# Tkinter Window for displaying the DataFrame
def show_dataframe(df):
    root = tk.Tk()
    root.title("Performance DataFrame")

    frame = ttk.Frame(root)
    frame.pack(fill="both", expand=True)

    tree = ttk.Treeview(frame, columns=list(df.columns), show="headings")
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center")

    for _, row in df.iterrows():
        tree.insert("", "end", values=list(row))

    tree.pack(fill="both", expand=True)
    root.mainloop()

# Plotting errors as function of neurons and gamma
def plot_errors_vs_neurons(df):
    unique_gammas = df['gamma'].unique()
    plt.figure()
    for gamma in unique_gammas:
        subset = df[df['gamma'] == gamma]
        plt.plot(subset['n_neurons'], subset['mean_quad_perturbation_approximation_error'], label=f'gamma={gamma:.3f}')
    plt.xlabel('Number of Neurons')
    plt.ylabel('Mean Quadratic Perturbation Approximation Error')
    plt.title('Error vs Neurons for Different Gamma Values')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_errors_vs_gamma(df):
    unique_neurons = df['n_neurons'].unique()
    plt.figure()
    for n_neurons in unique_neurons:
        subset = df[df['n_neurons'] == n_neurons]
        plt.plot(subset['gamma'], subset['mean_quad_perturbation_approximation_error'], label=f'n_neurons={n_neurons}')
    plt.xlabel('Gamma')
    plt.ylabel('Mean Quadratic Perturbation Approximation Error')
    plt.title('Error vs Gamma for Different Number of Neurons')
    plt.legend()
    plt.grid(True)
    plt.show()

# Display the DataFrame
show_dataframe(performance)

# Plot the errors
plot_errors_vs_neurons(performance)
plot_errors_vs_gamma(performance)
