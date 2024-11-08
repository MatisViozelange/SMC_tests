import numpy as np
import pandas as pd
from tqdm import tqdm
from modules import ASTWC, NN_based_STWC, basic_system, pendule

################################################## SIMULATION ########################################################
time = 5
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
gamma_tests     = np.arange(0.005, 0.5, 0.005)
n_neurons_tests = np.arange(2, 101, 2)

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

print(f'Saving performance DataFrame to results_{system.name}.csv ...')

# Save the performance DataFrame to a CSV file
performance.to_csv(f'results_{system.name}.csv', index=False)

print(f'Performance DataFrame saved to results_{system.name}.csv')