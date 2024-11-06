import numpy as np
import pandas as pd
from tqdm import tqdm
from controllers import basic_system, pendule

class MVCModel:
    def __init__(self, time, Te):
        self.time = time
        self.Te = Te
        self.n = int(time / Te)
        self.times = np.linspace(0, time, self.n)
        self.systems = {'Basic System': basic_system, 'Pendulum': pendule}
        self.current_system = None
        self.controlers = {}
        self.results_df = None

    def set_system(self, system_name):
        if system_name in self.systems:
            self.current_system = self.systems[system_name](self.times)
        else:
            raise ValueError("System not found")

    def add_controler(self, controler_name, controler):
        self.controlers[controler_name] = controler

    def run_simulation(self):
        for i in tqdm(range(self.n)):
            for controler in self.controlers.values():
                controler.compute_input(i)
                if hasattr(controler, 'controler'):
                    inner_controler = controler.controler
                else:
                    inner_controler = controler
                u = controler.u[i]
                x1_dot, x2_dot = self.current_system.compute_dynamics(inner_controler.x1[i], inner_controler.x2[i], u, self.times[i])
                inner_controler.x1[i + 1] = inner_controler.x1[i] + x1_dot * self.Te
                inner_controler.x2[i + 1] = inner_controler.x2[i] + x2_dot * self.Te

    def compute_performance(self):
        real_perturbation = self.current_system.get_perturbation(self.times, self.controlers['ASTWC'].x1, self.controlers['ASTWC'].x2)
        real_perturbation_nn = self.current_system.get_perturbation(self.times, self.controlers['NN_based_ASTWC'].controler.x1, self.controlers['NN_based_ASTWC'].controler.x2)
        
        NN_perturbation_approximation_BO = self.controlers['NN_BO_Observator'].perturbation[:-1]
        NN_perturbation_approximation = self.controlers['NN_based_ASTWC'].perturbation[:-1]

        error_bf = np.sum((NN_perturbation_approximation - real_perturbation_nn) ** 2)
        mean_error_bf = np.mean(NN_perturbation_approximation - real_perturbation_nn)
        std_deviation_bf = np.sqrt(np.mean((NN_perturbation_approximation - real_perturbation_nn - mean_error_bf) ** 2))

        error_bo = np.sum((NN_perturbation_approximation_BO - real_perturbation) ** 2)
        mean_error_bo = np.mean(NN_perturbation_approximation_BO - real_perturbation)
        std_deviation_bo = np.sqrt(np.mean((NN_perturbation_approximation_BO - real_perturbation - mean_error_bo) ** 2))

        data = {
            'Metric': ['Quadratic Error', 'Mean Error', 'Standard Deviation'],
            'BF Estimation': [error_bf, mean_error_bf, std_deviation_bf],
            'BO Estimation': [error_bo, mean_error_bo, std_deviation_bo]
        }
        self.results_df = pd.DataFrame(data)

