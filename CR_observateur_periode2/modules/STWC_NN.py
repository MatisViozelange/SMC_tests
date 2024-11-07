import numpy as np
from scipy import integrate




################################################## Controler Class ########################################################
class ASTWC():
    def __init__(self, time, Te, reference=None) -> None:
        self.n = int(time / Te)
        self.times = np.linspace(0, time, self.n)
        self.Te = self.times[1] - self.times[0]
        self.time = time
        
        self.y_ref = np.zeros(self.n) if reference is None else reference
        self.y_ref_dot = np.gradient(self.y_ref, self.Te) if reference is not None else np.zeros(self.n)
        
        self.c1 = 1
        self.s = np.zeros(self.n)
        self.k = np.zeros(self.n + 1)
        self.k_dot = np.zeros(self.n)
        
        self.alpha = 100 
        self.alpha_star = 3
        self.epsilon = 0.02
        self.k[0] = 0.1
        
        self.x1 = np.zeros(self.n + 1)
        self.x2 = np.zeros(self.n + 1)
        
        self.y = np.zeros(self.n)
        self.e = np.zeros(self.n)
        self.e_dot = np.zeros(self.n)
        
        self.u = np.zeros(self.n)
        self.v_dot = np.zeros(self.n)

    def compute_input(self, i):
        self.y[i] = self.x1[i]
        self.e[i] = self.y[i] - self.y_ref[i]
        self.e_dot[i] = self.x2[i] - self.y_ref_dot[i]
        
        self.s[i] = self.e_dot[i] + self.c1 * self.e[i]
        
        if np.abs(self.s[i]) <= self.epsilon:
            self.k_dot[i] = -self.alpha_star * self.k[i]
        else:
            self.k_dot[i] = self.alpha / np.sqrt(np.abs(self.s[i]))
            
        self.k[i + 1] = self.k[i] + self.k_dot[i] * self.Te
        self.v_dot[i] = -self.k[i + 1] * np.sign(self.s[i])
        self.u[i] = -self.k[i + 1] * np.sqrt(abs(self.s[i])) * np.sign(self.s[i]) + integrate.simpson(self.v_dot[:i + 1], dx=self.Te)



class RBF_neural_network():
    def __init__(self, time, Te, neurones=50, gamma=0.075) -> None:
        self.neurones = neurones
        self.n = int(time / Te)
        self.times = np.linspace(0, time, self.n)
        self.Te = self.times[1] - self.times[0]
        
        # centers 
        self.c = 0.2 * np.random.rand(self.neurones) - 0.1
        
        self.eta = 0.5
        self.gamma = gamma
        
        self.initial_weights = 2 * 4.12 * np.random.rand(self.neurones) - 4.12
        
        self.weights = np.zeros((self.n + 1, self.neurones))
        self.weights[0] = self.initial_weights
        self.weights_dot = np.zeros((self.n, self.neurones))
        self.hidden_neurons = np.zeros(self.neurones)
        
        # perturbation
        self.perturbation = np.zeros(self.n + 1)
        self.perturbation_dot = np.zeros(self.n)
        
    def compute_hidden_layer(self, i, s):
        for j in range(self.neurones):
            # Gaussian kernel
            self.hidden_neurons[j] = np.exp(-np.square(s - self.c[j]) / (2 * self.eta**2))
            # sigmoid kernel
            # self.hidden_neurons[j] = 1 / (1 + np.exp(-self.eta * (s - self.c[j])))
            # tanh kernel
            # self.hidden_neurons[j] = np.tanh(self.eta * (s - self.c[j]))

    def compute_weights(self, i, k, s, epsilon):
        self.weights_dot[i] = self.gamma * k * np.sign(s) * self.hidden_neurons / (epsilon + np.sqrt(np.abs(s)))
        
        self.weights[i + 1] = self.weights[i] + self.weights_dot[i] * self.Te

    def compute_perturbation(self, i):
        self.perturbation_dot[i] = self.hidden_neurons @ self.weights[i + 1]
        self.perturbation[i + 1] = self.perturbation[i] + self.perturbation_dot[i] * self.Te
        
        return self.perturbation[i + 1]
    
class NN_based_STWC(RBF_neural_network):
    def __init__(self, controler, Te, time, neurones=50, gamma=0.075) -> None:
        self.controler = controler
        super().__init__(Te, time, neurones=neurones, gamma=gamma)
        
        self.u = np.zeros(self.n)
        
    def compute_input(self, i):
        self.controler.compute_input(i)
        u_ASTWC =  self.controler.u[i]
        
        self.compute_hidden_layer(i, self.controler.s[i])
        self.compute_weights(i, self.controler.k[i], self.controler.s[i], self.controler.epsilon)
        perturbation = self.compute_perturbation(i)
        
        self.u[i] = u_ASTWC - perturbation 
     
