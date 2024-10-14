import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from tqdm import tqdm


################################################## Controler Class ########################################################
class ASTWC():
    def __init__(self, time, Te, reference=None) -> None:
        self.n = int(time / Te)
        self.times = np.linspace(0, time, self.n)
        self.Te = self.times[1] - self.times[0]
        self.time = time
        
        if reference is None:
            self.y_ref = np.zeros(self.n + 1)
            self.y_ref_dot = np.zeros(self.n)
        else :
            self.y_ref = reference
            self.y_ref_dot = np.gradient(self.y_ref, self.Te) 
        
        
        # sliding variable
        self.c1 = 1
        self.s = np.zeros(self.n)
        
        # gain
        self.k = np.zeros(self.n + 1)
        self.k_dot = np.zeros(self.n)
        
        self.alpha = 100 
        self.alpha_star = 3
        
        self.epsilon = 0.02 # 0.02
        
        # intialize gains
        self.k[0] = 0.1
        
        # state variables
        self.x1 = np.zeros(self.n + 1)
        self.x2 = np.zeros(self.n + 1)
        
        # system output
        self.y = np.zeros(self.n)
        
        # error
        self.e = np.zeros(self.n)
        self.e_dot = np.zeros(self.n)
        
        # control input
        self.u = np.zeros(self.n)
        self.v_dot = np.zeros(self.n)
        
    # class methods
    def update_output(self, i):
        self.y[i] = self.x1[i]
        
    def compute_error(self, i):
        self.e[i] = self.y[i] - self.y_ref[i]
        self.e_dot[i] = self.x2[i] - self.y_ref_dot[i]
            
    def update_sliding_variable(self, i):
        self.s[i] = self.e_dot[i] + self.c1 * self.e[i]
        
    def update_gains(self, i):
        if np.abs(self.s[i]) <= self.epsilon:
            self.k_dot[i] = - self.alpha_star * self.k[i]
        else:
            self.k_dot[i] = self.alpha / np.sqrt(np.abs(self.s[i]))
            
        self.k[i + 1] = self.k[i] + self.k_dot[i] * self.Te
          
    def STWC(self, i):
        self.v_dot[i] = - self.k[i + 1] * np.sign(self.s[i])
        self.u[i] = - self.k[i + 1] * np.sqrt(abs(self.s[i])) * np.sign(self.s[i]) + integrate.simpson(self.v_dot[:i + 1], dx=self.Te)
         
    def compute_input(self, i):
        self.update_output(i)
        self.compute_error(i)
        self.update_sliding_variable(i)
        self.update_gains(i)
        self.STWC(i)

class RBF_neural_network():
    def __init__(self, time, Te) -> None:
        self.neurones = 50
        self.n = int(time / Te)
        self.times = np.linspace(0, time, self.n)
        self.Te = self.times[1] - self.times[0]
        
        # centers 
        self.c = 0.2 * np.random.rand(self.neurones) - 0.1
        
        self.pi = 0.5
        self.gamma = 0.075
        
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
            self.hidden_neurons[j] = np.exp(-np.square(s - self.c[j]) / (2 * self.pi**2))

    def compute_weights(self, i, k, s, epsilon):
        self.weights_dot[i] = self.gamma * k * np.sign(s) * self.hidden_neurons / (epsilon + np.sqrt(np.abs(s)))
        
        self.weights[i + 1] = self.weights[i] + self.weights_dot[i] * self.Te

    def compute_perturbation(self, i):
        self.perturbation_dot[i] = self.hidden_neurons @ self.weights[i + 1]
        self.perturbation[i + 1] = self.perturbation[i] + self.perturbation_dot[i] * self.Te
        
        return self.perturbation[i + 1]
    
class NN_based_STWC(RBF_neural_network):
    def __init__(self, controler, Te, time) -> None:
        self.controler = controler
        super().__init__(Te, time)
        
        self.u = np.zeros(self.n)
        
    def compute_input(self, i):
        self.controler.compute_input(i)
        u_ASTWC =  self.controler.u[i]
        
        self.compute_hidden_layer(i, self.controler.s[i])
        self.compute_weights(i, self.controler.k[i], self.controler.s[i], self.controler.epsilon)
        perturbation = self.compute_perturbation(i)
        
        self.u[i] = u_ASTWC - perturbation
     
################################################## Dynamic System ########################################################
class pendule():
    def __init__(self, times) -> None:
        self.g = 9.81
        self.m = 2 
        
        self.Te = times[1] - times[0]
        
        self.longueur_pendule = self.compute_longueur_pendule(times)
        self.longueur_pendule_dot = np.gradient(self.longueur_pendule, self.Te)
        
    # Pendulum model
    def compute_longueur_pendule(self, t):
        return 0.8 + 0.1 * np.sin(8 * t) + 0.3 * np.cos(4 * t)

    def pendule(self, x1, x2, u, t):
        i = int(t / self.Te)
        l = self.longueur_pendule[i]
        l_dot = self.longueur_pendule_dot[i]
        
        x1_dot = x2
        x2_dot = -2 * l_dot / l - self.g / l * np.sin(x1) + 2 * np.sin(5 * t) + (1 + 0.5 * np.sin(t)) / (self.m * l**2) * u
        
        return x1_dot, x2_dot
    
    def compute_dynamics(self, x1, x2, u, t):
        return self.pendule(x1, x2, u, t)
    
    def get_perturbation(self, times, x1, x2):
        a = -2 * self.longueur_pendule_dot / self.longueur_pendule - self.g / self.longueur_pendule * np.sin(x1[:-1]) + 2 * np.sin(5 * times)
        b = ((1 + 0.5 * np.sin(times)) / (self.m * self.longueur_pendule**2))
        perturbation = a / b
        
        return perturbation
        
class basic_system():
    def __init__(self, times) -> None:
        self.a = 5
        
    def compute_dynamics(self, x1, x2, u, t):
        x1_dot = x2
        x2_dot = u + self.a * np.sin(t)
        return x1_dot, x2_dot
    
    def get_perturbation(self, times, x1, x2):
        return self.a * np.sin(times)

################################################## SIMULATION ########################################################
time = 15
Te = 0.0002
n = int(time / Te) 
y_ref = 10 * np.sin((np.arange(0, n + 1, 1) / (n + 1)) * 2 * np.pi * 4)

# First controler without neural network
controler = ASTWC(time, Te, reference=None)

# seconde controler with neural network
NN_inner_controler = ASTWC(time, Te, reference=None)
NN_controler = NN_based_STWC(NN_inner_controler, time, Te)

# dynamics
system = basic_system(controler.times)
# system = pendule(controler.times)


for i in tqdm(range(n)):
    # SuperTwisting control law
    controler.compute_input(i)
    NN_controler.compute_input(i)
    
    # Update system response using the control input for ASTWC
    u = controler.u[i]
    x1_dot, x2_dot = system.compute_dynamics(
                        controler.x1[i], 
                        controler.x2[i], 
                        u, 
                        controler.times[i]
                    )
    
    # Update states for ASTWC
    controler.x1[i + 1] = controler.x1[i] + x1_dot * controler.Te
    controler.x2[i + 1] = controler.x2[i] + x2_dot * controler.Te
    
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



################################ PLOTTING ################################
fig, axs = plt.subplots(3, 3, num=1, figsize=(8, 6), sharex=True, sharey=False)
(ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = axs.flat

# perturbations
times = np.linspace(0, time, n)

perturbation    = system.get_perturbation(times, controler.x1, controler.x2)
perturbation_nn = system.get_perturbation(times, NN_controler.controler.x1, NN_controler.controler.x2)

# Now compute the derivatives after smoothing
dot_perturbation = np.gradient(perturbation, Te)
dot_perturbation_nn = np.gradient(perturbation_nn, Te) 

d_max    = np.max(np.abs(dot_perturbation))
d_max_nn = np.max(np.abs(dot_perturbation_nn))

#######################################################################
################################ No NN ################################


################################ ax1 ################################
# states

ax1.title.set_text('ASTWC')
ax1.plot(controler.times, controler.x1[:-1], label='x1')
ax1.plot(controler.times, controler.x2[:-1], label='x2')
ax1.plot(controler.times, controler.y_ref[:-1], label='ref')
ax1.set_ylabel('x1, x2')
ax1.legend()

################################ ax2 ################################
# Plot the perturbation and the gain response without NN

ax2.set_title('ASTWC perturbations derivatives')
ax2.plot(times, np.abs(dot_perturbation), label='|d_dot|')
ax2.plot(times, controler.k[:-1], label='k')
ax2.axhline(y=d_max, color='r', linestyle='--', label=f'Delta_d = {d_max:.2f}')
ax2.set_ylabel('perturbation derivative')
ax2.legend()

################################ ax3 ################################
# SLiding variable

ax3.set_title('ASTWC sliding variable')
ax3.plot(controler.times, controler.s, label='s')
ax3.axhline(y=controler.epsilon, color='r', linestyle='--', label=f'epsilon = {controler.epsilon:.2f}')
ax3.axhline(y=-controler.epsilon, color='r', linestyle='--')
ax3.set_ylabel('sliding variable')
ax3.legend()



################################ ax8 ################################
# inputs
ax8.set_title('ASTWC')
ax8.plot(times, controler.u, label='u')
ax8.plot(times, controler.v_dot, label='v_dot')
ax8.set_xlabel('Time')
ax8.set_ylabel('inputs')
ax8.legend()


####################################################################
################################ NN ################################


################################ ax4 ################################
ax4.title.set_text('NN_based_ASTWC')
ax4.plot(NN_controler.controler.times, NN_controler.controler.x1[:-1], label='x1')
ax4.plot(NN_controler.controler.times, NN_controler.controler.x2[:-1], label='x2')
ax4.plot(NN_controler.controler.times, NN_controler.controler.y_ref[:-1], label='ref')
ax4.set_ylabel('x1, x2')
ax4.legend()

################################ ax5 ################################
# Plot the perturbation and the gain response with NN

ax5.set_title('NN_based_ASTWC perturbations derivatives')
ax5.plot(times, np.abs(dot_perturbation_nn), label='|d_dot|')
ax5.plot(times, NN_controler.controler.k[:-1], label='k')
ax5.axhline(y=d_max_nn, color='r', linestyle='--', label=f'Delta_d = {d_max_nn:.2f}')
ax5.set_ylabel('perturbation derivative')
ax5.legend()




################################ ax6 ################################
# Sliding variable

ax6.set_title('NN_based_ASTWC sliding variable')
ax6.plot(NN_controler.controler.times, NN_controler.controler.s, label='s')
ax6.axhline(y=NN_controler.controler.epsilon, color='r', linestyle='--', label=f'epsilon = {NN_controler.controler.epsilon:.2f}')
ax6.axhline(y=-NN_controler.controler.epsilon, color='r', linestyle='--')
ax6.set_ylabel('sliding variable')
ax6.legend()



################################ ax7 ################################
# estimated perturbation by neural network
ax7.set_title('perturbations and NN approximation')
ax7.plot(times, perturbation_nn, label='d')
ax7.plot(times, NN_controler.perturbation[:-1], label='NN d approx')
ax7.set_xlabel('Time')
ax7.set_ylabel('perturbation')
ax7.legend()

################################ ax9 ################################
#inputs
ax9.set_title('NN_based_ASTWC')
ax9.plot(times, NN_controler.controler.u, label='u')
ax9.plot(times, NN_controler.controler.v_dot, label='v_dot')
ax9.set_xlabel('Time')
ax9.set_ylabel('inputs')
ax9.legend()


plt.show()

