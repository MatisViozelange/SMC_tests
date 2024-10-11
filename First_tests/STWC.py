import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

class STWC():
    def __init__(self, tau, Te, system, adaptatif=True,
                                        self_tuning=True,
                                        auto_tuning=True,
                                        reference=None, 
                                        multiplicative_perturbation=None) -> None:
        # time sampling period
        self.Te = Te
        
        # differenciator time constant
        self.tau = tau
        
        # dynamical system
        self.system = system
        
        # adaptative control
        self.adaptatif = adaptatif
        
        # reference signal
        self.reference = reference
        
        # The multiplicative_perturbation is a function of time
        if multiplicative_perturbation is not None:
            self.multiplicative_perturbation = multiplicative_perturbation
        
        
    # class methods
    def update_output(self):
        self.y[self.i] = self.x1[self.i]
        
    def compute_error(self):
        self.e[self.i] = self.y[self.i] - self.y_ref[self.i]
        self.e_dot[self.i] = self.x2[self.i] - self.y_ref_dot[self.i]
            
    def update_sliding_variable(self):
        self.s[self.i] = self.e_dot[self.i] + self.c1 * self.e[self.i]
    
    def update_observator(self,x1_dot=0):
        
        psi_dot = 1 / self.tau * (-x1_dot - self.psi[self.i])
        self.psi[self.i + 1] = self.psi[self.i] + psi_dot * self.Te
        
    def update_gains(self):
        #compute alpha 
        self.r_dot[self.i] = self.k2[self.i] * np.sign(self.s[self.i])
        self.alpha[self.i] = self.k1[self.i] * np.sqrt(abs(self.s[self.i])) + abs(self.psi[self.i]) + abs(integrate.simpson(self.r_dot[:self.i + 1]))
        
        # update epsilon
        if self.i > 0:
            self.epsilon[self.i] = (self.alpha[self.i] + abs(self.psi[self.i]) + abs(integrate.simpson(self.r_dot[:self.i]))) * self.Te + self.k2[self.i] * self.Te**2
        else:
            self.epsilon[self.i] = (self.alpha[self.i] + abs(self.psi[self.i])) * self.Te + self.k2[self.i] * self.Te**2
        
        # update gains
        self.k1_dot[self.i] = self.alpha[self.i] / (abs(self.psi[self.i]) + self.epsilon[self.i]) if abs(self.s[self.i]) > self.epsilon[self.i] else -self.k1[self.i]
        self.k2_dot[self.i] = self.alpha[self.i] / (2 * np.sqrt(abs(self.s[self.i]))) if abs(self.s[self.i]) > self.epsilon[self.i] else -self.k2[self.i]
        
        self.k1[self.i + 1] = self.k1[self.i] + self.k1_dot[self.i] * self.Te
        self.k2[self.i + 1] = self.k2[self.i] + self.k2_dot[self.i] * self.Te
        
        
    def STWC(self, time):
        # initialize control
        self.n = int(time / Te)
        self.times = np.linspace(0, time, self.n)
        
        if self.reference is None:
            self.y_ref = np.zeros(self.n)
            self.y_ref_dot = np.zeros(self.n)
        else :
            self.y_ref = self.reference
            self.y_ref_dot = np.diff(self.y_ref) / self.Te
   
        # sliding variable
        self.c1 = 0.5
        self.s = np.zeros(self.n)
        # observator of derivative of sliding variable
        self.psi = np.zeros(self.n)
        
        # gain
        self.k1 = np.zeros(self.n)
        self.k1_dot = np.zeros(self.n)
        self.k2 = np.zeros(self.n)
        self.k2_dot = np.zeros(self.n)
        
        self.alpha = np.zeros(self.n)
        self.r_dot = np.zeros(self.n)
        
        self.epsilon = np.zeros(self.n)
        
        # intialize gains
        self.k1[0] = 20
        self.k2[0] = 10
        
        # state variables
        self.x1 = np.zeros(self.n)
        self.x2 = np.zeros(self.n)
        
        # system output
        self.y = np.zeros(self.n)
        
        # error
        self.e = np.zeros(self.n)
        self.e_dot = np.zeros(self.n)
        
        # control input
        self.u = np.zeros(self.n)
        self.w = np.zeros(self.n)
        self.v_dot = np.zeros(self.n)
        
        # discret time
        self.i = 0
        self.time = self.times[self.i]
        
        # tune is adative control is activated
        if self.adaptatif:
            print('Adaptative control is activated')
            
        else:
            print('Adaptative control is not activated')
            print('choose the value of the control gains')
            input('k1 = ')
            input('k2 = ')
            
        
                
        self.v_dot[self.i] = - self.k2[self.i + 1] * np.sign(self.s[self.i])
        self.w[self.i] = - self.k1[self.i + 1] * np.sqrt(abs(self.s[self.i])) * np.sign(self.s[self.i]) + integrate.simpson(self.v_dot[:self.i + 1])
        
        if self.multiplicative_perturbation is not None:
            self.u[self.i] = 1 / multiplicative_perturbation(self.time) * self.w[self.i]
        else:
            self.u[self.i] = self.w[self.i]
         
        return self.u[self.i]
    
        


# Pendulum model
def longueur_pendule(t):
    return 0.8 + 0.1 * np.sin(8 * t) + 0.3 * np.cos(4 * t)

def pendule(x1, x2, u, t, dt):
    l = longueur_pendule(t)
    l_h = longueur_pendule(t - dt)
    l_dot = (l - l_h) / dt
    
    g = 9.81
    m = 2
    
    x1_dot = x2
    x2_dot = -2 * l_dot / l - g / l * np.sin(x1) + 2 * np.sin(5 * t) + (1 + 0.5 * np.sin(t)) / (m * l**2) * u
    
    return x1_dot, x2_dot

def additive_perturbation(t):
    l = longueur_pendule(t)
    l_dot = (l - longueur_pendule(t - 0.01)) / 0.01
    return -2 * l_dot / l - 9.81 / l * np.sin(t) + 2 * np.sin(5 * t)

def multiplicative_perturbation(t):
    m = 2
    l = longueur_pendule(t)
    return (1 + 0.5 * np.sin(t)) / (m * l**2)

def reference_signal(t):
    return 10 * np.sin(2 * np.pi * 4 * t)
    
time = 20
Te = 0.0005
tau = 0.01
n = int(time / Te)
y_ref = reference_signal(np.linspace(0, time, n))

controler = STWC(time, tau, Te, adaptatif=True, reference=reference_signal, multiplicative_perturbation=multiplicative_perturbation)

x1_dot = 0

# Main simulation loop
for i in range(n - 1):
    # System output
    controler.update_output(i)
    
    # Compute error and its derivative
    controler.compute_error(i)
    
    # Compute sliding variable and it's derivative with an observator
    controler.update_sliding_variable(i)
    controler.update_observator(i, x1_dot) 
    
    # Update gains
    controler.update_gains(i)
    
    # SuperTwisting control law
    u = controler.STWC(i)
    
    # Update system response using the control input
    x1_dot, x2_dot = pendule(controler.x1[i], controler.x2[i], u, controler.times[i], Te)
    
    # Update states
    controler.x1[i + 1] = controler.x1[i] + x1_dot * controler.Te
    controler.x2[i + 1] = controler.x2[i] + x2_dot * controler.Te

# Plotting results
fig, axs = plt.subplots(2, 3, num=1, figsize=(8, 6), sharex=True, sharey=False)
(ax1, ax2, ax3, ax4, ax5, ax6) = axs.flat

ax1.plot(controler.times, controler.x1, label='x1')
ax1.plot(controler.times, controler.x2, label='x2')
ax1.plot(controler.times, controler.y_ref, label='ref')
ax1.set_xlabel('Time')
ax1.set_ylabel('x1, x2')
ax1.legend()

# Plot phase portrait
ax2.plot(controler.times, controler.s, label='s')
ax2.set_xlabel('Time')
ax2.set_ylabel('sliding variable')
ax2.legend()

ax3.plot(controler.times, controler.alpha, label='aplha')
ax3.plot(controler.times, controler.epsilon, label='epsilon')
ax3.plot(controler.times, controler.k1[:-1], label='k1')
ax3.plot(controler.times, controler.k2[:-1], label='k2')
ax3.set_xlabel('Time')
ax3.set_ylabel('adaptative gain')
ax3.legend()

ax4.plot(controler.times, controler.u, label='u')
ax4.plot(controler.times, controler.v_dot, label='v_dot')
ax4.set_xlabel('Time')
ax4.set_ylabel('inputs')
ax4.legend()


# perturbations
times = np.linspace(0, time, n)
l = longueur_pendule(times)
l_dot = np.diff(l) / Te

perturbation = -2 * l_dot / l[:-1] - 9.81 / l[:-1] * np.sin(controler.x1[:-1]) + 2 * np.sin(5 * times[:-1]) 

# Plot the maximum value of the perturbation as a horizontal line
max_perturbation = np.max(np.abs(perturbation))

ax5.plot(times[:-1], perturbation, label='a')
ax5.axhline(y=max_perturbation, color='r', linestyle='--', label=f'Max perturbation = {max_perturbation:.2f}')
ax5.axhline(y=-max_perturbation, color='r', linestyle='--')
ax5.set_xlabel('Time')
ax5.set_ylabel('perturbation')
ax5.legend()


# Compute the derivative of the perturbation
dot_perturbation = np.diff(perturbation) / Te

# Compute the maximum absolute value of the derivative of the perturbation
max_dot_perturbation = np.max(np.abs(dot_perturbation))

# Plot the derivative of the perturbation on ax6
ax6.plot(times[:-2], dot_perturbation, label='a_dot')
ax6.axhline(y=max_dot_perturbation, color='g', linestyle='--', label=f'Delta_d = {max_dot_perturbation:.2f}')
ax6.axhline(y=-max_dot_perturbation, color='g', linestyle='--')
ax6.set_xlabel('Time')
ax6.set_ylabel('perturbation derivative')
ax6.legend()

# plot delta_d on ax3
ax3.axhline(y=max_dot_perturbation, color='g', linestyle='--', label=f'Delta_d')

plt.show()


