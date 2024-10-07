import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from tqdm import tqdm

   
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
        
        self.type = 'ASTWC'
        
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
        self.x1[0] = 0.5
        
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
        self.u[i] = - self.k[i + 1] * np.sqrt(abs(self.s[i])) * np.sign(self.s[i]) + integrate.simpson(self.v_dot[:i + 1])
         
    def compute_input(self, i):
        self.update_output(i)
        self.compute_error(i)
        self.update_sliding_variable(i)
        self.update_gains(i)
        self.STWC(i)
        
        return self.u[i]
    
################################ DYNAMICAL SYSTEM ################################

def system(x1, x2, u):
    a = 0.7
    x1_dot = x2
    x2_dot = u #+ a * np.sin(x1)
    return x1_dot, x2_dot
    
################################ SIMULATION ################################
time = 15
Te = 0.0005
n = int(time / Te) 
y_ref = 10 * np.sin((np.arange(0, n + 1, 1) / (n + 1)) * 2 * np.pi * 4)

controler = ASTWC(time, Te, reference=None)


for i in tqdm(range(n)):
    # SuperTwisting control law
    u = controler.compute_input(i)
    
    # Update system response using the control input for ASTWC
    x1_dot, x2_dot = system(
                        controler.x1[i], 
                        controler.x2[i], 
                        u, 
                    )
    
    # Update states for ASTWC
    controler.x1[i + 1] = controler.x1[i] + x1_dot * controler.Te
    controler.x2[i + 1] = controler.x2[i] + x2_dot * controler.Te
    
# Plotting results
fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharex=False, sharey=False)
((ax1, ax2), (ax3, ax4)) = axs

ax1.plot(controler.times, controler.x1[:-1], label='x1')
ax1.plot(controler.times, controler.x2[:-1], label='x2')
ax1.plot(controler.times, controler.y_ref[:-1], label='ref')
ax1.set_xlabel('Time')
ax1.legend()

ax2.plot(controler.times, controler.s, label='s')
ax2.hlines(controler.epsilon, 0, controler.time, color='r', linestyle='--')
ax2.hlines(-controler.epsilon, 0, controler.time, color='r', linestyle='--')
ax2.set_xlabel('Time')
ax2.legend()

ax3.plot(controler.times, controler.k[:-1], label='k')
ax3.set_xlabel('Time')
ax3.legend()

ax4.plot(controler.times, controler.u, label='u')
ax4.plot(controler.times, controler.v_dot, label='v_dot')
ax4.set_xlabel('Time')
ax4.legend()


plt.show()