import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from tqdm import tqdm

###################################### Controler Class ###########################################
class STWC():
    def __init__(self, time, Te, reference=None) -> None:
        self.n = int(time / Te)
        self.time = time
        
        self.times = np.linspace(0, time, self.n)
        self.Te = self.times[1] - self.times[0]
        
        if reference is None:
            self.y_ref = np.zeros(self.n + 1)
            self.y_ref_dot = np.zeros(self.n)
        else :
            self.y_ref = reference
            self.y_ref_dot = np.gradient(self.y_ref, self.Te) 
        
        # sliding variable
        self.s = np.zeros(self.n)
        
        # gain
        self.k1 = 5
        self.k2 = 10
        
        # state variables
        self.x1 = np.zeros(self.n + 1)
        
        # system output
        self.y = np.zeros(self.n)
        
        # error
        self.e = np.zeros(self.n)
        
        # control input
        self.u = np.zeros(self.n)
        self.v_dot = np.zeros(self.n)

    # class methods
    def update_output(self, i):
        self.y[i] = self.x1[i]
        
    def compute_error(self, i):
        self.e[i] = self.y[i] - self.y_ref[i]
            
    def update_sliding_variable(self, i):
        self.s[i] = self.e[i]
          
    def STWC(self, i):
        self.v_dot[i] = - self.k2 * np.sign(self.s[i])
        self.u[i] = - self.k1 * np.sqrt(abs(self.s[i])) * np.sign(self.s[i]) + integrate.simpson(self.v_dot[:i + 1], dx=self.Te)
         
    def compute_input(self, i):
        self.update_output(i)
        self.compute_error(i)
        self.update_sliding_variable(i)
        self.STWC(i)
        
        return self.u[i]

###################################### Dynamic System ############################################

def system(u, t):
    d = 5 * np.sin(t)
    x1_dot = d + u
    return x1_dot


################################ SIMULATION ################################
time = 10
Te = 0.0001
n = int(time / Te) 
y_ref = 10 * np.sin((np.arange(0, n + 1, 1) / (n + 1)) * 2 * np.pi * 4)

controler = STWC(time, Te, reference=None)

for i in tqdm(range(n)):
    # SuperTwisting control law
    u = controler.compute_input(i)
    
    # Update system response using the control input for ASTWC
    x1_dot = system(u, controler.times[i])
    
    # Update states for ASTWC
    controler.x1[i + 1] = controler.x1[i] + x1_dot * controler.Te
    # print('-----------------------------------')
    # print(f'loop {i} done')
    # print('debug inforamtions : ')
    # print(f'x1 = {controler.x1[i]} | x1_dot = {x1_dot} | u = {u} | s = {controler.s[i]}')
    # print('-----------------------------------')
    # print(f'e = {controler.e[i]} | v_dot = {controler.v_dot[i]} | y = {controler.y[i]} | y_ref = {controler.y_ref[i]}')
    # print('-----------------------------------')
    # input('Press Enter to continue...')
    
# Plotting results
fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=False)
((ax1, ax2), (ax3, ax4)) = axs

ax1.plot(controler.times, controler.x1[:-1], label='x1')
ax1.plot(controler.times, controler.y_ref[:-1], label='ref')
ax1.set_xlabel('Time')
ax1.legend()

ax2.plot(controler.times, controler.s, label='s')
ax2.set_xlabel('Time')
ax2.legend()

perturbation_dot = 5 * np.cos(controler.times)

d_max = np.max(np.abs(perturbation_dot))

# ax3.plot(controler.times, np.abs(perturbation_dot), label='|d|')
# ax3.axhline(d_max, color='r', linestyle='--')
ax3.plot(controler.times, np.gradient(controler.s, controler.Te), label='s_dot')
ax3.set_xlabel('Time')
ax3.legend()

ax4.plot(controler.times, controler.u, label='u')
ax4.plot(controler.times, controler.v_dot, label='v_dot')
ax4.set_xlabel('Time')
ax4.legend()


plt.show()