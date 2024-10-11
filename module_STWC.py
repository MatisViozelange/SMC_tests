import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from tqdm import tqdm

###################################### Controler Class ###########################################
class STWC():
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
        self.s = np.zeros(self.n)
        
        # gain
        self.k1 = 5
        self.k2 = 10
        
        # state variables
        self.x1 = np.zeros(self.n + 1)
        self.x2 = np.zeros(self.n + 1)
        
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
        self.v_dot[i] = - self.k[i] / 2 * np.sign(self.s[i])
        self.u[i] = - self.k[i] * np.sqrt(abs(self.s[i])) * np.sign(self.s[i]) + integrate.simpson(self.v_dot[:i + 1])
         
    def compute_input(self, i):
        self.update_output(i)
        self.compute_error(i)
        self.update_sliding_variable(i)
        self.update_gains(i)
        self.STWC(i)
        
        return self.u[i]

###################################### Dynamic System ############################################