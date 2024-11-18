import numpy as np

################################################## Dynamic System ########################################################
class pendule():
    def __init__(self, times) -> None:
        self.g = 9.81
        self.m = 2 
        
        self.name = 'pendule'
        
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
        x2_dot = -2 * l_dot / l - self.g / l * np.sin(x1) + 2 * np.sin(t) + u # (1 + 0.5 * np.sin(t)) / (self.m * l**2) * 
        
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
        self.name = 'basic_system'
        
    def compute_dynamics(self, x1, x2, u, t):
        x1_dot = x2
        x2_dot = u + self.a * np.sin(t)
        return x1_dot, x2_dot
    
    def get_perturbation(self, times, x1, x2):
        return self.a * np.sin(times)
    
class easy_first_order():
    def __init__(self, times) -> None:
        self.a = 5
        self.name = 'easy_first_order'
        
    def compute_dynamics(self, x1, x2, u, t):
        x1_dot = u + self.a * np.sin(t)
        x2_dot = 0
        return x1_dot, x2_dot
    
    def get_perturbation(self, times, x1, x2):
        return self.a * np.sin(times)
    
class ultra_perturbed_system():
    def __init__(self, times) -> None:
        self.a = 50
        self.name = 'ultra_perturbed_system'
        
    def compute_dynamics(self, x1, x2, u, t):
        x1_dot = x2
        x2_dot = u + self.a * np.sin(t) + 7 * np.sin(100 * t) + 20 * x1
        return x1_dot, x2_dot
    
    def get_perturbation(self, times, x1, x2):
        return self.a * np.sin(times) + 7 * np.sin(100 * times) + 20 * x1[:-1]


