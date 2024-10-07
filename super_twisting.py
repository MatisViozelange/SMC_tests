import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from tqdm import tqdm   

def system(x1, x2, u):
    a = 0.7
    x1_dot = x2
    x2_dot = u + a * np.sin(x1)
    return x1_dot, x2_dot

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


def multiplicative_perturbation(t):
    m = 2
    l = longueur_pendule(t)
    return (1 + 0.5 * np.sin(t)) / (m * l**2)

def d(x, i, dt):
    if i < 1:
        dx = 0
    else:
        dx = (x[i] - x[i - 1]) / dt
    return dx

   
time = 20
Te = 0.0001
n = int(time / Te)
y_ref = 10 * np.sin((np.arange(0, n, 1) / n) * 2 * np.pi * 4)

times = np.linspace(0, 20, n)
dt = Te
delta_d = 300.48
k1 = 25 # delta_d
k2 = 1.4 * np.sqrt(delta_d + k1)

print('k2: ', k2)

c1 = 1

s = np.zeros(n)
s_dot = np.zeros(n)

x1 = np.zeros(n)
x2 = np.zeros(n)

x1[0] = 0.5

y = x1
y_ref = np.zeros(n) # 10 * np.sin((np.arange(0, n, 1) / n) * 2 * np.pi * 4)
y_ref_dot = np.gradient(y_ref, dt)
e = np.zeros(n)
e_dot = np.zeros(n)

u = np.zeros(n)
v_dot = np.zeros(n)


# Main simulation loop
for i in tqdm(range(n)):
    # System output
    y[i] = x1[i]
    
    # Compute error and its derivative
    e[i] = y[i] #- y_ref[i]
    e_dot[i] = x2[i] #- y_ref_dot[i]
    
    # Compute sliding variable
    s[i] = e_dot[i] + c1 * e[i]
    s_dot[i] = d(s, i, dt)
    
    # SuperTwisting control law
    v_dot[i] = - k2 * np.sign(s[i])
    u[i] = -k1 * np.sqrt(abs(s[i])) * np.sign(s[i]) + integrate.simpson(v_dot[:i + 1])
    
    
    # Update system response using the control input
    # x1_dot, x2_dot = pendule(x1[i], x2[i], u[i], times[i], dt)
    x1_dot, x2_dot = system(x1[i], x2[i], u[i])
    
    # Update states
    if i < n - 1:
        x1[i + 1] = x1[i] + x1_dot * dt
        x2[i + 1] = x2[i] + x2_dot * dt

# Plotting results
fig, axs = plt.subplots(2, 2, num=1, figsize=(8, 6), sharex=False, sharey=False)
(ax1, ax2), (ax3, ax4) = axs 

ax1.plot(times, x1, label='x1')
ax1.plot(times, x2, label='x2')
ax1.plot(times, y_ref, label='ref')
# ax1.plot(times, y_ref_dot, label='dy_ref')
ax1.set_xlabel('Time')
ax1.set_ylabel('x1, x2')
ax1.legend()

ax2.plot(x1, x2, label='Phase diagram')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.legend()

ax3.plot(times, s, label='s')
# ax3.plot(times, s_dot, label='s_dot')
ax3.set_xlabel('Time')
ax3.set_ylabel('sliding variable')
ax3.legend()

total_chattering = np.sum(np.abs(s))
print('Total chattering: ', total_chattering)

ax4.plot(times, u, label='u')
ax4.plot(times, v_dot, label='v_dot')
ax4.set_xlabel('Time')
ax4.set_ylabel('inputs')
ax4.legend()

plt.show()
