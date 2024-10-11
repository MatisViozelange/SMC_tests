import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def system(x1, x2, u):
    a = 0.7
    x1_dot = x2
    x2_dot = u + a * np.sin(x1)
    return x1_dot, x2_dot

Te = 0.0001
time = 20
n = int(time / Te)
times = np.linspace(0, time, n)

K = 3
c1 = 0.5

S = np.zeros(n)

x1 = np.zeros(n + 1)
x2 = np.zeros(n + 1)

y = np.zeros(n)
y_ref = 10 * np.sin((np.arange(0, n + 1, 1) / (n + 1)) * 2 * np.pi * 4)
dy_ref = np.gradient(y_ref, Te)

e = np.zeros(n)
d_e = np.zeros(n)

x1[0] = 0.5
x2[0] = 0

u = np.zeros(n)

for i in tqdm(range(n)):
    y[i] = x1[i]
    
    e[i] = y[i] #- y_ref[i]
    d_e[i] = x2[i] #- dy_ref[i]
    
    # Compute sliding variable
    S[i] = d_e[i] + c1 * e[i]
    
    # Control law using np.sign directly
    u[i] = -K * np.sign(S[i])
    
    # System response update
    x1_dot, x2_dot = system(x1[i], x2[i], u[i])
    
    x2[i + 1] = x2[i] + x2_dot * Te
    x1[i + 1] = x1[i] + x1_dot * Te

# Plotting results
fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharex=False, sharey=False)
((ax1, ax2), (ax3, ax4)) = axs

ax1.plot(times, x1[:-1], label='x1')
ax1.plot(times, x2[:-1], label='x2')
# ax1.plot(times, y_ref[:-1], label='ref')
ax1.set_xlabel('Time')
ax1.set_ylabel('x1, x2, x3')
ax1.legend()

# Phase plane plot
ax2.plot(x1, x2, label='Phase Plane')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.legend()

ax3.plot(times, u, label='Control input')
ax3.set_xlabel('Time')
ax3.set_ylabel('u')
ax3.legend()

ax4.plot(times, S, label='Sliding variable')
ax4.set_xlabel('Time')
ax4.set_ylabel('s')
ax4.legend()

plt.show()
