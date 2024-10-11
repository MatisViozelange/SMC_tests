import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def d(x, i, dt):
    if i < 1:
        dx = 0
    else:
        dx = (x[i] - x[i - 1]) / dt
    return dx

def longueur_pendule(t):
    return 0.8 + 0.1 * np.sin(8 * t) + 0.3 * np.cos(4 * t)

def pendule(x1, x2, u, t):
    l = longueur_pendule(t)
    l_h = longueur_pendule(t - dt)
    l_dot = (l - l_h) / dt
    
    g = 9.81
    m = 2
    
    x1_dot = x2
    x2_dot = -2 * l_dot / l - g / l * np.sin(x1) + 2 * np.sin(5 * t) + (1 + 0.5 * np.sin(t)) / (m * l**2) * u
    
    return x1_dot, x2_dot


n = 100000
times = np.linspace(0, 20, n)
dt = times[1]

alpha = np.ones(n + 1)
alpha_dot = np.zeros(n + 1)
beta = np.ones(n + 1)
# alpha[0] = 10
# beta[0] = 20
mu = 0.05
epsilon = 0.05
eta = 5
w = 100
gamma = 0.4
alpha_m = 0.001

c1 = 1

s = np.zeros(n)
s_dot = np.zeros(n)

x1 = np.zeros(n)
x2 = np.zeros(n)

x1[0] = 0.5

y = x1
y_ref = np.zeros(n) #10 * np.sin((np.arange(0, n, 1) / n) * 2 * np.pi * 4)
y_ref_dot = np.gradient(y_ref, dt)
e = np.zeros(n)
e_dot = np.zeros(n)

u = np.zeros(n)
v_dot = np.zeros(n)


# Main simulation loop
for i in range(n):
    # System output
    y[i] = x1[i]
    
    # Compute error and its derivative
    e[i] = y[i] - y_ref[i]
    e_dot[i] = x2[i] - y_ref_dot[i]
    
    # Compute sliding variable
    s[i] = e_dot[i] + c1 * e[i]
    s_dot[i] = d(s, i, dt)
    
    # compute adaptative gains
    if alpha[i] > alpha_m:
        alpha_dot[i] = w * np.sqrt(gamma / 2) *  np.sign(abs(s[i]) - mu)
    else:
        alpha_dot[i] = eta
    
    alpha[i + 1] = alpha[i] + alpha_dot[i] * dt
    beta[i + 1] = 2 * epsilon * alpha[i + 1]
    
    # SuperTwisting control law
    v_dot[i] = - beta[i + 1] * np.sign(s[i])
    u[i] = - alpha[i + 1] * np.sqrt(abs(s[i])) * np.sign(s[i]) + integrate.simpson(v_dot[:i + 1])
    
    # Update system response using the control input
    x1_dot, x2_dot = pendule(x1[i], x2[i], u[i], times[i])
    
    # Update states
    if i < n - 1:
        x1[i + 1] = x1[i] + x1_dot * dt
        x2[i + 1] = x2[i] + x2_dot * dt

# Plotting results
fig, axs = plt.subplots(2, 2, num=1, figsize=(8, 6), sharex=True, sharey=False)
(ax1, ax2), (ax3, ax4) = axs 

ax1.plot(times, x1, label='x1')
ax1.plot(times, x2, label='x2')
ax1.plot(times, y_ref, label='ref')
ax1.set_xlabel('Time')
ax1.set_ylabel('x1, x2')
ax1.legend()

# Plot phase portrait
ax2.plot(times, s, label='s')
# ax2.plot(times, s_dot, label='s_dot')
ax2.set_xlabel('Time')
ax2.set_ylabel('sliding variable')
ax2.legend()

ax3.plot(times, alpha[: -1], label='aplha')
ax3.plot(times, beta[: -1], label='beta')
ax3.set_xlabel('Time')
ax3.set_ylabel('adaptative gain')
ax3.legend()

ax4.plot(times, u, label='u')
ax4.plot(times, v_dot, label='v_dot')
ax4.set_xlabel('Time')
ax4.set_ylabel('inputs')
ax4.legend()

# # Set common time axis for zoom
# for ax in axs.flat:
#     ax.label_outer()

plt.show()
