import numpy as np
import matplotlib.pyplot as plt

def system(x1, x2, x3, u):
    a = 0.7
    x1_dot = x2
    x2_dot = x3
    x3_dot = u + a * np.sin(x1)
    return x1_dot, x2_dot, x3_dot

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

# K = 50
K = np.ones(n + 1)
K[0] = 5
K_dot = np.zeros(n)
Km = 0.05
Kmm = 0.01 # 0 < Kmm <= Km
l = 20
lm = 4
N = 10 # N >= 2
mu = 50
alpha = np.zeros(n)
q = 1.2 # q > 1

c1 = 1

S = np.zeros(n)
s_dot = np.zeros(n)

x1 = np.zeros(n)
x2 = np.zeros(n)
x3 = np.zeros(n)

y = x1
y_ref = 10 * np.sin((np.arange(0, n, 1) / n) * 2 * np.pi * 4)
e = np.zeros(n)
e_dot = np.zeros(n)
e_ddot = np.zeros(n)

x1[0] = 0.5
x2[0] = 0
x3[0] = 0

u = np.zeros(n)


# Main simulation loop
for i in range(n):
    
    e[i] = y[i] - y_ref[i]
    e_dot[i] = d(e, i, dt)
    
    # Compute sliding variable
    S[i] = e_dot[i] + c1 * e[i]
    s_dot[i] = d(S, i, dt)
    
    # adaptative gain law
    alpha[i] = 1
    for s in S[i-N:i]:
        if abs(s) > mu * K[i] * dt:
            alpha[i] = -1
            break
    
    if K[i] > Km:
        K_dot[i] = - alpha[i] * l * K[i]
    elif Kmm < K[i] <= Km:
        K_dot[i] = -alpha[i] * lm
    else:
        K_dot[i] = lm
        
    if alpha[i - 1] == 1 and alpha[i] == -1:
        K[i + 1] = q * K[i] + K_dot[i] * dt
    else:
        K[i + 1] = K[i] + K_dot[i] * dt
    
    # Control law 
    u[i] = -K[i + 1] * np.sign(S[i])

    # Update system response using the control input
    x1_dot, x2_dot = pendule(x1[i], x2[i], u[i], times[i])
    
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
ax1.set_xlabel('Time')
ax1.set_ylabel('x1, x2')
ax1.legend()

ax2.plot(times, S, label='S')
ax2.set_xlabel('Time')
ax2.set_ylabel('sliding variable')
ax2.legend()

ax3.plot(times, K[:-1], label='K')
ax3.set_xlabel('Time')
ax3.set_ylabel('adaptative gain')
ax3.legend()

ax4.plot(times, alpha, label='alpha')
ax4.set_xlabel('Time')
ax4.set_ylabel('critère de glissement')
ax4.legend()

plt.show()
