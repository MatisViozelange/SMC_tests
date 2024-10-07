import numpy as np
import matplotlib.pyplot as plt

def system(x1, x2, u):
    a = 0.7
    x1_dot = x2
    x2_dot = u + a * np.sin(x1)
    return x1_dot, x2_dot

def d(x, i, dt):
    if i <= 1:
        dx = 0
    else :
        dx = (x[i] - x[i - 1]) / dt
    return dx

n = 100000
times = np.linspace(0, 20, n)
dt = times[1]

k1 = 20
k2 = 10

s = np.zeros(n)
s_dot = np.zeros(n)

x1 = np.zeros(n)
x2 = np.zeros(n)

x1[0] = 0.5

y = x1
y_ref =  10 * np.sin((np.arange(0, n, 1) / n) * 2 * np.pi * 4)
e = np.zeros(n)


for i in range(n):
    # System output
    y[i] = x1[i]
    
    # Compute error and its derivative
    e[i] = y[i] - y_ref[i]
    
    # Compute sliding variable
    s[i] = e[i]
    s_dot[i] = d(s, i, dt)
    
    # if i < 100:
    #     print(f's_dot: {s_dot[i]}')
    
    # Twisting control law
    u = -k1 * np.sign(s[i]) - k2 * np.sign(s_dot[i])
    
    # Update system response using the control input
    x1_dot, x2_dot = system(x1[i], x2[i], u)
    
    # Update states
    if i < n - 1:
        x1[i + 1] = x1[i] + x1_dot * dt
        x2[i + 1] = x2[i] + x2_dot * dt

# Plotting results
fig, axs = plt.subplots(2, 2, num=1, figsize=(8, 6), sharex=False, sharey=False)
(ax1, ax2), (ax3, ax4) = axs 

ax1.plot(times, x1, label='x1')
ax1.plot(times, x2, label='x2')
ax1.set_xlabel('Time')
ax1.set_ylabel('x1, x2, x3')
ax1.legend()
# Phase diagram plot
ax2.plot(s[2:], s_dot[2:], label='Phase diagram')
ax2.set_xlabel('s')
ax2.set_ylabel('s_dot')
ax2.legend()

ax3.plot(times, s, label='s')
ax3.plot(times, s_dot, label='s_dot')
ax3.set_xlabel('Time')
ax3.set_ylabel('s, s_dot')
ax3.legend()

ax4.plot(times, y_ref, label='ref')
ax4.plot(times, y, label='y')
ax4.set_xlabel('Time')
ax4.set_ylabel('y, ref')
ax4.legend()

plt.show()
