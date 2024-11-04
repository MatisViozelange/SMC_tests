import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from modules.STWC_NN import ASTWC, NN_based_STWC
from modules.Dynamic_systems import basic_system, pendule

################################################## SIMULATION ########################################################
time = 20
Te = 0.0005
n = int(time / Te) 
times = np.linspace(0, time, n)
y_ref = 10 * np.sin(times)

# First controler without neural network
controler = ASTWC(time, Te, reference=None)
# NN_observation = NN_based_STWC(controler, time, Te)

# seconde controler with neural network
NN_inner_controler = ASTWC(time, Te, reference=None)
NN_controler = NN_based_STWC(NN_inner_controler, time, Te)

# dynamics
system = basic_system(controler.times)
# system = pendule(controler.times)


# Loop simulation
for i in tqdm(range(n)):
    # SuperTwisting control law
    controler.compute_input(i)
    NN_controler.compute_input(i)
    
    # NN_observation.compute_hidden_layer(i, controler.s[i])
    # NN_observation.compute_weights(i, controler.k[i], controler.s[i], controler.epsilon)
    # NN_observation.compute_perturbation(i)
    
    # Update system response using the control input for ASTWC
    u = controler.u[i]
    x1_dot, x2_dot = system.compute_dynamics(
                        controler.x1[i], 
                        controler.x2[i], 
                        u, 
                        controler.times[i]
                    )
    
    # Update states for ASTWC
    controler.x1[i + 1] = controler.x1[i] + x1_dot * controler.Te
    controler.x2[i + 1] = controler.x2[i] + x2_dot * controler.Te
    
    # Update system response using the control input for NN_based_ASTWC
    u = NN_controler.u[i]
    x1_dot, x2_dot = system.compute_dynamics(
                        NN_controler.controler.x1[i], 
                        NN_controler.controler.x2[i], 
                        u, 
                        NN_controler.controler.times[i]
                    )
    
    # Update states for NN_based_ASTWC
    NN_controler.controler.x1[i + 1] = NN_controler.controler.x1[i] + x1_dot * NN_controler.controler.Te
    NN_controler.controler.x2[i + 1] = NN_controler.controler.x2[i] + x2_dot * NN_controler.controler.Te

################################ PLOTTING ################################
fig, axs = plt.subplots(3, 3, num=1, figsize=(8, 6), sharex=True, sharey=False)
(ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = axs.flat

# perturbations

perturbation    = system.get_perturbation(times, controler.x1, controler.x2)
perturbation_nn = system.get_perturbation(times, NN_controler.controler.x1, NN_controler.controler.x2)

# Now compute the derivatives after smoothing
dot_perturbation = np.gradient(perturbation, Te)
dot_perturbation_nn = np.gradient(perturbation_nn, Te) 

d_max    = np.max(np.abs(dot_perturbation))
d_max_nn = np.max(np.abs(dot_perturbation_nn))

#######################################################################
################################ No NN ################################


################################ ax1 ################################
# states

ax1.title.set_text('ASTWC')
ax1.plot(controler.times, controler.x1[:-1], label='x1')
ax1.plot(controler.times, controler.x2[:-1], label='x2')
ax1.plot(controler.times, controler.y_ref, label='ref')
ax1.set_ylabel('x1, x2')
ax1.legend()

################################ ax2 ################################
# Plot the perturbation and the gain response without NN

ax2.set_title('ASTWC perturbations derivatives')
ax2.plot(times, np.abs(dot_perturbation), label='|d_dot|')
ax2.plot(times, controler.k[:-1], label='k')
ax2.axhline(y=d_max, color='r', linestyle='--', label=f'Delta_d = {d_max:.2f}')
ax2.set_ylabel('perturbation derivative')
ax2.legend()

################################ ax3 ################################
# SLiding variable

ax3.set_title('ASTWC sliding variable')
ax3.plot(controler.times, controler.s, label='s')
ax3.axhline(y=controler.epsilon, color='r', linestyle='--', label=f'epsilon = {controler.epsilon:.2f}')
ax3.axhline(y=-controler.epsilon, color='r', linestyle='--')
ax3.set_ylabel('sliding variable')
ax3.legend()



################################ ax8 ################################
# inputs

ax8.set_title('ASTWC')
ax8.plot(times, controler.u, label='u')
ax8.plot(times, controler.v_dot, label='v_dot')
ax8.set_xlabel('Time')
ax8.set_ylabel('inputs')
ax8.legend()


####################################################################
################################ NN ################################


################################ ax4 ################################

ax4.title.set_text('NN_based_ASTWC')
ax4.plot(NN_controler.controler.times, NN_controler.controler.x1[:-1], label='x1')
ax4.plot(NN_controler.controler.times, NN_controler.controler.x2[:-1], label='x2')
ax4.plot(NN_controler.controler.times, NN_controler.controler.y_ref, label='ref')
ax4.set_ylabel('x1, x2')
ax4.legend()

################################ ax5 ################################
# Plot the perturbation and the gain response with NN

ax5.set_title('NN_based_ASTWC perturbations derivatives')
ax5.plot(times, np.abs(dot_perturbation_nn), label='|d_dot|')
ax5.plot(times, NN_controler.controler.k[:-1], label='k')
ax5.axhline(y=d_max_nn, color='r', linestyle='--', label=f'Delta_d = {d_max_nn:.2f}')
ax5.set_ylabel('perturbation derivative')
ax5.legend()

################################ ax6 ################################
# Sliding variable

ax6.set_title('NN_based_ASTWC sliding variable')
ax6.plot(NN_controler.controler.times, NN_controler.controler.s, label='s')
ax6.axhline(y=NN_controler.controler.epsilon, color='r', linestyle='--', label=f'epsilon = {NN_controler.controler.epsilon:.2f}')
ax6.axhline(y=-NN_controler.controler.epsilon, color='r', linestyle='--')
ax6.set_ylabel('sliding variable')
ax6.legend()

################################ ax7 ################################
# estimated perturbation by neural network
# ax7.set_title('perturbations and NN approximation')
# ax7.plot(times, perturbation, label='d')
# ax7.plot(times, NN_observation.perturbation[:-1], label='NN d approx')
# ax7.set_xlabel('Time')
# ax7.set_ylabel('perturbation')
# ax7.legend()

ax7.set_title('perturbations and NN approximation')
ax7.plot(times, perturbation_nn, label='d')
ax7.plot(times, NN_controler.perturbation[:-1], label='NN d approx')
ax7.set_xlabel('Time')
ax7.set_ylabel('perturbation')
ax7.legend()

################################ ax9 ################################
#inputs

ax9.set_title('NN_based_ASTWC')
ax9.plot(times, NN_controler.controler.u, label='u')
ax9.plot(times, NN_controler.controler.v_dot, label='v_dot')
ax9.set_xlabel('Time')
ax9.set_ylabel('inputs')
ax9.legend()


plt.show()