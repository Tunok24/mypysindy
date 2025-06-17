import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Original Hamiltonian system ---
def hamiltonian_pendulum(t, y, g=9.81, l=1.0, m=1.0):
    theta, p = y
    dtheta_dt = p / m
    dp_dt = -m * g * l * np.sin(theta)
    return [dtheta_dt, dp_dt]

# --- Learned SINDy model ---
def learned_pendulum(t, y):
    theta, p = y
    dtheta_dt = 0.778 * p
    dp_dt = 0.141 * np.sin(theta)# + 0.156 * p
    return [dtheta_dt, dp_dt]

# --- Simulation setup ---
t_span = (0, 10)
t_eval = np.linspace(*t_span, 1000)
y0 = [np.pi / 4, 0.0]  # initial state: 45 degrees, zero momentum

# --- Simulate true dynamics ---
sol_true = solve_ivp(hamiltonian_pendulum, t_span, y0, t_eval=t_eval)
theta_true, p_true = sol_true.y

# --- Simulate learned dynamics ---
sol_learned = solve_ivp(learned_pendulum, t_span, y0, t_eval=t_eval)
theta_learned, p_learned = sol_learned.y

# === Plotting ===
plt.figure(figsize=(14, 6))

# --- Time-series plot ---
plt.subplot(1, 2, 1)
plt.plot(t_eval, theta_true, label=r'$\theta_{\mathrm{true}}$', linewidth=2)
plt.plot(t_eval, p_true, label=r'$p_{\mathrm{true}}$', linewidth=2)
plt.plot(t_eval, theta_learned, '--', label=r'$\theta_{\mathrm{learned}}$', linewidth=2)
plt.plot(t_eval, p_learned, '--', label=r'$p_{\mathrm{learned}}$', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.title('State Evolution')
plt.legend()
plt.grid()

# --- Phase space comparison ---
plt.subplot(1, 2, 2)
plt.plot(theta_true, p_true, label='True System', linewidth=2)
plt.plot(theta_learned, p_learned, '--', label='Learned System', linewidth=2)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$p$')
plt.title('Phase Space Trajectories')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("Learned_vs_True_Pendulum_Comparison.png", dpi=200)
