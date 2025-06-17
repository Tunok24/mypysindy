import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# ---------- Pendulum Dynamics ----------
def damped_pendulum(t, y, g=9.81, l=1, m=1.0, mu=0.8):
    theta, omega = y
    dtheta_dt = omega
    c = mu / m
    domega_dt = - (g / l) * np.sin(theta) - c * omega
    return [dtheta_dt, domega_dt]

# ---------- Simulation Params ----------
train_range = [0, 5]
t_train = np.linspace(0, 5, 100)
y0 = [0, np.pi]

# ---------- Simulate True System ----------
sol = solve_ivp(damped_pendulum, train_range, y0, t_eval=t_train)
X_true = sol.y.T                      # [theta, omega]
X_dot_true = np.gradient(X_true, t_train, axis=0)

# ---------- Add Gaussian Noise ----------
np.random.seed(42)
noise_strength = 0.2
X_noisy = X_true + noise_strength * np.random.normal(size=X_true.shape)

# ---------- Choose Feature Library ----------
library_choice = "polynomial"  # Change to: "polynomial", "fourier", or "custom"

if library_choice == "polynomial":
    feature_library = ps.PolynomialLibrary(degree=3)
elif library_choice == "fourier":
    feature_library = ps.FourierLibrary(n_frequencies=3)
elif library_choice == "custom1":
    feature_library = ps.CustomLibrary(
        library_functions=[
            lambda x: x[:, 0] * np.sin(x[:, 0]),
            lambda x: x[:, 1] * np.cos(x[:, 0])
        ],
        function_names=["theta*sin(theta)", "omega*cos(theta)"]
    )
else:
    raise ValueError("Unknown library choice. Choose: 'polynomial', 'fourier', 'custom1' etc.")

# ---------- Choose Optimizer ----------
optimizer_choice = "sr3"  # Options: "stlsq", "sr3", "lasso", "elasticnet", "ensemble"

if optimizer_choice == "stlsq":
    optimizer = ps.STLSQ(threshold=0.1)
elif optimizer_choice == "sr3":
    optimizer = ps.SR3(threshold=0.1, nu=1e-2)
elif optimizer_choice == "lasso":
    optimizer = Lasso(alpha=1e-2, fit_intercept=False, max_iter=10000)
elif optimizer_choice == "elasticnet":
    optimizer = ElasticNet(alpha=1e-2, l1_ratio=0.7, fit_intercept=False, max_iter=10000)
elif optimizer_choice == "ensemble":
    base_optimizer = ps.STLSQ(threshold=0.1)
    optimizer = ps.EnsembleOptimizer(opt=base_optimizer, n_models=10, bagging=True)
else:
    raise ValueError("Unknown optimizer. Choose: 'stlsq', 'sr3', 'lasso', 'elasticnet', or 'ensemble'.")


# ---------- Fit Model ----------
model = ps.SINDy(
    feature_library=feature_library,
    differentiation_method=ps.SmoothedFiniteDifference(),
    optimizer=optimizer
)
model.fit(X_noisy, t=t_train)
model.print()

# ---------- Predict & Simulate ----------
t_test = np.linspace(0, 10, 200)
X_dot_pred = model.predict(X_true)
X_pred = model.simulate(X_noisy[0], t_test)

# ----------  (Optional) Generate X_true_extrap for comparison ----------
sol_extrap = solve_ivp(damped_pendulum, [0, 10], y0, t_eval=t_test)
X_true_extrap = sol_extrap.y.T

# ---------- Score and Report ----------
r2 = r2_score(X_true_extrap, X_pred)
mse = mean_squared_error(X_true_extrap, X_pred)
sparsity = np.sum(model.coefficients() != 0)

results = pd.DataFrame([{
    "Optimizer": optimizer_choice,
    "Library": library_choice,
    "R2 Score (vs. True)": r2,
    "MSE (vs. True)": mse,
    "Nonzero Terms": sparsity
}])

print(results.to_string(index=False))


# ---------- PLOT 1: Pendulum Motion (Theta vs. Time) ----------
# ---------- Optional: Plotting Theta ----------
plt.figure(figsize=(10, 4))
plt.plot(t_test, X_pred[:, 0], '--', label=r'$\theta_{pred}$', linewidth=2)
plt.plot(t_test, X_true_extrap[:, 0], label=r'$\theta_{true}$', linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Theta (rad)")
plt.title(f"Theta vs. Time Comparison {optimizer_choice} {library_choice}")
plt.legend()
plt.grid()
plt.savefig(f"theta_comparison.png")



# plt.figure(figsize=(10, 4))
# plt.plot(t_train, X_noisy[:, 0], label=r'$\theta(t)$ (True Noisy)', linewidth=2, color='blue')
# plt.plot(t_train, X_true[:, 0], label=r'$\theta(t)$ (True)', linewidth=0.9, color='red')
# plt.xlabel("Time (s)")
# plt.ylabel("Theta (radians)")
# plt.title("Pendulum Motion: Theta vs. Time")
# plt.legend()
# plt.grid()
# plt.savefig("pendulum_motion.png")
# print(f"Saved image: pendulum_motion.png")

# # ---------- PLOT 2: Phase Space (Theta vs. Omega) ----------
# plt.figure(figsize=(6, 6))
# plt.plot(X_true[:, 0], X_true[:, 1], label="True Trajectory", linewidth=2)
# plt.xlabel("Theta (radians)")
# plt.ylabel("Angular Velocity (rad/s)")
# plt.title("Phase Space: Theta vs. Omega")
# plt.legend()
# plt.grid()
# plt.savefig("phasespace_theta_omega.png")
# print(f"Saved image: phasespace_theta_omega.png")

# ---------- PLOT 3: True vs. Learned Equation ----------
# plt.figure(figsize=(10, 4))
# plt.plot(t_train, X_dot_true[:, 1], label="True dω/dt", linestyle='dashed', linewidth=2, color='black')
# plt.plot(t_train, X_dot_pred[:, 1], label="SINDy Predicted dω/dt", linestyle='dotted', linewidth=2, color='red')
# plt.xlabel("Time (s)")
# plt.ylabel("dω/dt")
# plt.title("True vs. SINDy Learned Equation")
# plt.legend()
# plt.grid()
# plt.savefig("true_vs_sindy_eq.png")
# print(f"Saved image: true_vs_sindy_eq.png")