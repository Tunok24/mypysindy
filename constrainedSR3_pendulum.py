from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from pysindy.feature_library import CustomLibrary
from pysindy.optimizers import ConstrainedSR3
import pysindy as ps

def hamiltonian_pendulum(t, y, mu=0.0):
    theta, p = y
    dtheta_dt = p
    dp_dt = - np.sin(theta) - mu * p
    return [dtheta_dt, dp_dt]

# Time and initial condition
t = np.linspace(0, 20, 500)
y0 = [np.pi / 4, 0.0]
sol = solve_ivp(hamiltonian_pendulum, [t[0], t[-1]], y0, t_eval=t)

X_true = sol.y.T  # Columns = [theta, p]
X_dot = np.gradient(X_true, t, axis=0)

# --- θ(t) and p(t) ---
theta_true = X_true[:, 0]
p_true = X_true[:, 1]

theta_norm = (theta_true - theta_true.min()) / (theta_true.max() - theta_true.min())
p_norm = (p_true - p_true.min()) / (p_true.max() - p_true.min())

X_true[:, 0] = theta_norm
X_true[:, 1] = p_norm

print("X shape:", X_true.shape)

####################################
plt.figure(figsize=(5, 2))
plt.plot(t, theta_norm, label=r'$\theta_{true}(t)$', linewidth=2)
plt.plot(t, p_norm, label=r'$p_{true}(t)$', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('θ, p')
plt.title('Pendulum Motion: θ and p over Time')
plt.legend()
plt.savefig("HamiltonianPendulumTrue.jpg", dpi=200)

####################################


def sin_theta(X):
    X = np.atleast_2d(X)
    return np.sin(X[:, [0]])

def p_only(X):
    X = np.atleast_2d(X)
    return X[:, [1]]

library = CustomLibrary(
    library_functions=[sin_theta, p_only],
    function_names=[
        lambda x: "sin(theta)",
        lambda x: "p",
    ]
)

# library = ps.PolynomialLibrary(degree=3)
# library = ps.FourierLibrary(n_frequencies=3)


# Total number of features × outputs = 2 × 2 = 4
# Each row of C selects an entry of vec(Ξ)
# We'll enforce that unwanted coefficients are zero

C = np.zeros((2, 8))
d = np.zeros(2)

# θ̇ → keep only p (index 1), zero others
C[0, 0] = 1  # sin(θ) -> θ̇
C[1, 3] = 1  # p -> p'


optimizer = ConstrainedSR3(
    threshold=0.1,
    constraint_lhs=C,
    constraint_rhs=d,
    constraint_order="target",
    verbose=True
)

# optimizer = ps.SR3(threshold=0.1, nu=1e-2)


model = ps.SINDy(
    optimizer=optimizer,
    feature_library=library,
    differentiation_method=ps.SmoothedFiniteDifference(),
    feature_names=["sin(θ)", "p"]
)
library.fit(X_true)
Θ = library.transform(X_true)
print("Θ(X) shape:", Θ.shape)
print("First few rows of Θ(X):\n", Θ[:5])

model.fit(X_true, t=t)

print(f"Model Equations: {model.equations()}")
print(f"Model Coefficients Shape: {model.coefficients().shape}")
# OR, print(f"Model Coefficients Shape: {model.optimizer.coef_.shape}")

