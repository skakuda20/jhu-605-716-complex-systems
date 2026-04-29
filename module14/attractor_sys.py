import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def system(t, state, alpha):
    x, y, z = state
    dxdt = -alpha * x + np.sin(y)
    dydt = -alpha * y + np.sin(z)
    dzdt = -alpha * z + np.sin(x)
    return [dxdt, dydt, dzdt]

def jacobian(state, alpha):
    x, y, z = state
    return np.array([
        [-alpha, np.cos(y),       0      ],
        [0,      -alpha,          np.cos(z)],
        [np.cos(x), 0,            -alpha ]
    ])

def largest_lyapunov_exponent(alpha, T=300, dt=0.01, transient=100):
    """Estimate the largest Lyapunov exponent via QR renormalization."""
    t_span = (0, T)
    t_eval = np.arange(0, T, dt)

    sol = solve_ivp(system, t_span, [1, 2, 3], args=(alpha,),
                    t_eval=t_eval, rtol=1e-9, atol=1e-9)
    traj = sol.y.T  # shape (N, 3)

    # Discard transient
    start = int(transient / dt)
    traj = traj[start:]
    N = len(traj)

    Q = np.eye(3)
    lyap_sum = 0.0

    for i in range(N):
        J = jacobian(traj[i], alpha)
        Q = Q + dt * (J @ Q)
        Q, R = np.linalg.qr(Q)
        lyap_sum += np.log(abs(R[0, 0]))

    return lyap_sum / (N * dt)

def classify_attractor(alpha, T=300, dt=0.01, transient=100):
    """Classify the attractor type based on LLE and trajectory variance."""
    t, x, y, z = simulate(alpha, T=T, dt=dt, transient=transient)

    total_variance = np.var(x) + np.var(y) + np.var(z)

    lle = largest_lyapunov_exponent(alpha, T=T, dt=dt, transient=transient)

    if total_variance < 1e-6:
        label = "Fixed Point"
    elif lle > 0.01:
        label = "Strange / Chaotic Attractor"
    elif lle < -0.01:
        if total_variance < 0.01:
            label = "Fixed Point"
        else:
            label = "Limit Cycle"
    else:
        label = "Limit Cycle / Quasi-Periodic"

    return label, lle

def simulate(alpha, T=300, dt=0.01, transient=100):
    t_eval = np.arange(0, T, dt)

    sol = solve_ivp(
        system,
        t_span=(0, T),
        y0=[1, 2, 3],
        args=(alpha,),
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-9
    )

    t = sol.t
    x, y, z = sol.y

    mask = t > transient

    return t[mask], x[mask], y[mask], z[mask]

def plot_results(alpha, T=300, transient=100):
    t, x, y, z = simulate(alpha, T=T, transient=transient)
    attractor_type, lle = classify_attractor(alpha, T=T, transient=transient)

    print(f"alpha = {alpha:.2f}  |  Attractor Type: {attractor_type}  |  LLE ≈ {lle:.4f}")

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t, x)
    plt.ylabel("x(t)")
    plt.title(f"alpha = {alpha}  |  {attractor_type}  (LLE ≈ {lle:.4f})")

    plt.subplot(3, 1, 2)
    plt.plot(t, y)
    plt.ylabel("y(t)")

    plt.subplot(3, 1, 3)
    plt.plot(x, z, linewidth=0.6)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("Phase Plane: x vs z")

    plt.tight_layout()
    plt.show()

alphas = [0.30, 0.22, 0.20, 0.10]

print("=" * 60)
for alpha in alphas:
    plot_results(alpha, T=300, transient=100)
print("=" * 60)