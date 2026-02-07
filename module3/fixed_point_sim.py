import numpy as np

N = 200_000

def classify_fixed_point(a, b, c, d, eps=1e-6):
    """
    Classify fixed point using trace and determinant.
    """
    tau = a + d
    delta = a*d - b*c
    disc = tau**2 - 4*delta

    if abs(delta) < eps:
        return "non-isolated"

    if delta < 0:
        return "saddle"

    if abs(disc) < eps:
        return "degenerate"

    if disc < 0:
        if abs(tau) < eps:
            return "center"
        elif tau < 0:
            return "stable spiral"
        else:
            return "unstable spiral"

    # disc > 0
    if tau < 0:
        return "stable node"
    else:
        return "unstable node"


def monte_carlo_sim(N=100_000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    counts = {
        "center": 0,
        "degenerate": 0,
        "non-isolated": 0,
        "saddle": 0,
        "stable node": 0,
        "unstable node": 0,
        "stable spiral": 0,
        "unstable spiral": 0,
    }

    for i in range(N):
        a, b, c, d = np.random.uniform(-1, 1, 4)
        fp_type = classify_fixed_point(a, b, c, d)
        counts[fp_type] += 1

    probabilities = {k: v / N for k, v in counts.items()}
    return probabilities


if __name__ == "__main__":
    results = monte_carlo_sim(N, seed=42)

    print(f"Fixed Point probabilities (N={N})")
    for k, v in sorted(results.items()):
        print(f"{k:25s}: {v:.5f}")
