import numpy as np
import matplotlib.pyplot as plt

def chaos_game(P, a, n_points=2**18, discard=100):
    # Random initial point
    x, y = np.random.rand(), np.random.rand()

    xs = []
    ys = []

    for i in range(n_points):
        # Pick random vertex
        Xi, Yi = P[np.random.randint(len(P))]

        # Iterate
        x = a * x + Xi
        y = a * y + Yi

        # Store after transient
        if i >= discard:
            xs.append(x)
            ys.append(y)

    return np.array(xs), np.array(ys)


#  Part a
P_a = [(-2, np.sqrt(3)), (2, np.sqrt(3)), (0, -2)]
a_a = 0.5

x_a, y_a = chaos_game(P_a, a_a)

plt.figure(figsize=(6,6))
plt.scatter(x_a, y_a, s=0.1)
plt.title("Chaos Game")
plt.axis("equal")
plt.show()


# Part b
P_b = [(951,309), (588,-809), (-588,-809), (-951,309), (0,1000)]
a_b = 41/108

x_b, y_b = chaos_game(P_b, a_b)

plt.figure(figsize=(6,6))
plt.scatter(x_b, y_b, s=0.1)
plt.title("Chaos Game")
plt.axis("equal")
plt.show()
