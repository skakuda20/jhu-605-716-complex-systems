import numpy as np
import matplotlib.pyplot as plt

n = 100
steps = 200
num_trials = 20

def initialize(p):
    return (np.random.random((n, n)) < p).astype(int)

# def update(config):
#     nextconfig = np.zeros_like(config)

#     for x in range(n):
#         for y in range(n):
#             count = 0

#             for dx in [-1, 0, 1]:
#                 for dy in [-1, 0, 1]:
#                     count += config[(x + dx) % n, (y + dy) % n]

#             nextconfig[x, y] = 1 if count >= 4 else 0

#     return nextconfig
def update_fast(config):
    count = (
        np.roll(np.roll(config, -1, axis=0), -1, axis=1) +
        np.roll(np.roll(config, -1, axis=0),  0, axis=1) +
        np.roll(np.roll(config, -1, axis=0),  1, axis=1) +
        np.roll(np.roll(config,  0, axis=0), -1, axis=1) +
        np.roll(np.roll(config,  0, axis=0),  0, axis=1) +
        np.roll(np.roll(config,  0, axis=0),  1, axis=1) +
        np.roll(np.roll(config,  1, axis=0), -1, axis=1) +
        np.roll(np.roll(config,  1, axis=0),  0, axis=1) +
        np.roll(np.roll(config,  1, axis=0),  1, axis=1)
    )

    return (count >= 4).astype(int)

# def run_simulation(p, steps=200):
#     config = initialize(p)

#     panic_history = []

#     for t in range(steps):
#         panic_fraction = config.mean()
#         panic_history.append(panic_fraction)

#         if panic_fraction == 0.0 or panic_fraction == 1.0:
#             break

#         config = update(config)

#     return panic_history
def run_simulation(p, steps=200):
    config = initialize(p)
    panic_history = []

    for t in range(steps):
        panic_fraction = config.mean()
        panic_history.append(panic_fraction)

        if panic_fraction == 0.0 or panic_fraction == 1.0:
            break

        config = update_fast(config)

    return panic_history


def main():
    p_values = np.linspace(0.1, 0.5, 50)
    print(f"Testing {len(p_values)} values...")
    success_rates = []

    for p in p_values:
        full_panic_count = 0

        for trial in range(num_trials):
            history = run_simulation(p, steps)

            if history[-1] == 1.0:
                full_panic_count += 1

        success_rate = full_panic_count / num_trials
        success_rates.append(success_rate)

        print(f"p = {p:.3f}, full panic probability = {success_rate:.2f}")

    plt.plot(p_values, success_rates, marker="o")
    plt.xlabel("Initial panic probability p")
    plt.ylabel("Probability of reaching 100% panic")
    plt.title("Estimated phase transition in Panic Cellular Automaton")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.show()

    for p, rate in zip(p_values, success_rates):
        if rate >= 0.5:
            pc_estimate = p
            break
    
    print("Estimated critical pc:", pc_estimate)

    p = pc_estimate
    history = run_simulation(p, steps=200)

    print("Initial panic fraction:", history[0])
    print("Minimum panic fraction:", min(history))
    print("Final panic fraction:", history[-1])

    plt.plot(history)
    plt.xlabel("Time step")
    plt.ylabel("Fraction panicked")
    plt.title(f"Panic evolution near pc = {p:.3f}")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
