import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Parameters
N = 100
P_VALUES = np.linspace(0.0, 1.0, 30)

RUNS_DEFAULT = 50
RUNS_CRITICAL = 200
CRITICAL_RANGE = (0.5, 0.7)

# Cell states
EMPTY = 0
TREE = 1
BURNING = 2
BURNED = 3

# Moore neighborhood kernel
KERNEL = np.ones((3, 3))
KERNEL[1, 1] = 0


# Helper functions
def initialize_grid(N, p):
    grid = (np.random.rand(N, N) < p).astype(int)
    return grid


def ignite_tree(grid):
    tree_positions = np.argwhere(grid == TREE)
    if len(tree_positions) == 0:
        return False

    i, j = tree_positions[np.random.randint(len(tree_positions))]
    grid[i, j] = BURNING
    return True


def step(grid):
    burning_neighbors = convolve2d((grid == BURNING), KERNEL, mode="same", boundary="fill", fillvalue=0)
    new_grid = grid.copy()
    new_grid[grid == BURNING] = BURNED
    new_grid[(grid == TREE) & (burning_neighbors > 0)] = BURNING
    return new_grid


# Single run
def run_simulation(N, p):
    grid = initialize_grid(N, p)

    initial_trees = np.sum(grid == TREE)
    if initial_trees == 0:
        return 0.0, 0

    ignited = ignite_tree(grid)
    if not ignited:
        return 0.0, 0

    t = 0
    while np.any(grid == BURNING):
        grid = step(grid)
        t += 1

    burned = np.sum(grid == BURNED)
    burned_fraction = burned / initial_trees

    return burned_fraction, t


# Monte Carlo
def monte_carlo(p_values):
    avg_burned = []
    avg_time = []

    for p in p_values:
        if CRITICAL_RANGE[0] <= p <= CRITICAL_RANGE[1]:
            runs = RUNS_CRITICAL
        else:
            runs = RUNS_DEFAULT

        burned_list = []
        time_list = []
        for _ in range(runs):
            burned, duration = run_simulation(N, p)
            burned_list.append(burned)
            time_list.append(duration)

        avg_burned.append(np.mean(burned_list))
        avg_time.append(np.mean(time_list))
        print(f"p={p:.3f} | burned={avg_burned[-1]:.3f} | time={avg_time[-1]:.2f}")

    return np.array(avg_burned), np.array(avg_time)


# Critical Point Estimation
def estimate_pc(p_values, avg_burned):
    # Threshold crossing
    threshold_indices = np.where(avg_burned > 0.5)[0]
    pc_threshold = p_values[threshold_indices[0]] if len(threshold_indices) > 0 else None

    # Max slope
    slopes = np.gradient(avg_burned, p_values)
    pc_slope = p_values[np.argmax(slopes)]

    return pc_threshold, pc_slope


# Plot Results
def plot_results(p_values, avg_burned, avg_time, pc_estimates):
    pc_threshold, pc_slope = pc_estimates

    # Burned area
    plt.figure()
    plt.plot(p_values, avg_burned)
    plt.xlabel("Tree Density (p)")
    plt.ylabel("Burned Fraction")
    plt.title("Burned Area vs p")

    if pc_slope is not None:
        plt.axvline(pc_slope, linestyle="--", label=f"pc ≈ {pc_slope:.3f}")
        plt.legend()

    # Time to extinction
    plt.figure()
    plt.plot(p_values, avg_time)
    plt.xlabel("Tree Density (p)")
    plt.ylabel("Time Until Fire Stops")
    plt.title("Fire Duration vs p")

    plt.show()


# Visualize Grid
def visualize_grid(N, p):
    """Display the 100x100 grid evolution during a single simulation."""
    grid = initialize_grid(N, p)
    initial_trees = np.sum(grid == TREE)
    
    if initial_trees == 0:
        print("No trees to burn with this probability.")
        return
    
    ignited = ignite_tree(grid)
    if not ignited:
        print("Could not ignite a tree.")
        return
    
    # Create a visualization showing initial and final states
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Initial state
    grid_initial = initialize_grid(N, p)
    ignite_tree(grid_initial)
    axes[0].imshow(grid_initial, cmap="RdYlGn_r", vmin=0, vmax=3)
    axes[0].set_title(f"Initial Grid (p={p:.2f})")
    axes[0].set_xlabel("100x100 Forest")
    
    # Run simulation
    t = 0
    while np.any(grid == BURNING):
        grid = step(grid)
        t += 1
    
    # Final state
    axes[1].imshow(grid, cmap="RdYlGn_r", vmin=0, vmax=3)
    axes[1].set_title(f"Final Grid (after {t} steps)")
    axes[1].set_xlabel("Burned area shown in red")
    
    plt.tight_layout()
    plt.show()


def main():
    avg_burned, avg_time = monte_carlo(P_VALUES)
    pc_estimates = estimate_pc(P_VALUES, avg_burned)
    print("\nEstimated critical points:")
    print(f"Threshold method: {pc_estimates[0]}")
    print(f"Slope method:     {pc_estimates[1]}")
    plot_results(P_VALUES, avg_burned, avg_time, pc_estimates)
    
    # Show grid visualization
    print("\nVisualizing 100x100 grid...")
    visualize_grid(N, 0.6)


if __name__ == "__main__":
    main()
