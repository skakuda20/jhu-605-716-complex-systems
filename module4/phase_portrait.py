#!/usr/bin/env python3
"""
Phase Portrait Generator for 2D Dynamical Systems

This script analyzes and visualizes 2D ODE systems, including:
- Fixed points and their stability classification
- Nullclines
- Vector field
- Sample trajectories
- Eigenvalue analysis and Jacobian matrices
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')


class PhasePortrait2D:
    """
    A class to analyze and visualize 2D dynamical systems.
    
    Parameters
    ----------
    f : callable
        Function defining dx/dt = f(x, y)
    g : callable
        Function defining dy/dt = g(x, y)
    x_range : tuple
        (x_min, x_max) for the phase portrait
    y_range : tuple
        (y_min, y_max) for the phase portrait
    """
    
    def __init__(self, f, g, x_range=(-5, 5), y_range=(-5, 5)):
        self.f = f
        self.g = g
        self.x_range = x_range
        self.y_range = y_range
        self.fixed_points = []
        
    def system(self, state, t):
        """ODE system for numerical integration"""
        x, y = state
        return [self.f(x, y), self.g(x, y)]
    
    def jacobian(self, x, y, h=1e-6):
        """
        Compute the Jacobian matrix at point (x, y) using finite differences
        
        Returns
        -------
        J : 2x2 numpy array
            Jacobian matrix [[df/dx, df/dy], [dg/dx, dg/dy]]
        """
        # Partial derivatives using central differences
        df_dx = (self.f(x + h, y) - self.f(x - h, y)) / (2 * h)
        df_dy = (self.f(x, y + h) - self.f(x, y - h)) / (2 * h)
        dg_dx = (self.g(x + h, y) - self.g(x - h, y)) / (2 * h)
        dg_dy = (self.g(x, y + h) - self.g(x, y - h)) / (2 * h)
        
        return np.array([[df_dx, df_dy],
                        [dg_dx, dg_dy]])
    
    def classify_fixed_point(self, x, y):
        """
        Classify a fixed point based on eigenvalues of the Jacobian
        
        Returns
        -------
        classification : str
            Type of fixed point (stable node, saddle, etc.)
        eigenvalues : complex array
            Eigenvalues of the Jacobian
        """
        J = self.jacobian(x, y)
        eigenvalues = np.linalg.eigvals(J)
        
        trace = np.trace(J)
        det = np.linalg.det(J)
        
        # Classification based on eigenvalues
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)
        
        if det < 0:
            classification = "Saddle Point"
        elif det > 0:
            if trace < 0:
                if np.allclose(imag_parts, 0):
                    classification = "Stable Node"
                else:
                    classification = "Stable Spiral"
            elif trace > 0:
                if np.allclose(imag_parts, 0):
                    classification = "Unstable Node"
                else:
                    classification = "Unstable Spiral"
            else:
                classification = "Center (borderline)"
        else:
            classification = "Degenerate (det=0)"
            
        return classification, eigenvalues, J
    
    def find_fixed_points(self, num_guesses=20):
        """
        Find fixed points by solving f(x,y) = 0 and g(x,y) = 0
        Uses multiple initial guesses to find different fixed points
        """
        def equations(p):
            x, y = p
            return [self.f(x, y), self.g(x, y)]
        
        # Generate initial guesses on a grid
        x_guesses = np.linspace(self.x_range[0], self.x_range[1], int(np.sqrt(num_guesses)))
        y_guesses = np.linspace(self.y_range[0], self.y_range[1], int(np.sqrt(num_guesses)))
        
        found_points = []
        tolerance = 1e-4
        
        for x0 in x_guesses:
            for y0 in y_guesses:
                try:
                    sol = fsolve(equations, (x0, y0), full_output=True)
                    x_fp, y_fp = sol[0]
                    info = sol[1]
                    
                    # Check if solution is valid
                    if info['fvec'].dot(info['fvec']) < 1e-9:
                        # Check if point is within range
                        if (self.x_range[0] <= x_fp <= self.x_range[1] and 
                            self.y_range[0] <= y_fp <= self.y_range[1]):
                            
                            # Check if we already found this point
                            is_new = True
                            for existing in found_points:
                                if np.sqrt((x_fp - existing[0])**2 + (y_fp - existing[1])**2) < tolerance:
                                    is_new = False
                                    break
                            
                            if is_new:
                                found_points.append((x_fp, y_fp))
                except:
                    pass
        
        self.fixed_points = found_points
        return found_points
    
    def plot_phase_portrait(self, num_trajectories=10, figsize=(14, 10)):
        """
        Create a comprehensive phase portrait
        
        Parameters
        ----------
        num_trajectories : int
            Number of sample trajectories to plot
        figsize : tuple
            Figure size (width, height)
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create grid for vector field and nullclines
        x = np.linspace(self.x_range[0], self.x_range[1], 30)
        y = np.linspace(self.y_range[0], self.y_range[1], 30)
        X, Y = np.meshgrid(x, y)
        
        # Compute vector field
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                U[i, j] = self.f(X[i, j], Y[i, j])
                V[i, j] = self.g(X[i, j], Y[i, j])
        
        # Normalize vectors for better visualization
        M = np.sqrt(U**2 + V**2)
        M[M == 0] = 1  # Avoid division by zero
        U_norm = U / M
        V_norm = V / M

        # Plot vector field with uniform arrow color (e.g., dark blue)
        ax.quiver(X, Y, U_norm, V_norm, color='navy', alpha=0.2, scale=35)
        
        # Plot nullclines
        # x-nullcline: where dx/dt = 0
        x_fine = np.linspace(self.x_range[0], self.x_range[1], 200)
        y_fine = np.linspace(self.y_range[0], self.y_range[1], 200)
        X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
        
        U_fine = np.zeros_like(X_fine)
        V_fine = np.zeros_like(Y_fine)
        for i in range(X_fine.shape[0]):
            for j in range(X_fine.shape[1]):
                U_fine[i, j] = self.f(X_fine[i, j], Y_fine[i, j])
                V_fine[i, j] = self.g(X_fine[i, j], Y_fine[i, j])
        
        ax.contour(X_fine, Y_fine, U_fine, levels=[0], colors='red', linewidths=2, 
                   linestyles='--', label='x-nullcline (dx/dt=0)')
        ax.contour(X_fine, Y_fine, V_fine, levels=[0], colors='green', linewidths=2, 
                   linestyles='--', label='y-nullcline (dy/dt=0)')
        
        # Find and plot fixed points
        print("\n" + "="*70)
        print("FIXED POINT ANALYSIS")
        print("="*70)
        
        self.find_fixed_points()
        
        if len(self.fixed_points) == 0:
            print("No fixed points found in the specified range.")
        
        for i, (x_fp, y_fp) in enumerate(self.fixed_points):
            classification, eigenvalues, J = self.classify_fixed_point(x_fp, y_fp)
            
            print(f"\nFixed Point #{i+1}: ({x_fp:.4f}, {y_fp:.4f})")
            print(f"Classification: {classification}")
            print(f"Jacobian matrix:")
            print(f"  [{J[0,0]:8.4f}  {J[0,1]:8.4f}]")
            print(f"  [{J[1,0]:8.4f}  {J[1,1]:8.4f}]")
            print(f"Eigenvalues: {eigenvalues[0]:.4f}, {eigenvalues[1]:.4f}")
            print(f"Trace: {np.trace(J):.4f}")
            print(f"Determinant: {np.linalg.det(J):.4f}")
            
            # Plot fixed point with appropriate marker
            if "Stable" in classification:
                marker = 'o'
                color = 'blue'
                size = 150
            elif "Unstable" in classification:
                marker = 'o'
                color = 'red'
                size = 150
            elif "Saddle" in classification:
                marker = 'X'
                color = 'orange'
                size = 200
            else:
                marker = 's'
                color = 'purple'
                size = 150
            
            ax.scatter(x_fp, y_fp, c=color, marker=marker, s=size, 
                      edgecolors='black', linewidths=2, zorder=5,
                      label=f'FP{i+1}: {classification}')
        
        # Plot sample trajectories
        t = np.linspace(0, 10, 1000)
        
        # Generate initial conditions
        np.random.seed(42)
        x0_vals = np.random.uniform(self.x_range[0], self.x_range[1], num_trajectories)
        y0_vals = np.random.uniform(self.y_range[0], self.y_range[1], num_trajectories)
        
        for x0, y0 in zip(x0_vals, y0_vals):
            try:
                # Forward integration
                sol = odeint(self.system, [x0, y0], t)
                ax.plot(sol[:, 0], sol[:, 1], 'gray', alpha=0.4, linewidth=1)
                
                # Add arrow to show direction
                mid = len(sol) // 2
                dx = sol[mid+1, 0] - sol[mid, 0]
                dy = sol[mid+1, 1] - sol[mid, 1]
                ax.arrow(sol[mid, 0], sol[mid, 1], dx, dy, 
                        head_width=0.15, head_length=0.1, fc='gray', ec='gray', alpha=0.5)
            except:
                pass
        
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.set_title('Phase Portrait', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(self.x_range)
        ax.set_ylim(self.y_range)
        
        # Create custom legend
        handles, labels = ax.get_legend_handles_labels()
        # Remove duplicate nullcline labels
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
        
        ax.legend(unique_handles, unique_labels, loc='best', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('phase_portrait.png', dpi=300, bbox_inches='tight')
        print("\n" + "="*70)
        print("Phase portrait saved to: phase_portrait.png")
        print("="*70 + "\n")
        
        return fig, ax


# ============================================================================
# EXAMPLE SYSTEMS
# ============================================================================

def example_1_predator_prey():
    """Lotka-Volterra predator-prey model"""
    print("\n" + "="*70)
    print("EXAMPLE 1: PREDATOR-PREY MODEL (Lotka-Volterra)")
    print("="*70)
    print("System: dx/dt = x(1 - y)")
    print("        dy/dt = y(x - 1)")
    print("Where x = prey population, y = predator population")
    
    def f(x, y):
        return x * (1 - y)
    
    def g(x, y):
        return y * (x - 1)
    
    portrait = PhasePortrait2D(f, g, x_range=(0, 3), y_range=(0, 3))
    portrait.plot_phase_portrait(num_trajectories=15)
    plt.show()


def example_2_competing_species():
    """Competing species model"""
    print("\n" + "="*70)
    print("EXAMPLE 2: COMPETING SPECIES MODEL")
    print("="*70)
    print("System: dx/dt = x(3 - x - 2y)")
    print("        dy/dt = y(2 - x - y)")
    
    def f(x, y):
        return x * (3 - x - 2*y)
    
    def g(x, y):
        return y * (2 - x - y)
    
    portrait = PhasePortrait2D(f, g, x_range=(-0.5, 4), y_range=(-0.5, 3))
    portrait.plot_phase_portrait(num_trajectories=15)
    plt.show()


def example_3_van_der_pol():
    """Van der Pol oscillator"""
    print("\n" + "="*70)
    print("EXAMPLE 3: VAN DER POL OSCILLATOR")
    print("="*70)
    print("System: dx/dt = y")
    print("        dy/dt = μ(1 - x²)y - x")
    print("Where μ = 1.0 (damping parameter)")
    
    mu = 1.0
    
    def f(x, y):
        return y
    
    def g(x, y):
        return mu * (1 - x**2) * y - x
    
    portrait = PhasePortrait2D(f, g, x_range=(-4, 4), y_range=(-4, 4))
    portrait.plot_phase_portrait(num_trajectories=12)
    plt.show()


def example_4_linear_saddle():
    """Simple linear system with a saddle point"""
    print("\n" + "="*70)
    print("EXAMPLE 4: LINEAR SYSTEM WITH SADDLE POINT")
    print("="*70)
    print("System: dx/dt = x + y")
    print("        dy/dt = 4x - 2y")
    
    def f(x, y):
        return x + y
    
    def g(x, y):
        return 4*x - 2*y
    
    portrait = PhasePortrait2D(f, g, x_range=(-3, 3), y_range=(-3, 3))
    portrait.plot_phase_portrait(num_trajectories=15)
    plt.show()


def strogatz_613():
    def f(x, y):
        return x**2 - x*y
    
    def g(x, y):
        return 2*y*x - y**2

    portrait = PhasePortrait2D(f, g, x_range=(-3, 3), y_range=(-3, 3))
    portrait.plot_phase_portrait(num_trajectories=15)
    plt.show()

def strogatz_618():
    def f(x, y):
        return y
    
    def g(x, y):
        return y * (1 - x**2) - x

    portrait = PhasePortrait2D(f, g, x_range=(-3, 3), y_range=(-3, 3))
    portrait.plot_phase_portrait(num_trajectories=15)
    plt.show()

def strogatz_641():
    def f(x, y):
        return x * (3 - x - y)
    
    def g(x, y):
        return y  * (2 - x - y)

    portrait = PhasePortrait2D(f, g, x_range=(-1, 5), y_range=(-1, 5))
    portrait.plot_phase_portrait(num_trajectories=15)
    plt.show()

def hw_5_2(u=0.1):
    def f(x, y):
        return y
    
    def g(x, y):
        return -x - u * (x**2 -1) * y
    

    portrait = PhasePortrait2D(f, g, x_range=(-3, 3), y_range=(-3, 3))
    portrait.plot_phase_portrait(num_trajectories=15)
    plt.show()

def hw_5_3():
    def f(x, y):
        return (0.14 -x) - (x - 1) - y + 0.112
    
    def g(x, y):
        return 0.01 * (x- 2.54 * y)

    portrait = PhasePortrait2D(f, g, x_range=(-3, 3), y_range=(-3, 3))
    portrait.plot_phase_portrait(num_trajectories=15)
    plt.show()

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Run examples or define your own system"""
    
    # print("\n")
    # print("╔" + "="*68 + "╗")
    # print("║" + " "*15 + "2D PHASE PORTRAIT GENERATOR" + " "*26 + "║")
    # print("╚" + "="*68 + "╝")
    
    # # Run examples
    # example_1_predator_prey()
    # example_2_competing_species()
    # example_3_van_der_pol()
    # example_4_linear_saddle()
    
    # print("\n" + "="*70)
    # print("TO USE WITH YOUR OWN SYSTEM:")
    # print("="*70)
    # print("1. Define your functions f(x,y) and g(x,y)")
    # print("2. Create a PhasePortrait2D object:")
    # print("   portrait = PhasePortrait2D(f, g, x_range=(-5,5), y_range=(-5,5))")
    # print("3. Generate the plot:")
    # print("   portrait.plot_phase_portrait()")
    # print("="*70 + "\n")

    hw_5_3()

if __name__ == "__main__":
    main()