"""
Spherical Pendulum – Runge–Kutta Methods Comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import os

# ----------------------------------------------------------------------
# Physical system
# ----------------------------------------------------------------------

class SphericalPendulum:
    """
    Spherical pendulum with mass m, length l, gravity g.
    State vector y = [theta, omega_theta, phi, omega_phi].
    """
    def __init__(self, g=9.81, l=1.0, m=1.0):
        self.g = g
        self.l = l
        self.m = m

    def rhs(self, t, y):
        """Right-hand side of the first-order ODE system."""
        th, w_th, ph, w_ph = y

        # Safe cotangent: avoid division by zero when th is small
        sin_th = np.sin(th)
        cos_th = np.cos(th)
        # Use limit cot(th) = cos(th)/sin(th); if sin(th) is tiny

        if abs(sin_th) < 1e-14:
            cot_th = 0.0
        else:
            cot_th = cos_th / sin_th

        dth = w_th
        dw_th = sin_th * cos_th * w_ph**2 - (self.g / self.l) * sin_th
        dph = w_ph
        dw_ph = -2.0 * cot_th * w_th * w_ph

        return np.array([dth, dw_th, dph, dw_ph])

    def energy(self, y):
        """Total mechanical energy (per unit mass, m=1)."""
        th, w_th, ph, w_ph = y
        T = 0.5 * self.l**2 * (w_th**2 + (np.sin(th)**2) * w_ph**2)
        V = -self.g * self.l * np.cos(th)
        return T + V

# ----------------------------------------------------------------------
# Runge–Kutta steppers
# ----------------------------------------------------------------------

class RungeKuttaIntegrator(SphericalPendulum):
    """
    Extends SphericalPendulum with step methods for ERK3, DIRK3, IRK3, RK4.
    """
    def __init__(self, g=9.81, l=1.0, m=1.0):
        super().__init__(g, l, m)
        self.h = None       # current step size (set before stepping)

    # -------------------- ERK3 (explicit) --------------------
    def step_erk3(self, t, y, h):
        """
        Classical 3rd-order explicit Runge–Kutta (Heun's 3rd-order).
        Returns (t_new, y_new).
        """
        k1 = self.rhs(t, y)
        k2 = self.rhs(t + 0.5*h, y + 0.5*h*k1)
        k3 = self.rhs(t + h, y + h*(-k1 + 2*k2))
        y_new = y + (h/6.0) * (k1 + 4*k2 + k3)
        return t + h, y_new

    # -------------------- DIRK3 (diagonally implicit) --------------------
    def _dirk3_stage(self, t, y, h, c, a_diag, sum_prev, max_iter=10, tol=1e-12):
        """
        Solve a single DIRK stage k = f(t + c*h, y + h*(sum_prev + a_diag*k))
        using fixed-point iteration.
        """
        k = np.zeros_like(y)
        for _ in range(max_iter):
            arg = y + h * (sum_prev + a_diag * k)
            k_new = self.rhs(t + c * h, arg)
            if np.linalg.norm(k_new - k) < tol:
                return k_new
            k = k_new
        # If not converged, return last estimate
        return k

    def step_dirk3(self, t, y, h):
        """
        Alexander's 3-stage DIRK3 (diagonally implicit).
        Returns (t_new, y_new).
        """
        gamma = 1.0 - np.sqrt(2.0)/2.0
        c1 = gamma
        c2 = (1.0 + gamma) / 2.0
        c3 = 1.0

        a21 = (1.0 - gamma) / 2.0
        a31 = 1.0/(4.0*gamma) - 0.5
        a32 = a31   # same coefficient

        # Stage 1
        k1 = self._dirk3_stage(t, y, h, c1, gamma, np.zeros_like(y))
        # Stage 2
        sum2 = a21 * k1
        k2 = self._dirk3_stage(t, y, h, c2, gamma, sum2)
        # Stage 3
        sum3 = a31 * k1 + a32 * k2
        k3 = self._dirk3_stage(t, y, h, c3, gamma, sum3)

        # Update: b coefficients = [a31, a32, gamma]
        y_new = y + h * (a31*k1 + a32*k2 + gamma*k3)
        return t + h, y_new

    # -------------------- IRK3 (fully implicit) --------------------
    def _irk3_system(self, K_flat, t, y, h):
        """
        Nonlinear system for IRK3 stages: F(K) = 0.
        K = [k1, k2] (each is 4‑D).
        """
        k1 = K_flat[:4]
        k2 = K_flat[4:]
        # Radau IIA coefficients
        A11, A12 = 5.0/12.0, -1.0/12.0
        A21, A22 = 3.0/4.0,  1.0/4.0
        c1, c2 = 1.0/3.0, 1.0

        arg1 = y + h * (A11*k1 + A12*k2)
        arg2 = y + h * (A21*k1 + A22*k2)
        F1 = k1 - self.rhs(t + c1*h, arg1)
        F2 = k2 - self.rhs(t + c2*h, arg2)
        return np.concatenate([F1, F2])

    def step_irk3(self, t, y, h):
        """
        Fully implicit 2‑stage Radau IIA (3rd‑order, A‑stable).
        Uses scipy.optimize.fsolve to solve the coupled system.
        """
        K0 = np.zeros(8)  # initial guess
        sol = fsolve(self._irk3_system, K0, args=(t, y, h), xtol=1e-12)
        k1, k2 = sol[:4], sol[4:]
        # Update: b = [3/4, 1/4]
        y_new = y + h * (0.75*k1 + 0.25*k2)
        return t + h, y_new

    # -------------------- RK4 (explicit, 4th-order) --------------------
    def step_rk4(self, t, y, h):
        k1 = self.rhs(t, y)
        k2 = self.rhs(t + 0.5*h, y + 0.5*h*k1)
        k3 = self.rhs(t + 0.5*h, y + 0.5*h*k2)
        k4 = self.rhs(t + h, y + h*k3)
        y_new = y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return t + h, y_new

# ----------------------------------------------------------------------
# Simulation wrapper
# ----------------------------------------------------------------------

def simulate(pendulum, method, h, t_f, y0, step_func):
    """
    Run simulation from t=0 to t_f with fixed step h.
    Returns times array and history of y.
    """
    t = 0.0
    y = y0.copy()
    times = [t]
    y_hist = [y.copy()]

    while t < t_f - 1e-12:
        t, y = step_func(t, y, h)
        times.append(t)
        y_hist.append(y.copy())

    return np.array(times), np.array(y_hist)

# ----------------------------------------------------------------------
# Analysis functions (to be completed)
# ----------------------------------------------------------------------

def plot_trajectories(times_hist, y_hist_dict, labels, title="Trajectories"):
    """
    Plot θ(t) and φ(t) for all methods on the same figure.
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for name, (t, yh) in y_hist_dict.items():
        plt.plot(t, np.degrees(yh[:, 0]), label=name)
    plt.xlabel('Time (s)')
    plt.ylabel('θ (degrees)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for name, (t, yh) in y_hist_dict.items():
        plt.plot(t, np.degrees(yh[:, 2]), label=name)
    plt.xlabel('Time (s)')
    plt.ylabel('φ (degrees)')
    plt.legend()
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('trajectories.pdf', dpi=150)
    plt.show()

def plot_errors_over_time(times_ref, y_ref, y_hist_dict, labels):
    """
    Compute and plot ||y(t) - y_ref(t)||_2 vs time.
    """
    # Interpolate reference at the simulation time points
    from scipy.interpolate import interp1d
    ref_interp = interp1d(times_ref, y_ref, axis=0, kind='linear',
                          bounds_error=False, fill_value='extrapolate')

    plt.figure(figsize=(10, 6))
    for name, (t, yh) in y_hist_dict.items():
        y_ref_at_t = ref_interp(t)
        err = np.linalg.norm(yh - y_ref_at_t, axis=1)
        plt.semilogy(t, err, label=name)

    plt.xlabel('Time (s)')
    plt.ylabel('Global error (2‑norm)')
    plt.legend()
    plt.grid(True)
    plt.title('Error vs. time')
    plt.savefig('error_vs_time.pdf', dpi=150)
    plt.show()

def convergence_order(h_list, errors_dict, labels):
    """
    Plot log(error) vs log(h) and estimate slopes.
    """
    plt.figure(figsize=(8, 6))
    for name, errs in errors_dict.items():
        plt.loglog(h_list, errs, 'o-', label=name)

    plt.xlabel('Step size h')
    plt.ylabel('Global error at t_f')
    plt.legend()
    plt.grid(True)
    plt.title('Convergence (order estimation)')
    plt.savefig('convergence.pdf', dpi=150)
    plt.show()

    # Estimate slopes (linear regression in log-log)
    log_h = np.log(h_list)
    for name, errs in errors_dict.items():
        if np.all(np.isfinite(errs)):
            log_err = np.log(errs)
            slope, intercept = np.polyfit(log_h, log_err, 1)
            print(f"{name}: estimated order = {slope:.3f}")

def energy_conservation(times_hist, y_hist_dict, pendulum, labels):
    """
    Plot relative energy error (E(t)-E(0))/|E(0)| vs time.
    """
    plt.figure(figsize=(10, 6))
    for name, (t, yh) in y_hist_dict.items():
        E0 = pendulum.energy(yh[0])
        E = np.array([pendulum.energy(y) for y in yh])
        err_rel = (E - E0) / abs(E0)
        plt.semilogy(t, np.abs(err_rel), label=name)

    plt.xlabel('Time (s)')
    plt.ylabel('|Relative energy error|')
    plt.legend()
    plt.grid(True)
    plt.title('Energy conservation')
    plt.savefig('energy_error.pdf', dpi=150)
    plt.show()

def singularity_test(pendulum, h_list, method_step, method_name):
    """
    Test RK4 (or any explicit method) near theta ≈ 0.
    Run with decreasing h and observe blow-up.
    """
    th0 = 1e-6
    w_th0 = 0.0
    ph0 = 0.0
    w_ph0 = 1.0
    y0 = np.array([th0, w_th0, ph0, w_ph0])

    t_f = 1.0
    print(f"\n--- Singularity test: {method_name} near θ=0 ---")
    for h in h_list:
        times, y_hist = simulate(pendulum, method_name, h, t_f, y0, method_step)
        # Check if solution remains finite
        if np.any(np.isnan(y_hist)) or np.any(np.isinf(y_hist)):
            print(f"  h = {h:.4f} : solution diverged (NaN/Inf)")
        else:
            # Check max theta: it should remain close to initial
            max_th = np.max(np.abs(y_hist[:, 0]))
            print(f"  h = {h:.4f} : max |θ| = {max_th:.2e}")

# ----------------------------------------------------------------------
# Main script
# ----------------------------------------------------------------------

def main():
    # Physical parameters
    g = 9.81 # kg . m . s^(-2)
    l = 1.0  # m 
    m = 1.0  # kg
    pendulum = RungeKuttaIntegrator(g, l, m)

    # Initial conditions (as in assignment)
    th0 = np.pi / 2.0
    w_th0 = 0.0
    ph0 = 0.0
    w_ph0 = 2.0   # rad/s
    y0 = np.array([th0, w_th0, ph0, w_ph0])

    t_f = 10.0   # final time

    # Methods to test
    methods = {
        'ERK3': pendulum.step_erk3,
        'DIRK3': pendulum.step_dirk3,
        'IRK3': pendulum.step_irk3,
        'RK4': pendulum.step_rk4,
    }

    # Trajectory comparison at fixed h
    h_fixed = 0.01
    y_hist_dict = {}
    for name, step_func in methods.items():
        t, yh = simulate(pendulum, name, h_fixed, t_f, y0, step_func)
        y_hist_dict[name] = (t, yh)

    plot_trajectories(t, y_hist_dict, methods.keys(), title=f"Trajectories (h={h_fixed})")

    # Reference solution (high accuracy)
    print("\nComputing reference solution with solve_ivp (DOP853)...")
    def rhs_wrapper(t, y):
        return pendulum.rhs(t, y)
    sol = solve_ivp(rhs_wrapper, [0, t_f], y0, method='DOP853',
                    rtol=1e-12, atol=1e-12, dense_output=True)
    t_ref = sol.t
    y_ref = sol.y.T   # shape (n_points, 4)

    # Error vs time (for each method, using h_fixed)
    plot_errors_over_time(t_ref, y_ref, y_hist_dict, methods.keys())

    #Convergence order
    h_list = [0.1, 0.05, 0.025, 0.01]
    errors = {name: [] for name in methods}

    # Evaluate reference at final time
    y_ref_final = sol.y[:, -1]   # at t_f

    for h in h_list:
        print(f"\nRunning h = {h:.4f} ...")
        for name, step_func in methods.items():
            t, yh = simulate(pendulum, name, h, t_f, y0, step_func)
            err = np.linalg.norm(yh[-1] - y_ref_final)
            errors[name].append(err)
            print(f"  {name}: error = {err:.6e}")

    convergence_order(h_list, errors, methods.keys())

    # Energy conservation (using h_fixed)
    energy_conservation(t, y_hist_dict, pendulum, methods.keys())

    # Singularity test for RK4 (explicit method)
    h_sing = [0.01, 0.005, 0.001, 0.0005]
    singularity_test(pendulum, h_sing, pendulum.step_rk4, 'RK4')

    print("\nAll plots saved. Done.")

if __name__ == "__main__":
    main()
