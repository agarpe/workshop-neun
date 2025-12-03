import neun_py
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Simulation parameters
# -----------------------------
dt = 0.01
T = 5000
time = np.arange(0, T, dt)
n_steps = len(time)

# Base currents
I_regular = 2.5      # Regular HR
I_chaotic = 3.2       # Chaotic HR

# -----------------------------
# HR neuron helpers
# -----------------------------
def configure_hr(neuron):
    P = neun_py.HRDoubleParameter
    neuron.set_param(P.e, 0)
    neuron.set_param(P.mu, 0.006)
    neuron.set_param(P.S, 4)
    neuron.set_param(P.a, 1)
    neuron.set_param(P.b, 3)
    neuron.set_param(P.c, 1)
    neuron.set_param(P.d, 5)
    neuron.set_param(P.xr, -1.6)
    neuron.set_param(P.vh, 1)

def set_initial_conditions(neuron):
    V = neun_py.HRDoubleVariable
    neuron.set(V.x, -0.712841)
    neuron.set(V.y, -1.93688)
    neuron.set(V.z, 3.16568)

def simulate_hr(I_array):
    """Simulate HR neuron with a given input array"""
    neuron = neun_py.HRDoubleRK4(neun_py.HRDoubleConstructorArgs())
    configure_hr(neuron)
    set_initial_conditions(neuron)

    V_trace, y_trace, z_trace = [], [], []
    for I_t in I_array:
        neuron.add_synaptic_input(I_t)
        neuron.step(dt)
        V_trace.append(neuron.get(neun_py.HRDoubleVariable.x))
        y_trace.append(neuron.get(neun_py.HRDoubleVariable.y))
        z_trace.append(neuron.get(neun_py.HRDoubleVariable.z))

    return np.array(V_trace), np.array(y_trace), np.array(z_trace)

# -----------------------------
# Generate input currents
# -----------------------------
np.random.seed(42)
I_regular_clean = np.ones(n_steps) * I_regular
I_chaotic_array = np.ones(n_steps) * I_chaotic

# -----------------------------
# Run simulations
# -----------------------------
V_clean, y_clean, z_clean = simulate_hr(I_regular_clean)
V_chaotic, y_chaotic, z_chaotic = simulate_hr(I_chaotic_array)

# -----------------------------
# Plot membrane potentials
# -----------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

axes[0].plot(time, V_clean, color='steelblue', linewidth=1)
axes[0].set_title("HR Regular — Clean Input")
axes[0].set_ylabel("Membrane Potential (x)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(time, V_chaotic, color='darkred', linewidth=1)
axes[1].set_title("HR Chaotic")
axes[1].set_xlabel("Time (ms)")
axes[1].set_ylabel("Membrane Potential (x)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# -----------------------------
# Phase plane comparisons (x-y)
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 5))
axes[0].plot(V_clean, y_clean, color='steelblue', linewidth=0.7)
axes[0].set_title("Regular Clean: Phase Space (x-y)")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y"); axes[0].grid(True, alpha=0.3)

axes[1].plot(V_chaotic, y_chaotic, color='darkred', linewidth=0.7)
axes[1].set_title("Chaotic: Phase Space (x-y)")
axes[1].set_xlabel("x"); axes[1].set_ylabel("y"); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# -----------------------------
# Phase plane comparisons (x-z)
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 5))
axes[0].plot(V_clean, z_clean, color='steelblue', linewidth=0.7)
axes[0].set_title("Regular Clean: Phase Space (x-z)")
axes[0].set_xlabel("x"); axes[0].set_ylabel("z"); axes[0].grid(True, alpha=0.3)

axes[1].plot(V_chaotic, z_chaotic, color='darkred', linewidth=0.7)
axes[1].set_title("Chaotic: Phase Space (x-z)")
axes[1].set_xlabel("x"); axes[1].set_ylabel("z"); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# -----------------------------
# 3D phase space comparison
# -----------------------------
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(18, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(V_clean, y_clean, z_clean, color='steelblue', linewidth=0.5, alpha=0.8)
    ax1.set_title("Regular Clean 3D Phase Space")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(V_chaotic, y_chaotic, z_chaotic, color='darkred', linewidth=0.5, alpha=0.8)
    ax2.set_title("Chaotic 3D Phase Space")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")

    plt.tight_layout()
    plt.show()
except ImportError:
    print("3D plotting not available; skipping 3D phase space.")
