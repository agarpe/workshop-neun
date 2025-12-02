import neun_py
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------
dt = 0.01
T = 5000
time = np.arange(0, T, dt)
n_steps = len(time)

# ---------------------------------------------------------
# HR Parameter Setup
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# HR Simulation Function
# ---------------------------------------------------------
def simulate_HR(I_array):
    neuron = neun_py.HRDoubleRK4(neun_py.HRDoubleConstructorArgs())
    configure_hr(neuron)
    set_initial_conditions(neuron)

    V_trace = []
    for k in range(n_steps):
        neuron.add_synaptic_input(I_array[k])
        neuron.step(dt)
        V_trace.append(neuron.get(neun_py.HRDoubleVariable.x))

    return np.array(V_trace)

# ---------------------------------------------------------
# Peak-based Spike Detection (for HR model)
# ---------------------------------------------------------
def compute_ISI_stats_HR(V, threshold=0.0):
    """Detect spikes as local maxima in the HR x-variable."""
    spike_times = []

    for i in range(1, len(V)-1):
        if V[i] > threshold and V[i] > V[i-1] and V[i] > V[i+1]:
            # Avoid double-counting
            if len(spike_times) == 0 or (i*dt - spike_times[-1]) > 5:
                spike_times.append(i * dt)

    if len(spike_times) < 2:
        return None, None

    ISIs = np.diff(spike_times)
    CV = np.std(ISIs) / np.mean(ISIs)

    return ISIs, CV

# ---------------------------------------------------------
# Generate currents
# ---------------------------------------------------------
I_base_regular = 2.5
noise_std = 0.5

np.random.seed(42)
I_clean = np.ones(n_steps) * I_base_regular
I_noisy = I_base_regular + np.random.randn(n_steps) * noise_std

# Chaotic HR input
I_chaotic = 3.2
I_chaotic_array = np.ones(n_steps) * I_chaotic

# ---------------------------------------------------------
# Simulations
# ---------------------------------------------------------
V_clean = simulate_HR(I_clean)
V_noisy = simulate_HR(I_noisy)
V_chaotic = simulate_HR(I_chaotic_array)

# ---------------------------------------------------------
# Compute ISI statistics
# ---------------------------------------------------------
ISIs_clean, CV_clean = compute_ISI_stats_HR(V_clean)
ISIs_noisy, CV_noisy = compute_ISI_stats_HR(V_noisy)
ISIs_ch, CV_ch = compute_ISI_stats_HR(V_chaotic)

print("\n--- CV VALUES ---")
print(f"HR Regular (clean):   CV = {CV_clean:.3f}")
print(f"HR Regular (noisy):   CV = {CV_noisy:.3f}")
print(f"HR Chaotic:           CV = {CV_ch:.3f}")

# ---------------------------------------------------------
# Plot Voltage traces
# ---------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

axes[0].plot(time, V_clean, linewidth=1.1)
axes[0].set_title("HR Regular — Clean Input")
axes[0].grid(True, alpha=0.3)

axes[1].plot(time, V_noisy, color="orange", linewidth=1.1)
axes[1].set_title("HR Regular — Noisy Input")
axes[1].grid(True, alpha=0.3)

axes[2].plot(time, V_chaotic, color="red", linewidth=1.1)
axes[2].set_title("HR Chaotic Mode")
axes[2].set_xlabel("Time (ms)")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Plot ISI Distributions
# ---------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.hist(ISIs_clean, bins=40, alpha=0.5, label="HR Clean", density=True)
plt.hist(ISIs_noisy, bins=40, alpha=0.5, label="HR Noisy", density=True)
plt.hist(ISIs_ch, bins=40, alpha=0.5, label="HR Chaotic", density=True)
plt.xlabel("Inter-Spike Interval (ms)")
plt.ylabel("Probability Density")
plt.title("ISI Distributions: Clean vs Noisy vs Chaotic HR")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
