import numpy as np
import matplotlib.pyplot as plt
import neun_py

# ============================
# Crear neurona HH
# ============================
args = neun_py.HHDoubleConstructorArgs()
neuron = neun_py.HHDoubleRK4(args)

# Parámetros biofísicos
neuron.set_param(neun_py.HHDoubleParameter.cm, 1.0 * 7.854e-3)
neuron.set_param(neun_py.HHDoubleParameter.vna, 50.0)
neuron.set_param(neun_py.HHDoubleParameter.vk, -77.0)
neuron.set_param(neun_py.HHDoubleParameter.vl, -54.387)
neuron.set_param(neun_py.HHDoubleParameter.gna, 120 * 7.854e-3)
neuron.set_param(neun_py.HHDoubleParameter.gk, 36 * 7.854e-3)
neuron.set_param(neun_py.HHDoubleParameter.gl, 0.3 * 7.854e-3)

# Función para resetear neurona (neun_py no tiene reset() nativo)
def reset_neuron():
    neuron.set(neun_py.HHDoubleVariable.v, -80.0)
    neuron.set(neun_py.HHDoubleVariable.m, 0.1)
    neuron.set(neun_py.HHDoubleVariable.h, 0.01)
    neuron.set(neun_py.HHDoubleVariable.n, 0.7)

# ============================
# Parámetros de simulación
# ============================
dt = 0.001  # ms
T = 500     # ms para estimación estable de frecuencia
n_steps = int(T / dt)

# ============================
# Rango de corrientes
# ============================
I_range = np.linspace(0, 30, 30)
firing_rates = []

# ============================
# Cálculo de la curva F–I
# ============================
for I_amp in I_range:
    reset_neuron()
    spike_count = 0
    prev_V = neuron.get(neun_py.HHDoubleVariable.v)

    for step in range(n_steps):
        neuron.add_synaptic_input(I_amp)
        neuron.step(dt)

        V = neuron.get(neun_py.HHDoubleVariable.v)

        # Detectar spikes por cruce hacia arriba
        if prev_V < 0 and V >= 0:
            spike_count += 1

        prev_V = V

    firing_rate = (spike_count / (T / 1000))  # Hz
    firing_rates.append(firing_rate)

# ============================
# Plot de la curva F–I
# ============================
plt.figure(figsize=(10, 6))
plt.plot(I_range, firing_rates, 'o-', linewidth=2, markersize=6)
plt.xlabel('Corriente de entrada (µA/cm²)', fontsize=12)
plt.ylabel('Frecuencia de disparo (Hz)', fontsize=12)
plt.title('Curva F–I del Neurona Hodgkin–Huxley (neun_py)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
