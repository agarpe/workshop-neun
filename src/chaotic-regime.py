import neun_py
import numpy as np
import matplotlib.pyplot as plt

# Solo el modo caótico
I_chaotic = 3.2

dt = 0.01
T = 5000
time = np.arange(0, T, dt)

# Crear neurona Hindmarsh-Rose
args = neun_py.HRDoubleConstructorArgs()
neuron = neun_py.HRDoubleRK4(args)

# Parámetros del modelo
neuron.set_param(neun_py.HRDoubleParameter.e, 0)
neuron.set_param(neun_py.HRDoubleParameter.mu, 0.006)
neuron.set_param(neun_py.HRDoubleParameter.S, 4)
neuron.set_param(neun_py.HRDoubleParameter.a, 1)
neuron.set_param(neun_py.HRDoubleParameter.b, 3)
neuron.set_param(neun_py.HRDoubleParameter.c, 1)
neuron.set_param(neun_py.HRDoubleParameter.d, 5)
neuron.set_param(neun_py.HRDoubleParameter.xr, -1.6)
neuron.set_param(neun_py.HRDoubleParameter.vh, 1)

# Condiciones iniciales
neuron.set(neun_py.HRDoubleVariable.x, -0.712841)
neuron.set(neun_py.HRDoubleVariable.y, -1.93688)
neuron.set(neun_py.HRDoubleVariable.z, 3.16568)

# Simulación
V = []
for t in time:
    neuron.add_synaptic_input(I_chaotic)
    neuron.step(dt)
    V.append(neuron.get(neun_py.HRDoubleVariable.x))

# Plot
plt.figure(figsize=(12, 4))
plt.plot(time, V, 'b-', linewidth=1)
plt.title(f"Hindmarsh-Rose Neuron — Chaotic Mode (I = {I_chaotic})")
plt.ylabel("Membrane Potential")
plt.xlabel("Time (ms)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
