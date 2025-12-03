import matplotlib.pyplot as plt
import neun_py

# Dictionary of Izhikevich parameters for different cell types
neuron_types = {
    'Regular Spiking (RS)': {
        'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8,
        'I_amp': 10, 'color': 'blue'
    },
    # You could add more neuron types here
}

# Simulate and plot
dt = 0.1
T = 1000
n_steps = int(T / dt)

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (name, params) in enumerate(neuron_types.items()):
    # Create Izhikevich neuron
    args = neun_py.IzDoubleConstructorArgs()
    neuron = neun_py.IzDoubleRK4(args)

    # Set parameters
    neuron.set_param(neun_py.IzDoubleParameter.a, params['a'])
    neuron.set_param(neun_py.IzDoubleParameter.b, params['b'])
    neuron.set_param(neun_py.IzDoubleParameter.c, params['c'])
    neuron.set_param(neun_py.IzDoubleParameter.d, params['d'])

    # Set initial conditions (you can change them if you want)
    neuron.set(neun_py.IzDoubleVariable.v, -65.0)
    neuron.set(neun_py.IzDoubleVariable.u, params['b'] * -65.0)
    
    V_trace = []
    t_trace = []
    
    for step in range(n_steps):
        t = step * dt
        neuron.add_synaptic_input(params['I_amp'])
        neuron.step(dt)
        V_trace.append(neuron.get(neun_py.IzDoubleVariable.v))
        t_trace.append(t)
    
    axes[idx].plot(t_trace, V_trace, color=params['color'], linewidth=1.5)
    axes[idx].set_title(name, fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('V (mV)')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_ylim([-80, 40])

axes[-2].set_xlabel('Time (ms)')
axes[-1].set_xlabel('Time (ms)')
plt.tight_layout()
plt.show()