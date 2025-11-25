# Test subthreshold responses with different currents
I_values = [0, 2, 4, 6]  # µA/cm²
T = 50
n_steps = int(T / dt)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, I_amp in enumerate(I_values):
    hh_neuron.reset()
    V_trace = []
    t_trace = []
    
    for step in range(n_steps):
        t = step * dt
        hh_neuron.step(dt, I_amp)
        V_trace.append(hh_neuron.V)
        t_trace.append(t)
    
    axes[idx].plot(t_trace, V_trace, linewidth=2)
    axes[idx].set_xlabel('Time (ms)')
    axes[idx].set_ylabel('V (mV)')
    axes[idx].set_title(f'I = {I_amp} µA/cm²')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()